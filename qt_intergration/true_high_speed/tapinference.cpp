#include "tapinference.h"
#include "csvmetadata.h"

#include <QDir>
#include <QFileInfo>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// Qt 会把 slots 定义成宏，与 LibTorch 的 slots 标识符冲突。
#ifdef slots
#pragma push_macro("slots")
#undef slots
#define THS_RESTORE_QT_SLOTS_MACRO
#endif

#include <torch/cuda.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/script.h>
#include <torch/torch.h>

#ifdef THS_RESTORE_QT_SLOTS_MACRO
#pragma pop_macro("slots")
#undef THS_RESTORE_QT_SLOTS_MACRO
#endif

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kInputResolution = 512;
constexpr float kModelCoordinateSize = 256.0F;

std::string openCvPath(const QString &path)
{
    return path.toLocal8Bit().toStdString();
}

std::filesystem::path fileSystemPath(const QString &path)
{
#ifdef _WIN32
    return std::filesystem::path(path.toStdWString());
#else
    return std::filesystem::path(path.toStdString());
#endif
}

void checkCanceled(const InferenceCancelCheck &isCanceled)
{
    if (isCanceled && isCanceled()) {
        throw InferenceCanceled();
    }
}

torch::Tensor prepareVideoTensor(const cv::Mat &frame,
                                 const torch::Device &device)
{
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, rgb, cv::Size(kInputResolution, kInputResolution));
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 127.5, -1.0);
    if (!rgb.isContinuous()) {
        rgb = rgb.clone();
    }

    return torch::from_blob(
               rgb.data,
               {1, 1, kInputResolution, kInputResolution, 3},
               torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .to(device);
}

torch::Tensor prepareQueries(const QVector<QPointF> &points,
                             int width,
                             int height,
                             const torch::Device &device)
{
    std::vector<float> queryData(
        static_cast<std::size_t>(points.size() * 3), 0.0F);
    for (int index = 0; index < points.size(); ++index) {
        queryData[static_cast<std::size_t>(index * 3 + 1)] =
            static_cast<float>(points[index].y() / height)
            * kModelCoordinateSize;
        queryData[static_cast<std::size_t>(index * 3 + 2)] =
            static_cast<float>(points[index].x() / width)
            * kModelCoordinateSize;
    }
    return torch::from_blob(
               queryData.data(), {1, points.size(), 3},
               torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .to(device);
}

std::vector<torch::jit::IValue> tupleElements(torch::jit::IValue value,
                                              const char *methodName)
{
    if (!value.isTuple()) {
        throw std::runtime_error(
            std::string("TorchScript ") + methodName
            + " 没有返回 tuple");
    }
    return value.toTuple()->elements();
}

std::vector<torch::jit::IValue> initialize(
    torch::jit::script::Module &model,
    const torch::Tensor &video,
    const torch::Tensor &queries)
{
    return tupleElements(
        model.get_method("initialize")({video, queries}), "initialize");
}

std::vector<torch::jit::IValue> step(
    torch::jit::script::Module &model,
    const torch::Tensor &video,
    const std::vector<torch::jit::IValue> &previousOutputs)
{
    if (previousOutputs.size() < 5) {
        throw std::runtime_error("TAPNext++ 递归状态输出数量不足");
    }
    std::vector<torch::jit::IValue> inputs;
    inputs.reserve(previousOutputs.size() - 2);
    inputs.emplace_back(video);
    inputs.insert(inputs.end(), previousOutputs.begin() + 3,
                  previousOutputs.end());
    return tupleElements(model.get_method("step")(inputs), "step");
}

void assertCudaOutputs(const std::vector<torch::jit::IValue> &outputs)
{
    for (std::size_t index = 0; index < outputs.size(); ++index) {
        if (!outputs[index].isTensor()) {
            throw std::runtime_error(
                "TAPNext++ 输出/状态中出现非 Tensor 值");
        }
        const torch::Tensor tensor = outputs[index].toTensor();
        if (index == 3) {
            if (tensor.dim() != 0 || tensor.scalar_type() != torch::kInt64) {
                throw std::runtime_error(
                    "TAPNext++ outputs[3] 不是 int64 零维 step");
            }
            continue;
        }
        if (!tensor.is_cuda()) {
            std::ostringstream message;
            message << "TAPNext++ 输出/状态 outputs[" << index
                    << "] 不在 CUDA";
            throw std::runtime_error(message.str());
        }
    }
}

struct DecodedTapFrame
{
    std::vector<cv::Point2f> positions;
    std::vector<bool> visible;
};

DecodedTapFrame decodeOutputs(
    const std::vector<torch::jit::IValue> &outputs,
    int width,
    int height)
{
    if (outputs.size() < 3) {
        throw std::runtime_error("TAPNext++ 主输出数量不足");
    }
    torch::Tensor tracks = outputs[0].toTensor()
                               .select(0, 0)
                               .select(0, 0)
                               .to(torch::TensorOptions()
                                       .device(torch::kCPU)
                                       .dtype(torch::kFloat32))
                               .contiguous();
    torch::Tensor visibility = outputs[2].toTensor()
                                   .select(0, 0)
                                   .select(0, 0)
                                   .to(torch::TensorOptions()
                                           .device(torch::kCPU)
                                           .dtype(torch::kFloat32))
                                   .contiguous();
    if (visibility.dim() == 2) {
        visibility = visibility.select(1, 0).contiguous();
    }
    if (tracks.dim() != 2 || tracks.size(1) != 2
        || visibility.dim() != 1
        || visibility.size(0) != tracks.size(0)) {
        throw std::runtime_error("TAPNext++ tracks/visible_logits 形状异常");
    }

    DecodedTapFrame decoded;
    decoded.positions.reserve(static_cast<std::size_t>(tracks.size(0)));
    decoded.visible.reserve(static_cast<std::size_t>(tracks.size(0)));
    const auto trackAccessor = tracks.accessor<float, 2>();
    const auto visibilityAccessor = visibility.accessor<float, 1>();
    for (int64_t index = 0; index < tracks.size(0); ++index) {
        // 模型 tracks 顺序为 [y, x]。
        const float x = trackAccessor[index][1]
            / kModelCoordinateSize * width;
        const float y = trackAccessor[index][0]
            / kModelCoordinateSize * height;
        decoded.positions.emplace_back(x, y);
        decoded.visible.push_back(visibilityAccessor[index] > 0.0F);
    }
    return decoded;
}

cv::Mat drawPoints(const cv::Mat &frame, const DecodedTapFrame &decoded)
{
    cv::Mat result = frame.clone();
    for (std::size_t index = 0; index < decoded.positions.size(); ++index) {
        const cv::Point point(cvRound(decoded.positions[index].x),
                              cvRound(decoded.positions[index].y));
        const bool visible = decoded.visible[index];
        const cv::Scalar color = visible
            ? cv::Scalar(0, 0, 255)
            : cv::Scalar(0, 255, 255);
        cv::circle(result, point, 6, color, visible ? -1 : 2,
                   cv::LINE_AA);
        cv::putText(result, std::to_string(index + 1),
                    cv::Point(point.x + 8, point.y - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    cv::LINE_AA);
    }
    return result;
}

void writeCsvHeader(std::ofstream &stream,
                    const QString &videoPath,
                    const QString &processedVideoPath,
                    int pointCount)
{
    writeCsvSourceVideoPath(stream, videoPath);
    writeCsvProcessedVideoPath(stream, processedVideoPath);
    stream << "frame_index,video_timestamp_sec";
    for (int id = 1; id <= pointCount; ++id) {
        stream << ",id_" << id << "_x_px,id_" << id << "_y_px";
    }
    stream << '\n';
    stream << std::fixed << std::setprecision(9);
}

void writeCsvRow(std::ofstream &stream,
                 int frameIndex,
                 double timestamp,
                 const DecodedTapFrame &decoded)
{
    stream << frameIndex << ',' << timestamp;
    for (std::size_t index = 0; index < decoded.positions.size(); ++index) {
        if (!decoded.visible[index]) {
            stream << ",,";
        } else {
            stream << ',' << decoded.positions[index].x
                   << ',' << decoded.positions[index].y;
        }
    }
    stream << '\n';
}

} // namespace

InferenceResult TapInference::run(const QString &videoPath,
                                  const QString &modelPath,
                                  const QVector<QPointF> &points,
                                  const InferenceProgress &progress,
                                  const InferenceCancelCheck &isCanceled)
{
    if (points.isEmpty()) {
        throw std::runtime_error("TAPNext++ 至少需要一个首帧跟踪点");
    }
    if (!QFileInfo::exists(modelPath)) {
        throw std::runtime_error(
            QStringLiteral("找不到 TAPNext++ 模型：%1")
                .arg(QDir::toNativeSeparators(modelPath))
                .toUtf8()
                .constData());
    }
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("LibTorch CUDA 不可用，已拒绝回退 CPU");
    }
    checkCanceled(isCanceled);

    const torch::Device device(torch::kCUDA, 0);
    if (progress) {
        progress(0, QStringLiteral("正在加载 TAPNext++ TorchScript……"));
    }

    std::ifstream modelStream(fileSystemPath(modelPath), std::ios::binary);
    if (!modelStream) {
        throw std::runtime_error("无法读取 TAPNext++ TorchScript 文件");
    }
    torch::jit::script::Module model = torch::jit::load(modelStream, device);
    model.eval();
    for (const torch::Tensor &parameter : model.parameters()) {
        if (!parameter.is_cuda()) {
            throw std::runtime_error("TAPNext++ 模型参数未全部加载到 CUDA");
        }
    }
    model = torch::jit::freeze_module(
        model, {"initialize", "step"});

    cv::VideoCapture capture(openCvPath(videoPath));
    if (!capture.isOpened()) {
        throw std::runtime_error("OpenCV 无法打开 TAPNext++ 输入视频");
    }
    int totalFrames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 0.0) {
        fps = 25.0;
    }
    cv::Mat firstFrame;
    if (width <= 0 || height <= 0 || !capture.read(firstFrame)
        || firstFrame.empty()) {
        throw std::runtime_error("无法读取 TAPNext++ 视频第一帧");
    }

    for (const QPointF &point : points) {
        if (point.x() < 0.0 || point.x() >= width
            || point.y() < 0.0 || point.y() >= height) {
            throw std::runtime_error("首帧跟踪点超出原始视频坐标范围");
        }
    }

    const torch::Tensor queries = prepareQueries(points, width, height, device);
    if (!queries.is_cuda()) {
        throw std::runtime_error("TAPNext++ query_points 不在 CUDA");
    }

    checkCanceled(isCanceled);
    const QFileInfo inputInfo(videoPath);
    const QString outputVideo = QDir(inputInfo.absolutePath()).filePath(
        inputInfo.completeBaseName() + QStringLiteral("_tap_annotated.mp4"));
    const QString outputCsv = QDir(inputInfo.absolutePath()).filePath(
        inputInfo.completeBaseName() + QStringLiteral("_tap_raw.csv"));
    cv::VideoWriter writer(openCvPath(outputVideo),
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                           cv::Size(width, height));
    if (!writer.isOpened()) {
        throw std::runtime_error("无法创建 TAPNext++ 标注视频");
    }
    std::ofstream csv(fileSystemPath(outputCsv),
                      std::ios::binary | std::ios::trunc);
    if (!csv) {
        throw std::runtime_error("无法创建 TAPNext++ 原始坐标 CSV");
    }
    writeCsvHeader(csv, videoPath, outputVideo, points.size());

    int processedFrames = 0;
    cv::Mat frame = firstFrame;
    std::vector<torch::jit::IValue> outputs;
    {
        c10::InferenceMode inferenceMode;
        while (!frame.empty()) {
            checkCanceled(isCanceled);
            const torch::Tensor video = prepareVideoTensor(frame, device);
            outputs = outputs.empty()
                ? initialize(model, video, queries)
                : step(model, video, outputs);
            assertCudaOutputs(outputs);

            const DecodedTapFrame decoded = decodeOutputs(outputs, width, height);
            if (decoded.positions.size()
                != static_cast<std::size_t>(points.size())) {
                throw std::runtime_error(
                    "TAPNext++ 动态输出点数与用户选择点数不一致");
            }
            writer.write(drawPoints(frame, decoded));
            writeCsvRow(csv, processedFrames, processedFrames / fps, decoded);
            ++processedFrames;

            if (progress && (processedFrames == 1
                             || processedFrames % 5 == 0)) {
                const int percent = totalFrames > 0
                    ? std::clamp(processedFrames * 100 / totalFrames, 0, 100)
                    : 0;
                progress(percent,
                         QStringLiteral("TAPNext++ 跟踪 %1/%2 帧")
                             .arg(processedFrames)
                             .arg(totalFrames > 0
                                      ? totalFrames : processedFrames));
            }

            if (!capture.read(frame)) {
                break;
            }
        }
    }

    capture.release();
    writer.release();
    csv.close();
    checkCanceled(isCanceled);
    if (progress) {
        progress(100, QStringLiteral("TAPNext++ 通用点跟踪完成"));
    }
    return {outputVideo, outputCsv};
}
