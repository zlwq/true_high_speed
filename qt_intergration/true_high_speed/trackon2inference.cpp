#include "trackon2inference.h"
#include "csvmetadata.h"

#include <QDir>
#include <QFileInfo>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#ifdef slots
#pragma push_macro("slots")
#undef slots
#define THS_RESTORE_QT_SLOTS_MACRO
#endif

#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/cuda.h>
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
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kModelHeight = 384;
constexpr int kModelWidth = 512;

std::mutex &modelCacheMutex()
{
    static std::mutex mutex;
    return mutex;
}

QString &cachedModelPath()
{
    static QString path;
    return path;
}

std::shared_ptr<torch::jit::script::Module> &cachedModel()
{
    static std::shared_ptr<torch::jit::script::Module> model;
    return model;
}

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

std::vector<torch::jit::IValue> tupleElements(torch::jit::IValue value,
                                              const char *methodName)
{
    if (!value.isTuple()) {
        throw std::runtime_error(
            std::string("Track-On2 TorchScript ") + methodName
            + " 没有返回 tuple");
    }
    return value.toTuple()->elements();
}

torch::Tensor prepareFrame(const cv::Mat &frame,
                           const torch::Device &device)
{
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(kModelWidth, kModelHeight), 0.0, 0.0,
               cv::INTER_AREA);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    if (!resized.isContinuous()) {
        resized = resized.clone();
    }

    return torch::from_blob(
               resized.data, {kModelHeight, kModelWidth, 3},
               torch::TensorOptions().dtype(torch::kUInt8))
        .permute({2, 0, 1})
        .contiguous()
        .unsqueeze(0)
        .to(torch::TensorOptions().device(device).dtype(torch::kFloat32));
}

torch::Tensor prepareQueries(const QVector<QPointF> &points,
                             int width,
                             int height,
                             const torch::Device &device)
{
    std::vector<float> values(static_cast<std::size_t>(points.size() * 2));
    for (int index = 0; index < points.size(); ++index) {
        values[static_cast<std::size_t>(index * 2)] =
            static_cast<float>(points[index].x() / width) * kModelWidth;
        values[static_cast<std::size_t>(index * 2 + 1)] =
            static_cast<float>(points[index].y() / height) * kModelHeight;
    }
    return torch::from_blob(
               values.data(), {points.size(), 2},
               torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .to(device);
}

std::vector<torch::jit::IValue> initialize(
    torch::jit::script::Module &model,
    const torch::Tensor &frame,
    const torch::Tensor &queries)
{
    auto outputs = tupleElements(
        model.get_method("initialize")({frame, queries}), "initialize");
    if (outputs.size() != 5) {
        throw std::runtime_error(
            "Track-On2 initialize 输出数量不是 5");
    }
    return outputs;
}

std::vector<torch::jit::IValue> step(
    torch::jit::script::Module &model,
    const torch::Tensor &frame,
    const torch::jit::IValue &queryFeatures,
    const torch::jit::IValue &temporalMask,
    const torch::jit::IValue &pointMemory)
{
    auto outputs = tupleElements(
        model.get_method("step")(
            {frame, queryFeatures, temporalMask, pointMemory}),
        "step");
    if (outputs.size() != 4) {
        throw std::runtime_error("Track-On2 step 输出数量不是 4");
    }
    return outputs;
}

std::shared_ptr<torch::jit::script::Module> loadOrReuseModel(
    const QString &modelPath,
    const torch::Device &device,
    const InferenceProgress &progress)
{
    std::lock_guard<std::mutex> lock(modelCacheMutex());
    const QString normalizedPath = QFileInfo(modelPath).absoluteFilePath();
    if (cachedModel() && cachedModelPath() == normalizedPath) {
        if (progress) {
            progress(0, QStringLiteral(
                            "正在复用显存中的 Track-On2 模型……"));
        }
        return cachedModel();
    }

    cachedModel().reset();
    cachedModelPath().clear();
    c10::cuda::CUDACachingAllocator::emptyCache();
    if (progress) {
        progress(0, QStringLiteral(
                        "首次加载 Track-On2 TorchScript（本进程仅一次）……"));
    }
    std::ifstream modelStream(fileSystemPath(modelPath), std::ios::binary);
    if (!modelStream) {
        throw std::runtime_error("无法读取 Track-On2 TorchScript 文件");
    }
    auto model = std::make_shared<torch::jit::script::Module>(
        torch::jit::load(modelStream, device));
    model->eval();
    for (const torch::Tensor &parameter : model->parameters()) {
        if (!parameter.is_cuda()) {
            throw std::runtime_error("Track-On2 模型参数未全部加载到 CUDA");
        }
    }
    cachedModelPath() = normalizedPath;
    cachedModel() = model;
    return model;
}

struct DecodedFrame
{
    std::vector<cv::Point2f> positions;
    std::vector<bool> visible;
};

torch::Tensor removeLeadingSingletonDimensions(torch::Tensor tensor,
                                                int targetDimensions)
{
    while (tensor.dim() > targetDimensions && tensor.size(0) == 1) {
        tensor = tensor.select(0, 0);
    }
    return tensor;
}

DecodedFrame decode(const torch::jit::IValue &positionsValue,
                    const torch::jit::IValue &visibleValue,
                    int width,
                    int height)
{
    if (!positionsValue.isTensor() || !visibleValue.isTensor()) {
        throw std::runtime_error(
            "Track-On2 positions/visible 输出不是 Tensor");
    }

    torch::Tensor positions = removeLeadingSingletonDimensions(
        positionsValue.toTensor()
            .to(torch::TensorOptions().device(torch::kCPU)
                    .dtype(torch::kFloat32))
            .contiguous(),
        2);
    if (positions.dim() == 1 && positions.numel() == 2) {
        positions = positions.reshape({1, 2});
    }
    if (positions.dim() != 2 || positions.size(1) != 2) {
        throw std::runtime_error("Track-On2 positions 输出形状不是 [N,2]");
    }

    torch::Tensor visible = visibleValue.toTensor()
                                .to(torch::TensorOptions().device(torch::kCPU)
                                        .dtype(torch::kBool))
                                .reshape({-1})
                                .contiguous();
    if (visible.size(0) != positions.size(0)) {
        throw std::runtime_error(
            "Track-On2 positions 与 visible 点数不一致");
    }

    DecodedFrame decoded;
    decoded.positions.reserve(static_cast<std::size_t>(positions.size(0)));
    decoded.visible.reserve(static_cast<std::size_t>(positions.size(0)));
    const auto positionAccessor = positions.accessor<float, 2>();
    const auto visibleAccessor = visible.accessor<bool, 1>();
    for (int64_t index = 0; index < positions.size(0); ++index) {
        decoded.positions.emplace_back(
            positionAccessor[index][0] / kModelWidth * width,
            positionAccessor[index][1] / kModelHeight * height);
        decoded.visible.push_back(visibleAccessor[index]);
    }
    return decoded;
}

void draw(cv::Mat &frame, const DecodedFrame &decoded)
{
    for (std::size_t index = 0; index < decoded.positions.size(); ++index) {
        const cv::Point point(cvRound(decoded.positions[index].x),
                              cvRound(decoded.positions[index].y));
        const bool visible = decoded.visible[index];
        const cv::Scalar color = visible
            ? cv::Scalar(0, 0, 255)
            : cv::Scalar(0, 255, 255);
        cv::circle(frame, point, 6, color, visible ? -1 : 2, cv::LINE_AA);
        cv::putText(frame, std::to_string(index + 1),
                    cv::Point(point.x + 8, point.y - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
    }
}

void writeCsvHeader(std::ofstream &stream,
                    const QString &videoPath,
                    const QString &processedVideoPath,
                    int pointCount)
{
    writeCsvSourceVideoPath(stream, videoPath);
    writeCsvProcessedVideoPath(stream, processedVideoPath);
    stream << "frame_index,timestamp_sec";
    for (int id = 1; id <= pointCount; ++id) {
        stream << ",id_" << id << "_x_px,id_" << id << "_y_px";
    }
    stream << '\n';
    stream << std::fixed << std::setprecision(6);
}

void writeCsvRow(std::ofstream &stream,
                 int frameIndex,
                 double fps,
                 const DecodedFrame &decoded)
{
    stream << frameIndex << ',' << std::setprecision(9) << frameIndex / fps;
    for (std::size_t index = 0; index < decoded.positions.size(); ++index) {
        if (decoded.visible[index]) {
            stream << ',' << std::setprecision(6) << decoded.positions[index].x
                   << ',' << decoded.positions[index].y;
        } else {
            stream << ",,";
        }
    }
    stream << '\n';
}

} // namespace

void TrackOn2Inference::clearCachedModel()
{
    std::lock_guard<std::mutex> lock(modelCacheMutex());
    const bool hadCachedModel = static_cast<bool>(cachedModel());
    cachedModel().reset();
    cachedModelPath().clear();
    if (hadCachedModel) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
}

InferenceResult TrackOn2Inference::run(
    const QString &videoPath,
    const QString &modelPath,
    const QVector<QPointF> &points,
    const InferenceProgress &progress,
    const InferenceCancelCheck &isCanceled)
{
    if (points.isEmpty()) {
        throw std::runtime_error("Track-On2 至少需要一个首帧跟踪点");
    }
    if (!QFileInfo::exists(modelPath)) {
        throw std::runtime_error(
            QStringLiteral("找不到 Track-On2 模型：%1")
                .arg(QDir::toNativeSeparators(modelPath))
                .toUtf8()
                .constData());
    }
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("LibTorch CUDA 不可用，已拒绝回退 CPU");
    }
    checkCanceled(isCanceled);

    cv::VideoCapture capture(openCvPath(videoPath));
    if (!capture.isOpened()) {
        throw std::runtime_error("OpenCV 无法打开 Track-On2 输入视频");
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
        throw std::runtime_error("无法读取 Track-On2 视频第一帧");
    }
    for (const QPointF &point : points) {
        if (point.x() < 0.0 || point.x() >= width
            || point.y() < 0.0 || point.y() >= height) {
            throw std::runtime_error("Track-On2 首帧跟踪点超出视频范围");
        }
    }

    const torch::Device device(torch::kCUDA, 0);
    const auto model = loadOrReuseModel(modelPath, device, progress);

    const torch::Tensor queries = prepareQueries(points, width, height, device);
    checkCanceled(isCanceled);
    const QFileInfo inputInfo(videoPath);
    const QString outputVideo = QDir(inputInfo.absolutePath()).filePath(
        inputInfo.completeBaseName()
        + QStringLiteral("_track_on2_dinov2_annotated.mp4"));
    const QString outputCsv = QDir(inputInfo.absolutePath()).filePath(
        inputInfo.completeBaseName()
        + QStringLiteral("_track_on2_dinov2_raw.csv"));
    cv::VideoWriter writer(openCvPath(outputVideo),
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                           cv::Size(width, height));
    if (!writer.isOpened()) {
        throw std::runtime_error("无法创建 Track-On2 标注视频");
    }
    std::ofstream csv(fileSystemPath(outputCsv),
                      std::ios::binary | std::ios::trunc);
    if (!csv) {
        throw std::runtime_error("无法创建 Track-On2 原始坐标 CSV");
    }
    writeCsvHeader(csv, videoPath, outputVideo, points.size());

    capture.release();
    capture.open(openCvPath(videoPath));
    if (!capture.isOpened()) {
        throw std::runtime_error(
            "无法重新打开视频执行 Track-On2 正式推理");
    }
    int frameIndex = 0;
    cv::Mat frame;
    torch::jit::IValue queryFeatures;
    torch::jit::IValue temporalMask;
    torch::jit::IValue pointMemory;
    {
        c10::InferenceMode inferenceMode;
        while (capture.read(frame)) {
            checkCanceled(isCanceled);
            const torch::Tensor frameTensor = prepareFrame(frame, device);
            std::vector<torch::jit::IValue> outputs;
            if (frameIndex == 0) {
                outputs = initialize(*model, frameTensor, queries);
                queryFeatures = outputs[2];
                temporalMask = outputs[3];
                pointMemory = outputs[4];
            } else {
                outputs = step(*model, frameTensor, queryFeatures,
                               temporalMask, pointMemory);
                temporalMask = outputs[2];
                pointMemory = outputs[3];
            }

            const DecodedFrame decoded = decode(outputs[0], outputs[1],
                                                width, height);
            if (decoded.positions.size()
                != static_cast<std::size_t>(points.size())) {
                throw std::runtime_error(
                    "Track-On2 动态输出点数与用户选择点数不一致");
            }
            draw(frame, decoded);
            writer.write(frame);
            writeCsvRow(csv, frameIndex, fps, decoded);
            ++frameIndex;

            if (progress && (frameIndex == 1 || frameIndex % 10 == 0
                             || (totalFrames > 0
                                 && frameIndex == totalFrames))) {
                const int percent = totalFrames > 0
                    ? std::clamp(frameIndex * 100 / totalFrames, 0, 100)
                    : 0;
                progress(percent,
                         QStringLiteral("Track-On2 跟踪 %1/%2 帧")
                             .arg(frameIndex)
                             .arg(totalFrames > 0 ? totalFrames : frameIndex));
            }
        }
    }

    capture.release();
    writer.release();
    csv.close();
    if (frameIndex == 0) {
        throw std::runtime_error("Track-On2 输入视频没有可用帧");
    }
    checkCanceled(isCanceled);
    if (progress) {
        progress(100, QStringLiteral("Track-On2 通用点跟踪完成"));
    }
    return {outputVideo, outputCsv};
}
