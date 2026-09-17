#include "locotrackinference.h"
#include "csvmetadata.h"

#include <QDir>
#include <QFileInfo>
#include <QStringList>

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
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kModelResolution = 256;
constexpr int kMaximumPoints = 64;
constexpr std::array<int, 6> kFrameProfiles{64, 32, 16, 8, 4, 2};

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

void clearCudaCache()
{
    c10::cuda::CUDACachingAllocator::emptyCache();
}

struct VideoData
{
    torch::Tensor frames;
    int width = 0;
    int height = 0;
    int declaredFrames = 0;
    double fps = 25.0;
};

VideoData loadVideo(const QString &videoPath,
                    const InferenceProgress &progress,
                    const InferenceCancelCheck &isCanceled)
{
    cv::VideoCapture capture(openCvPath(videoPath));
    if (!capture.isOpened()) {
        throw std::runtime_error("OpenCV 无法打开 LocoTrack 输入视频");
    }

    VideoData result;
    result.width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    result.height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    result.declaredFrames = static_cast<int>(
        capture.get(cv::CAP_PROP_FRAME_COUNT));
    result.fps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(result.fps) || result.fps <= 0.0) {
        result.fps = 25.0;
    }
    if (result.width <= 0 || result.height <= 0) {
        throw std::runtime_error("LocoTrack 输入视频分辨率无效");
    }

    constexpr std::size_t bytesPerFrame =
        static_cast<std::size_t>(kModelResolution * kModelResolution * 3);
    std::vector<std::uint8_t> pixels;
    if (result.declaredFrames > 0) {
        pixels.reserve(bytesPerFrame
                       * static_cast<std::size_t>(result.declaredFrames));
    }

    cv::Mat frame;
    int frameCount = 0;
    while (capture.read(frame)) {
        checkCanceled(isCanceled);
        cv::Mat resized;
        cv::resize(frame, resized,
                   cv::Size(kModelResolution, kModelResolution), 0.0, 0.0,
                   cv::INTER_AREA);
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        if (!resized.isContinuous()) {
            resized = resized.clone();
        }
        const auto *begin = resized.ptr<std::uint8_t>();
        pixels.insert(pixels.end(), begin, begin + bytesPerFrame);
        ++frameCount;

        if (progress && (frameCount == 1 || frameCount % 50 == 0)) {
            const int percent = result.declaredFrames > 0
                ? std::clamp(frameCount * 10 / result.declaredFrames, 0, 10)
                : 0;
            progress(percent,
                     QStringLiteral("LocoTrack 读取并缩放 %1/%2 帧")
                         .arg(frameCount)
                         .arg(result.declaredFrames > 0
                                  ? result.declaredFrames : frameCount));
        }
    }
    capture.release();
    if (frameCount == 0) {
        throw std::runtime_error("LocoTrack 输入视频没有可用帧");
    }

    result.frames = torch::from_blob(
                        pixels.data(),
                        {1, frameCount, kModelResolution,
                         kModelResolution, 3},
                        torch::TensorOptions().dtype(torch::kUInt8))
                        .clone();
    return result;
}

QString profileModelPath(const QString &directory, int profile)
{
    return QDir(directory).filePath(
        QStringLiteral("locotrack_base_t%1_torchscript.pt").arg(profile));
}

std::vector<int> availableProfiles(const QString &directory)
{
    std::vector<int> profiles;
    for (const int profile : kFrameProfiles) {
        if (QFileInfo::exists(profileModelPath(directory, profile))) {
            profiles.push_back(profile);
        }
    }
    return profiles;
}

torch::jit::script::Module loadProfileModel(const QString &directory,
                                            int profile,
                                            const torch::Device &device)
{
    const QString path = profileModelPath(directory, profile);
    std::ifstream stream(fileSystemPath(path), std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            QStringLiteral("无法读取 LocoTrack 固定 %1 帧模型：%2")
                .arg(profile)
                .arg(QDir::toNativeSeparators(path))
                .toUtf8()
                .constData());
    }
    torch::jit::script::Module model = torch::jit::load(stream, device);
    model.eval();
    return model;
}

torch::Tensor prepareQueries(const std::vector<cv::Point2f> &points,
                             int width,
                             int height,
                             const torch::Device &device)
{
    std::vector<float> values(points.size() * 3, 0.0F);
    for (std::size_t index = 0; index < points.size(); ++index) {
        values[index * 3 + 1] =
            points[index].y / height * kModelResolution;
        values[index * 3 + 2] =
            points[index].x / width * kModelResolution;
    }
    return torch::from_blob(
               values.data(), {1, static_cast<int64_t>(points.size()), 3},
               torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .to(device);
}

bool isCudaOutOfMemory(const c10::Error &error)
{
    std::string message = error.what_without_backtrace();
    std::transform(message.begin(), message.end(), message.begin(),
                   [](unsigned char character) {
                       return static_cast<char>(std::tolower(character));
                   });
    return message.find("cuda") != std::string::npos
        && message.find("out of memory") != std::string::npos;
}

struct SegmentResult
{
    torch::Tensor tracks;
    torch::Tensor occluded;
};

SegmentResult runSegment(torch::jit::script::Module &model,
                         const torch::Tensor &video,
                         const torch::Tensor &queries,
                         int width,
                         int height,
                         const torch::Device &device)
{
    const torch::Tensor cudaVideo = video.contiguous().to(device);
    const torch::Tensor cudaQueries = queries.to(device);

    torch::jit::IValue value;
    {
        c10::InferenceMode inferenceMode;
        value = model.forward({cudaVideo, cudaQueries});
    }
    if (!value.isTuple()) {
        throw std::runtime_error("LocoTrack TorchScript 没有返回 tuple");
    }
    const auto outputs = value.toTuple()->elements();
    if (outputs.size() != 2 || !outputs[0].isTensor()
        || !outputs[1].isTensor()) {
        throw std::runtime_error("LocoTrack TorchScript 输出数量或类型异常");
    }
    if (!outputs[0].toTensor().is_cuda()
        || !outputs[1].toTensor().is_cuda()) {
        throw std::runtime_error("LocoTrack TorchScript 输出不在 CUDA");
    }

    torch::Tensor tracks = outputs[0].toTensor()
                               .select(0, 0)
                               .to(torch::TensorOptions().device(torch::kCPU)
                                       .dtype(torch::kFloat32))
                               .contiguous();
    torch::Tensor occluded = outputs[1].toTensor()
                                 .select(0, 0)
                                 .to(torch::TensorOptions().device(torch::kCPU)
                                         .dtype(torch::kBool))
                                 .contiguous();
    if (tracks.dim() != 3 || tracks.size(2) != 2
        || occluded.dim() != 2
        || tracks.size(0) != occluded.size(0)
        || tracks.size(1) != occluded.size(1)) {
        throw std::runtime_error("LocoTrack tracks/occluded 输出形状异常");
    }
    tracks.select(2, 0).mul_(static_cast<double>(width)
                             / kModelResolution);
    tracks.select(2, 1).mul_(static_cast<double>(height)
                             / kModelResolution);
    if (!torch::isfinite(tracks).all().item<bool>()) {
        throw std::runtime_error(
            "LocoTrack FP32 输出包含 NaN 或 Inf，已停止生成错误结果");
    }
    return {tracks, occluded};
}

int chooseProfile(const std::vector<int> &profiles,
                  int maximumIndex,
                  int remainingFrames)
{
    if (remainingFrames >= profiles[maximumIndex]) {
        return maximumIndex;
    }
    const int requiredFrames = std::max(2, remainingFrames);
    int selected = -1;
    for (int index = maximumIndex;
         index < static_cast<int>(profiles.size()); ++index) {
        if (profiles[index] >= requiredFrames) {
            selected = index;
        }
    }
    return selected >= 0 ? selected : static_cast<int>(profiles.size()) - 1;
}

struct FullTracks
{
    torch::Tensor tracks;
    torch::Tensor occluded;
};

FullTracks runSegmented(const torch::Tensor &video,
                        const QVector<QPointF> &selectedPoints,
                        int width,
                        int height,
                        const QString &modelDirectory,
                        const std::vector<int> &profiles,
                        const InferenceProgress &progress,
                        const InferenceCancelCheck &isCanceled)
{
    using torch::indexing::Slice;

    const int totalFrames = static_cast<int>(video.size(1));
    const torch::Device device(torch::kCUDA, 0);
    int segmentStart = 0;
    int maximumProfileIndex = 0;
    int loadedProfile = 0;
    int segmentCount = 0;
    int oomCount = 0;
    std::optional<torch::jit::script::Module> model;
    std::vector<cv::Point2f> currentPoints;
    currentPoints.reserve(static_cast<std::size_t>(selectedPoints.size()));
    for (const QPointF &point : selectedPoints) {
        currentPoints.emplace_back(static_cast<float>(point.x()),
                                   static_cast<float>(point.y()));
    }
    std::vector<torch::Tensor> trackParts;
    std::vector<torch::Tensor> occlusionParts;

    while (segmentStart < totalFrames) {
        checkCanceled(isCanceled);
        const int remainingFrames = totalFrames - segmentStart;
        int attemptProfileIndex = chooseProfile(
            profiles, maximumProfileIndex, remainingFrames);
        SegmentResult segment;
        int realSegmentFrames = 0;
        int segmentEnd = 0;

        while (true) {
            checkCanceled(isCanceled);
            const int profile = profiles[attemptProfileIndex];
            realSegmentFrames = std::min(profile, remainingFrames);
            segmentEnd = segmentStart + realSegmentFrames;
            torch::Tensor segmentVideo = video.index(
                {Slice(), Slice(segmentStart, segmentEnd)});
            if (realSegmentFrames < profile) {
                const torch::Tensor padding = segmentVideo.index(
                    {Slice(), Slice(realSegmentFrames - 1,
                                    realSegmentFrames)})
                                                   .repeat(
                                                       {1,
                                                        profile
                                                            - realSegmentFrames,
                                                        1, 1, 1});
                segmentVideo = torch::cat({segmentVideo, padding}, 1);
            }

            if (!model || loadedProfile != profile) {
                model.reset();
                clearCudaCache();
                if (progress) {
                    progress(
                        std::clamp(10 + segmentStart * 65 / totalFrames,
                                   10, 75),
                        QStringLiteral("正在加载 LocoTrack-B FP32 固定 %1 帧模型……")
                            .arg(profile));
                }
                model.emplace(loadProfileModel(modelDirectory, profile,
                                               device));
                loadedProfile = profile;
            }

            const torch::Tensor queries = prepareQueries(
                currentPoints, width, height, device);
            if (progress) {
                progress(
                    std::clamp(10 + segmentStart * 65 / totalFrames, 10, 75),
                    QStringLiteral(
                        "LocoTrack-B FP32 推理全局帧 %1~%2（固定 %3 帧档位）")
                        .arg(segmentStart)
                        .arg(segmentEnd - 1)
                        .arg(profile));
            }

            try {
                segment = runSegment(*model, segmentVideo, queries,
                                     width, height, device);
            } catch (const c10::Error &error) {
                if (!isCudaOutOfMemory(error)) {
                    throw;
                }
                ++oomCount;
                model.reset();
                loadedProfile = 0;
                clearCudaCache();
                if (attemptProfileIndex + 1
                    >= static_cast<int>(profiles.size())) {
                    throw std::runtime_error(
                        "即使固定 2 帧 LocoTrack-B 模型仍然显存不足；"
                        "请关闭其他 GPU 程序或减少跟踪点");
                }
                ++attemptProfileIndex;
                maximumProfileIndex = std::max(maximumProfileIndex,
                                                attemptProfileIndex);
                if (progress) {
                    progress(
                        std::clamp(10 + segmentStart * 65 / totalFrames,
                                   10, 75),
                        QStringLiteral(
                            "第 %1 次显存不足，自动切换到固定 %2 帧模型……")
                            .arg(oomCount)
                            .arg(profiles[attemptProfileIndex]));
                }
                continue;
            }

            segment.tracks = segment.tracks.index(
                {Slice(), Slice(0, realSegmentFrames), Slice()});
            segment.occluded = segment.occluded.index(
                {Slice(), Slice(0, realSegmentFrames)});
            ++segmentCount;
            break;
        }

        if (segmentEnd == totalFrames) {
            trackParts.push_back(segment.tracks);
            occlusionParts.push_back(segment.occluded);
            segmentStart = totalFrames;
            break;
        }

        int continuationLocal = -1;
        int visibleCount = -1;
        for (int frame = 1; frame < realSegmentFrames; ++frame) {
            const int count = static_cast<int>(
                segment.occluded.select(1, frame).logical_not().sum().item<int64_t>());
            if (count == static_cast<int>(currentPoints.size())) {
                continuationLocal = frame;
                visibleCount = count;
            }
        }
        bool allVisible = continuationLocal >= 0;
        if (!allVisible) {
            for (int frame = 1; frame < realSegmentFrames; ++frame) {
                const int count = static_cast<int>(
                    segment.occluded.select(1, frame)
                        .logical_not()
                        .sum()
                        .item<int64_t>());
                if (count >= visibleCount) {
                    continuationLocal = frame;
                    visibleCount = count;
                }
            }
        }
        if (continuationLocal <= 0) {
            throw std::runtime_error("LocoTrack 无法选择下一段续接帧");
        }

        trackParts.push_back(segment.tracks.index(
            {Slice(), Slice(0, continuationLocal), Slice()}));
        occlusionParts.push_back(segment.occluded.index(
            {Slice(), Slice(0, continuationLocal)}));
        const torch::Tensor continuationPoints =
            segment.tracks.select(1, continuationLocal).contiguous();
        const auto accessor = continuationPoints.accessor<float, 2>();
        for (int64_t point = 0; point < continuationPoints.size(0); ++point) {
            currentPoints[static_cast<std::size_t>(point)] =
                cv::Point2f(accessor[point][0], accessor[point][1]);
        }

        segmentStart += continuationLocal;
        if (progress) {
            progress(
                std::clamp(10 + segmentStart * 65 / totalFrames, 10, 75),
                allVisible
                    ? QStringLiteral(
                          "LocoTrack 第 %1 段完成；从全局帧 %2（全部点可见）续接")
                          .arg(segmentCount)
                          .arg(segmentStart)
                    : QStringLiteral(
                          "LocoTrack 第 %1 段完成；从全局帧 %2（最多 %3/%4 点可见）续接")
                          .arg(segmentCount)
                          .arg(segmentStart)
                          .arg(visibleCount)
                          .arg(currentPoints.size()));
        }
        clearCudaCache();
    }

    model.reset();
    clearCudaCache();
    torch::Tensor tracks = torch::cat(trackParts, 1).contiguous();
    torch::Tensor occluded = torch::cat(occlusionParts, 1).contiguous();
    if (tracks.size(1) != totalFrames || occluded.size(1) != totalFrames) {
        throw std::runtime_error(
            QStringLiteral("LocoTrack 分段合并后得到 %1 帧，但输入共有 %2 帧")
                .arg(tracks.size(1))
                .arg(totalFrames)
                .toUtf8()
                .constData());
    }
    return {tracks, occluded};
}

void writeResults(const QString &videoPath,
                  const QString &outputVideo,
                  const QString &outputCsv,
                  const torch::Tensor &tracks,
                  const torch::Tensor &occluded,
                  double fps,
                  int width,
                  int height,
                  const InferenceProgress &progress,
                  const InferenceCancelCheck &isCanceled)
{
    const int pointCount = static_cast<int>(tracks.size(0));
    const int frameCount = static_cast<int>(tracks.size(1));
    if (tracks.dim() != 3 || tracks.size(2) != 2
        || occluded.dim() != 2 || occluded.size(0) != pointCount
        || occluded.size(1) != frameCount) {
        throw std::runtime_error("LocoTrack 轨迹和遮挡输出形状不一致");
    }

    cv::VideoCapture capture(openCvPath(videoPath));
    if (!capture.isOpened()) {
        throw std::runtime_error("无法重新打开视频以写入 LocoTrack 结果");
    }
    cv::VideoWriter writer(openCvPath(outputVideo),
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                           cv::Size(width, height));
    if (!writer.isOpened()) {
        throw std::runtime_error("无法创建 LocoTrack 标注视频");
    }
    std::ofstream csv(fileSystemPath(outputCsv),
                      std::ios::binary | std::ios::trunc);
    if (!csv) {
        throw std::runtime_error("无法创建 LocoTrack 原始坐标 CSV");
    }
    writeCsvSourceVideoPath(csv, videoPath);
    writeCsvProcessedVideoPath(csv, outputVideo);
    csv << "frame_index,timestamp_sec";
    for (int id = 1; id <= pointCount; ++id) {
        csv << ",id_" << id << "_x_px,id_" << id << "_y_px";
    }
    csv << '\n' << std::fixed;

    const auto trackAccessor = tracks.accessor<float, 3>();
    const auto occlusionAccessor = occluded.accessor<bool, 2>();
    cv::Mat frame;
    for (int frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
        checkCanceled(isCanceled);
        if (!capture.read(frame)) {
            throw std::runtime_error(
                QStringLiteral("写入 LocoTrack 结果时无法读取第 %1 帧")
                    .arg(frameIndex)
                    .toUtf8()
                    .constData());
        }

        csv << frameIndex << ',' << std::setprecision(9)
            << frameIndex / fps;
        for (int point = 0; point < pointCount; ++point) {
            const float x = trackAccessor[point][frameIndex][0];
            const float y = trackAccessor[point][frameIndex][1];
            const bool visible = !occlusionAccessor[point][frameIndex];
            if (visible) {
                csv << ',' << std::setprecision(6) << x << ',' << y;
            } else {
                csv << ",,";
            }

            const cv::Point drawPoint(cvRound(x), cvRound(y));
            const cv::Scalar color = visible
                ? cv::Scalar(0, 0, 255)
                : cv::Scalar(0, 255, 255);
            cv::circle(frame, drawPoint, 6, color, visible ? -1 : 2,
                       cv::LINE_AA);
            cv::putText(frame, std::to_string(point + 1),
                        cv::Point(drawPoint.x + 8, drawPoint.y - 8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                        cv::LINE_AA);
        }
        csv << '\n';
        writer.write(frame);

        if (progress && (frameIndex == 0 || (frameIndex + 1) % 10 == 0
                         || frameIndex + 1 == frameCount)) {
            const int percent = 75
                + std::clamp((frameIndex + 1) * 25 / frameCount, 0, 25);
            progress(percent,
                     QStringLiteral("正在写入 LocoTrack 结果 %1/%2 帧")
                         .arg(frameIndex + 1)
                         .arg(frameCount));
        }
    }
}

} // namespace

InferenceResult LocoTrackInference::run(
    const QString &videoPath,
    const QString &modelDirectory,
    const QVector<QPointF> &points,
    const InferenceProgress &progress,
    const InferenceCancelCheck &isCanceled)
{
    if (points.isEmpty()) {
        throw std::runtime_error("LocoTrack 至少需要一个首帧跟踪点");
    }
    if (points.size() > kMaximumPoints) {
        throw std::runtime_error("LocoTrack 最多支持 64 个跟踪点");
    }
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("LibTorch CUDA 不可用，已拒绝回退 CPU");
    }

    const std::vector<int> profiles = availableProfiles(modelDirectory);
    if (profiles.empty()) {
        throw std::runtime_error(
            QStringLiteral("在 %1 中找不到任何 LocoTrack-B 固定帧模型")
                .arg(QDir::toNativeSeparators(modelDirectory))
                .toUtf8()
                .constData());
    }
    checkCanceled(isCanceled);
    if (progress) {
        QStringList profileNames;
        for (const int profile : profiles) {
            profileNames << QString::number(profile);
        }
        progress(0,
                 QStringLiteral("LocoTrack-B FP32 可用固定帧档位：%1")
                     .arg(profileNames.join(QStringLiteral(", "))));
    }

    const VideoData video = loadVideo(videoPath, progress, isCanceled);
    for (const QPointF &point : points) {
        if (point.x() < 0.0 || point.x() >= video.width
            || point.y() < 0.0 || point.y() >= video.height) {
            throw std::runtime_error("LocoTrack 首帧跟踪点超出视频范围");
        }
    }
    const FullTracks result = runSegmented(
        video.frames, points, video.width, video.height, modelDirectory,
        profiles, progress, isCanceled);
    checkCanceled(isCanceled);

    const QFileInfo inputInfo(videoPath);
    const QString outputVideo = QDir(inputInfo.absolutePath()).filePath(
        inputInfo.completeBaseName()
        + QStringLiteral("_locotrack_base_torchscript_annotated.mp4"));
    const QString outputCsv = QDir(inputInfo.absolutePath()).filePath(
        inputInfo.completeBaseName()
        + QStringLiteral("_locotrack_base_torchscript_raw.csv"));
    writeResults(videoPath, outputVideo, outputCsv, result.tracks,
                 result.occluded, video.fps, video.width, video.height,
                 progress, isCanceled);
    checkCanceled(isCanceled);
    if (progress) {
        progress(100, QStringLiteral("LocoTrack-B 通用点跟踪完成"));
    }
    return {outputVideo, outputCsv};
}
