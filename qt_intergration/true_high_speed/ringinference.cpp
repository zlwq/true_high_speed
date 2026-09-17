#include "ringinference.h"
#include "csvmetadata.h"

#include <onnxruntime_cxx_api.h>

#include <QByteArray>
#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr float kDetectionConfidence = 0.5F;
constexpr float kKeypointConfidence = 0.5F;
constexpr float kNmsIouThreshold = 0.70F;
constexpr float kBoxOverlapThreshold = 0.70F;
constexpr float kPointMinimumDistance = 7.0F;
constexpr int kContextRadius = 10;

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

Ort::Session makeSession(Ort::Env &environment,
                         const QString &modelPath,
                         const Ort::SessionOptions &options)
{
#ifdef _WIN32
    return Ort::Session(environment,
                        reinterpret_cast<const wchar_t *>(modelPath.utf16()),
                        options);
#else
    const QByteArray encoded = QFile::encodeName(modelPath);
    return Ort::Session(environment, encoded.constData(), options);
#endif
}

void ensureFileExists(const QString &path, const QString &description)
{
    if (!QFileInfo::exists(path)) {
        throw std::runtime_error(
            QStringLiteral("找不到%1：%2")
                .arg(description, QDir::toNativeSeparators(path))
                .toUtf8()
                .constData());
    }
}

void checkCanceled(const InferenceCancelCheck &isCanceled)
{
    if (isCanceled && isCanceled()) {
        throw InferenceCanceled();
    }
}

struct LetterboxResult
{
    std::vector<float> tensor;
    float scale = 1.0F;
    int left = 0;
    int top = 0;
};

LetterboxResult letterbox(const cv::Mat &frame, int imageSize)
{
    const float scale = std::min(
        static_cast<float>(imageSize) / static_cast<float>(frame.rows),
        static_cast<float>(imageSize) / static_cast<float>(frame.cols));
    const int resizedWidth = cvRound(frame.cols * scale);
    const int resizedHeight = cvRound(frame.rows * scale);

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(resizedWidth, resizedHeight));
    const int paddingWidth = imageSize - resizedWidth;
    const int paddingHeight = imageSize - resizedHeight;
    const int left = cvRound(paddingWidth / 2.0 - 0.1);
    const int right = cvRound(paddingWidth / 2.0 + 0.1);
    const int top = cvRound(paddingHeight / 2.0 - 0.1);
    const int bottom = cvRound(paddingHeight / 2.0 + 0.1);

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);
    padded.convertTo(padded, CV_32FC3, 1.0 / 255.0);

    LetterboxResult result;
    result.scale = scale;
    result.left = left;
    result.top = top;
    result.tensor.resize(static_cast<std::size_t>(3 * imageSize * imageSize));
    for (int y = 0; y < imageSize; ++y) {
        const auto *row = padded.ptr<cv::Vec3f>(y);
        for (int x = 0; x < imageSize; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                result.tensor[static_cast<std::size_t>(
                    channel * imageSize * imageSize + y * imageSize + x)] =
                    row[x][channel];
            }
        }
    }
    return result;
}

struct Detection
{
    cv::Rect2f box;
    float boxConfidence = 0.0F;
    std::optional<cv::Point2f> point;
    float pointConfidence = 0.0F;
};

float boxArea(const cv::Rect2f &box)
{
    return std::max(0.0F, box.width) * std::max(0.0F, box.height);
}

float intersectionArea(const cv::Rect2f &first, const cv::Rect2f &second)
{
    const float x1 = std::max(first.x, second.x);
    const float y1 = std::max(first.y, second.y);
    const float x2 = std::min(first.x + first.width,
                              second.x + second.width);
    const float y2 = std::min(first.y + first.height,
                              second.y + second.height);
    return std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
}

float boxIou(const cv::Rect2f &first, const cv::Rect2f &second)
{
    const float intersection = intersectionArea(first, second);
    const float unionArea = boxArea(first) + boxArea(second) - intersection;
    return unionArea <= 0.0F ? 0.0F : intersection / unionArea;
}

float smallerBoxOverlap(const cv::Rect2f &first, const cv::Rect2f &second)
{
    const float smallerArea = std::min(boxArea(first), boxArea(second));
    return smallerArea <= 0.0F
        ? 0.0F
        : intersectionArea(first, second) / smallerArea;
}

std::vector<int> nmsIndices(const std::vector<cv::Rect2f> &boxes,
                            const std::vector<float> &scores)
{
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int first, int second) {
        return scores[first] > scores[second];
    });

    std::vector<int> kept;
    while (!order.empty()) {
        const int current = order.front();
        kept.push_back(current);
        std::vector<int> remaining;
        remaining.reserve(order.size());
        for (std::size_t index = 1; index < order.size(); ++index) {
            if (boxIou(boxes[current], boxes[order[index]])
                <= kNmsIouThreshold) {
                remaining.push_back(order[index]);
            }
        }
        order.swap(remaining);
    }
    return kept;
}

std::vector<Detection> decodeRingOutput(const Ort::Value &output,
                                        const LetterboxResult &letterboxResult,
                                        int frameWidth,
                                        int frameHeight)
{
    const auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3 || shape[0] != 1) {
        throw std::runtime_error("ring_pose.onnx 输出不是预期的三维张量");
    }

    int64_t rowCount = 0;
    int64_t channelCount = 0;
    bool transposed = false;
    if (shape[2] == 8 || shape[2] == 9) {
        rowCount = shape[1];
        channelCount = shape[2];
    } else if (shape[1] == 8 || shape[1] == 9) {
        rowCount = shape[2];
        channelCount = shape[1];
        transposed = true;
    } else {
        throw std::runtime_error("无法识别 ring_pose.onnx 输出通道数");
    }

    const float *data = output.GetTensorData<float>();
    const auto valueAt = [&](int64_t row, int64_t channel) {
        return transposed
            ? data[channel * rowCount + row]
            : data[row * channelCount + channel];
    };

    std::vector<cv::Rect2f> candidateBoxes;
    std::vector<float> candidateScores;
    std::vector<std::array<float, 3>> candidateKeypoints;

    for (int64_t row = 0; row < rowCount; ++row) {
        const float score = valueAt(row, 4);
        if (score < kDetectionConfidence) {
            continue;
        }

        cv::Rect2f box;
        std::array<float, 3> keypoint{};
        if (channelCount == 9) {
            const int classId = cvRound(valueAt(row, 5));
            if (classId != 0) {
                continue;
            }
            const float x1 = valueAt(row, 0);
            const float y1 = valueAt(row, 1);
            const float x2 = valueAt(row, 2);
            const float y2 = valueAt(row, 3);
            box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
            keypoint = {valueAt(row, 6), valueAt(row, 7), valueAt(row, 8)};
        } else {
            const float centerX = valueAt(row, 0);
            const float centerY = valueAt(row, 1);
            const float width = valueAt(row, 2);
            const float height = valueAt(row, 3);
            box = cv::Rect2f(centerX - width / 2.0F,
                             centerY - height / 2.0F, width, height);
            keypoint = {valueAt(row, 5), valueAt(row, 6), valueAt(row, 7)};
        }
        candidateBoxes.push_back(box);
        candidateScores.push_back(score);
        candidateKeypoints.push_back(keypoint);
    }

    std::vector<int> keptIndices(candidateBoxes.size());
    std::iota(keptIndices.begin(), keptIndices.end(), 0);
    if (channelCount == 8) {
        keptIndices = nmsIndices(candidateBoxes, candidateScores);
    }

    std::vector<Detection> decoded;
    decoded.reserve(keptIndices.size());
    for (const int index : keptIndices) {
        const cv::Rect2f &modelBox = candidateBoxes[index];
        const float x1 = std::clamp(
            (modelBox.x - letterboxResult.left) / letterboxResult.scale,
            0.0F, static_cast<float>(frameWidth - 1));
        const float y1 = std::clamp(
            (modelBox.y - letterboxResult.top) / letterboxResult.scale,
            0.0F, static_cast<float>(frameHeight - 1));
        const float x2 = std::clamp(
            (modelBox.x + modelBox.width - letterboxResult.left)
                / letterboxResult.scale,
            0.0F, static_cast<float>(frameWidth - 1));
        const float y2 = std::clamp(
            (modelBox.y + modelBox.height - letterboxResult.top)
                / letterboxResult.scale,
            0.0F, static_cast<float>(frameHeight - 1));

        Detection detection;
        detection.box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
        detection.boxConfidence = candidateScores[index];
        detection.pointConfidence = candidateKeypoints[index][2];
        const float pointX =
            (candidateKeypoints[index][0] - letterboxResult.left)
            / letterboxResult.scale;
        const float pointY =
            (candidateKeypoints[index][1] - letterboxResult.top)
            / letterboxResult.scale;
        if (detection.pointConfidence >= kKeypointConfidence
            && pointX > 0.0F && pointX < frameWidth
            && pointY > 0.0F && pointY < frameHeight) {
            detection.point = cv::Point2f(pointX, pointY);
        }
        decoded.push_back(detection);
    }

    std::sort(decoded.begin(), decoded.end(), [](const Detection &first,
                                                  const Detection &second) {
        return first.boxConfidence > second.boxConfidence;
    });
    std::vector<Detection> filtered;
    for (const Detection &detection : decoded) {
        const bool overlaps = std::any_of(
            filtered.begin(), filtered.end(), [&](const Detection &kept) {
                return smallerBoxOverlap(detection.box, kept.box)
                    > kBoxOverlapThreshold;
            });
        if (!overlaps) {
            filtered.push_back(detection);
        }
    }

    std::vector<int> pointOrder(filtered.size());
    std::iota(pointOrder.begin(), pointOrder.end(), 0);
    std::sort(pointOrder.begin(), pointOrder.end(), [&](int first, int second) {
        if (filtered[first].pointConfidence
            != filtered[second].pointConfidence) {
            return filtered[first].pointConfidence
                > filtered[second].pointConfidence;
        }
        return filtered[first].boxConfidence
            > filtered[second].boxConfidence;
    });
    std::vector<cv::Point2f> keptPoints;
    for (const int index : pointOrder) {
        if (!filtered[index].point) {
            continue;
        }
        const bool tooClose = std::any_of(
            keptPoints.begin(), keptPoints.end(), [&](const cv::Point2f &point) {
                return cv::norm(*filtered[index].point - point)
                    < kPointMinimumDistance;
            });
        if (tooClose) {
            filtered[index].point.reset();
        } else {
            keptPoints.push_back(*filtered[index].point);
        }
    }

    std::sort(filtered.begin(), filtered.end(), [](const Detection &first,
                                                    const Detection &second) {
        const float firstY = first.box.y + first.box.height / 2.0F;
        const float secondY = second.box.y + second.box.height / 2.0F;
        if (firstY != secondY) {
            return firstY < secondY;
        }
        return first.box.x + first.box.width / 2.0F
            < second.box.x + second.box.width / 2.0F;
    });
    return filtered;
}

using Feature = std::array<float, 6>;
using FrameFeatures = std::vector<Feature>;

std::vector<FrameFeatures> buildFeatures(
    const std::vector<std::vector<Detection>> &allDetections,
    int width,
    int height)
{
    std::vector<FrameFeatures> allFeatures;
    allFeatures.reserve(allDetections.size());
    for (const auto &frameDetections : allDetections) {
        FrameFeatures features;
        for (const Detection &detection : frameDetections) {
            if (!detection.point) {
                continue;
            }
            features.push_back({
                detection.point->x / width,
                detection.point->y / height,
                detection.boxConfidence,
                detection.pointConfidence,
                detection.box.width / width,
                detection.box.height / height,
            });
        }
        allFeatures.push_back(std::move(features));
    }
    return allFeatures;
}

std::vector<float> buildContext(const std::vector<FrameFeatures> &allFeatures,
                                int frameIndex)
{
    const FrameFeatures &current = allFeatures[frameIndex];
    constexpr int contextLength = kContextRadius * 2 + 1;
    std::vector<float> context(
        current.size() * contextLength * 8, 0.0F);

    for (std::size_t detectionIndex = 0;
         detectionIndex < current.size(); ++detectionIndex) {
        const Feature &anchor = current[detectionIndex];
        for (int offset = -kContextRadius;
             offset <= kContextRadius; ++offset) {
            const int neighborFrame = frameIndex + offset;
            if (neighborFrame < 0
                || neighborFrame >= static_cast<int>(allFeatures.size())
                || allFeatures[neighborFrame].empty()) {
                continue;
            }

            const auto &neighbors = allFeatures[neighborFrame];
            auto nearest = neighbors.begin();
            float nearestDistance = std::numeric_limits<float>::max();
            for (auto iterator = neighbors.begin();
                 iterator != neighbors.end(); ++iterator) {
                const float deltaX = (*iterator)[0] - anchor[0];
                const float deltaY = (*iterator)[1] - anchor[1];
                const float distance = std::hypot(deltaX, deltaY);
                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearest = iterator;
                }
            }

            const int windowIndex = offset + kContextRadius;
            const std::size_t base =
                (detectionIndex * contextLength + windowIndex) * 8;
            context[base + 0] = (*nearest)[0] - anchor[0];
            context[base + 1] = (*nearest)[1] - anchor[1];
            context[base + 2] = nearestDistance;
            context[base + 3] = (*nearest)[2];
            context[base + 4] = (*nearest)[3];
            context[base + 5] = (*nearest)[4];
            context[base + 6] = (*nearest)[5];
            context[base + 7] = 1.0F;
        }
    }
    return context;
}

std::vector<int> minimumCostAssignment(
    const std::vector<std::vector<double>> &cost)
{
    const int rows = static_cast<int>(cost.size());
    if (rows == 0) {
        return {};
    }
    const int columns = static_cast<int>(cost.front().size());
    if (columns < rows) {
        throw std::runtime_error("匈牙利算法要求候选列数不少于检测数");
    }

    std::vector<double> rowPotential(rows + 1, 0.0);
    std::vector<double> columnPotential(columns + 1, 0.0);
    std::vector<int> matchedRow(columns + 1, 0);
    std::vector<int> previousColumn(columns + 1, 0);

    for (int row = 1; row <= rows; ++row) {
        matchedRow[0] = row;
        int currentColumn = 0;
        std::vector<double> minimum(columns + 1,
                                    std::numeric_limits<double>::infinity());
        std::vector<bool> used(columns + 1, false);
        do {
            used[currentColumn] = true;
            const int currentRow = matchedRow[currentColumn];
            double delta = std::numeric_limits<double>::infinity();
            int nextColumn = 0;
            for (int column = 1; column <= columns; ++column) {
                if (used[column]) {
                    continue;
                }
                const double reducedCost =
                    cost[currentRow - 1][column - 1]
                    - rowPotential[currentRow]
                    - columnPotential[column];
                if (reducedCost < minimum[column]) {
                    minimum[column] = reducedCost;
                    previousColumn[column] = currentColumn;
                }
                if (minimum[column] < delta) {
                    delta = minimum[column];
                    nextColumn = column;
                }
            }
            for (int column = 0; column <= columns; ++column) {
                if (used[column]) {
                    rowPotential[matchedRow[column]] += delta;
                    columnPotential[column] -= delta;
                } else {
                    minimum[column] -= delta;
                }
            }
            currentColumn = nextColumn;
        } while (matchedRow[currentColumn] != 0);

        do {
            const int previous = previousColumn[currentColumn];
            matchedRow[currentColumn] = matchedRow[previous];
            currentColumn = previous;
        } while (currentColumn != 0);
    }

    std::vector<int> assignment(rows, -1);
    for (int column = 1; column <= columns; ++column) {
        if (matchedRow[column] != 0) {
            assignment[matchedRow[column] - 1] = column - 1;
        }
    }
    return assignment;
}

class CoordinateTracker
{
public:
    explicit CoordinateTracker(Ort::Session &session)
        : session_(session)
    {
        const auto historyShape = session_.GetInputTypeInfo(0)
                                      .GetTensorTypeAndShapeInfo()
                                      .GetShape();
        const auto detectionShape = session_.GetInputTypeInfo(2)
                                        .GetTensorTypeAndShapeInfo()
                                        .GetShape();
        const auto contextShape = session_.GetInputTypeInfo(3)
                                      .GetTensorTypeAndShapeInfo()
                                      .GetShape();
        if (historyShape.size() != 4 || detectionShape.size() != 3
            || contextShape.size() != 4) {
            throw std::runtime_error("坐标关联模型输入维度不正确");
        }
        if (historyShape[1] > 0 || detectionShape[1] > 0) {
            throw std::runtime_error(
                "coordinate_association.onnx 仍为固定轨迹/检测容量模型");
        }
        historyLength_ = static_cast<int>(historyShape[2]);
        contextLength_ = static_cast<int>(contextShape[2]);
        if (historyLength_ <= 0 || contextLength_ <= 0) {
            throw std::runtime_error("坐标关联模型历史或上下文长度无效");
        }
    }

    int usedIdCount() const
    {
        return nextId_ - 1;
    }

    std::vector<int> update(const FrameFeatures &detections,
                            std::vector<float> &context)
    {
        const int detectionCount = static_cast<int>(detections.size());
        const int trackCount = usedIdCount();
        if (context.size()
            != static_cast<std::size_t>(detectionCount * contextLength_ * 8)) {
            throw std::runtime_error("detection_context 大小不正确");
        }

        std::vector<float> flatDetections;
        flatDetections.reserve(detections.size() * 6);
        for (const Feature &feature : detections) {
            flatDetections.insert(flatDetections.end(),
                                  feature.begin(), feature.end());
        }

        if (detectionCount == 0) {
            advanceHistory(flatDetections, {});
            return {};
        }

        const int onnxTrackCount = trackCount == 0 ? 1 : trackCount;
        std::vector<float> onnxHistory;
        if (trackCount == 0) {
            onnxHistory.assign(
                static_cast<std::size_t>(historyLength_ * 3), 0.0F);
        } else {
            onnxHistory = history_;
        }

        static_assert(sizeof(bool) == sizeof(std::uint8_t));
        std::vector<std::uint8_t> trackMask(
            static_cast<std::size_t>(onnxTrackCount),
            trackCount == 0 ? std::uint8_t{0} : std::uint8_t{1});
        std::vector<std::uint8_t> detectionMask(
            static_cast<std::size_t>(detectionCount), std::uint8_t{1});

        std::array<int64_t, 4> historyShape{
            1, onnxTrackCount, historyLength_, 3};
        std::array<int64_t, 2> trackMaskShape{1, onnxTrackCount};
        std::array<int64_t, 3> detectionShape{1, detectionCount, 6};
        std::array<int64_t, 4> contextShape{
            1, detectionCount, contextLength_, 8};
        std::array<int64_t, 2> detectionMaskShape{1, detectionCount};

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        std::array<Ort::Value, 5> inputs{
            Ort::Value::CreateTensor<float>(
                memoryInfo, onnxHistory.data(), onnxHistory.size(),
                historyShape.data(), historyShape.size()),
            Ort::Value::CreateTensor<bool>(
                memoryInfo, reinterpret_cast<bool *>(trackMask.data()),
                trackMask.size(), trackMaskShape.data(), trackMaskShape.size()),
            Ort::Value::CreateTensor<float>(
                memoryInfo, flatDetections.data(), flatDetections.size(),
                detectionShape.data(), detectionShape.size()),
            Ort::Value::CreateTensor<float>(
                memoryInfo, context.data(), context.size(),
                contextShape.data(), contextShape.size()),
            Ort::Value::CreateTensor<bool>(
                memoryInfo, reinterpret_cast<bool *>(detectionMask.data()),
                detectionMask.size(), detectionMaskShape.data(),
                detectionMaskShape.size()),
        };

        constexpr std::array<const char *, 5> inputNames{
            "history", "track_mask", "detections",
            "detection_context", "detection_mask"};
        constexpr std::array<const char *, 1> outputNames{"logits"};
        Ort::RunOptions runOptions;
        auto outputValues = session_.Run(
            runOptions, inputNames.data(), inputs.data(), inputs.size(),
            outputNames.data(), outputNames.size());

        const auto outputShape = outputValues.front()
                                     .GetTensorTypeAndShapeInfo()
                                     .GetShape();
        if (outputShape.size() != 3 || outputShape[0] != 1
            || outputShape[1] != detectionCount) {
            throw std::runtime_error("坐标关联模型 logits 形状不正确");
        }
        const int outputColumns = static_cast<int>(outputShape[2]);
        if (outputColumns < onnxTrackCount + 1) {
            throw std::runtime_error("坐标关联模型 logits 列数不足");
        }
        const float *logits = outputValues.front().GetTensorData<float>();

        constexpr double forbiddenScore = -1.0e12;
        const int assignmentColumns = trackCount + detectionCount;
        std::vector<std::vector<double>> costs(
            detectionCount,
            std::vector<double>(assignmentColumns, -forbiddenScore));
        for (int detection = 0; detection < detectionCount; ++detection) {
            for (int track = 0; track < trackCount; ++track) {
                costs[detection][track] =
                    -static_cast<double>(
                        logits[detection * outputColumns + track]);
            }
            const float newScore =
                logits[detection * outputColumns + (outputColumns - 1)];
            costs[detection][trackCount + detection] = -newScore;
        }

        const std::vector<int> selected = minimumCostAssignment(costs);
        std::vector<int> pointIds;
        pointIds.reserve(detectionCount);
        for (const int column : selected) {
            if (column < trackCount) {
                pointIds.push_back(column + 1);
            } else {
                pointIds.push_back(nextId_++);
            }
        }

        while (usedIdCount() * historyLength_ * 3
               > static_cast<int>(history_.size())) {
            history_.insert(history_.end(),
                            static_cast<std::size_t>(historyLength_ * 3),
                            0.0F);
        }

        std::vector<int> assignedSlots;
        assignedSlots.reserve(pointIds.size());
        for (const int pointId : pointIds) {
            assignedSlots.push_back(pointId - 1);
        }
        advanceHistory(flatDetections, assignedSlots);
        return pointIds;
    }

private:
    void advanceHistory(const std::vector<float> &detections,
                        const std::vector<int> &assignedSlots)
    {
        const int trackCount = usedIdCount();
        for (int slot = 0; slot < trackCount; ++slot) {
            const std::size_t slotBase =
                static_cast<std::size_t>(slot * historyLength_ * 3);
            const std::size_t lastBase = slotBase
                + static_cast<std::size_t>((historyLength_ - 1) * 3);
            const float previousX = history_[lastBase + 0];
            const float previousY = history_[lastBase + 1];
            for (int time = 0; time < historyLength_ - 1; ++time) {
                for (int channel = 0; channel < 3; ++channel) {
                    history_[slotBase + time * 3 + channel] =
                        history_[slotBase + (time + 1) * 3 + channel];
                }
            }
            history_[lastBase + 0] = previousX;
            history_[lastBase + 1] = previousY;
            history_[lastBase + 2] = 0.0F;
        }

        for (std::size_t detection = 0;
             detection < assignedSlots.size(); ++detection) {
            const int slot = assignedSlots[detection];
            const std::size_t lastBase = static_cast<std::size_t>(
                (slot * historyLength_ + historyLength_ - 1) * 3);
            history_[lastBase + 0] = detections[detection * 6 + 0];
            history_[lastBase + 1] = detections[detection * 6 + 1];
            history_[lastBase + 2] = 1.0F;
        }
    }

    Ort::Session &session_;
    int historyLength_ = 15;
    int contextLength_ = 21;
    int nextId_ = 1;
    std::vector<float> history_;
};

struct FrameRecord
{
    int frameIndex = 0;
    double timestamp = 0.0;
    std::map<int, cv::Point2i> visiblePoints;
};

std::map<int, cv::Point2i> drawResults(
    cv::Mat &frame,
    const std::vector<Detection> &detections,
    const std::vector<int> &pointIds)
{
    std::map<int, cv::Point2i> visible;
    std::size_t pointIndex = 0;
    for (const Detection &detection : detections) {
        cv::rectangle(frame, detection.box, cv::Scalar(0, 255, 0), 2,
                      cv::LINE_AA);
        if (!detection.point) {
            continue;
        }
        if (pointIndex >= pointIds.size()) {
            throw std::runtime_error("圆心数量与关联模型输出 ID 数量不一致");
        }

        const int pointId = pointIds[pointIndex++];
        const cv::Point2i point(cvRound(detection.point->x),
                               cvRound(detection.point->y));
        visible[pointId] = point;
        cv::circle(frame, point, 6, cv::Scalar(0, 0, 255), -1,
                   cv::LINE_AA);
        const std::string label = "ID " + std::to_string(pointId);
        const cv::Point labelPosition(point.x + 9, point.y - 9);
        cv::putText(frame, label, labelPosition, cv::FONT_HERSHEY_SIMPLEX,
                    0.65, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
        cv::putText(frame, label, labelPosition, cv::FONT_HERSHEY_SIMPLEX,
                    0.65, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }
    return visible;
}

void writeCsv(const QString &path,
              const QString &videoPath,
              const QString &processedVideoPath,
              const std::vector<FrameRecord> &records,
              int idCount)
{
    std::ofstream stream(fileSystemPath(path),
                         std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("无法创建圆环原始坐标 CSV");
    }
    writeCsvSourceVideoPath(stream, videoPath);
    writeCsvProcessedVideoPath(stream, processedVideoPath);
    stream << "frame_index,video_timestamp_sec";
    for (int id = 1; id <= idCount; ++id) {
        stream << ",id_" << id << "_x_px,id_" << id << "_y_px";
    }
    stream << '\n';
    stream << std::fixed << std::setprecision(9);

    for (const FrameRecord &record : records) {
        stream << record.frameIndex << ',' << record.timestamp;
        for (int id = 1; id <= idCount; ++id) {
            const auto found = record.visiblePoints.find(id);
            if (found == record.visiblePoints.end()) {
                stream << ",,";
            } else {
                stream << ',' << found->second.x << ',' << found->second.y;
            }
        }
        stream << '\n';
    }
}

} // namespace

InferenceResult RingInference::run(const QString &videoPath,
                                   const QString &ringModelPath,
                                   const QString &associationModelPath,
                                   const InferenceProgress &progress,
                                   const InferenceCancelCheck &isCanceled)
{
    ensureFileExists(ringModelPath, QStringLiteral("圆环检测模型"));
    ensureFileExists(associationModelPath, QStringLiteral("坐标关联模型"));
    checkCanceled(isCanceled);

    const std::vector<std::string> availableProviders =
        Ort::GetAvailableProviders();
    if (std::find(availableProviders.begin(), availableProviders.end(),
                  "CUDAExecutionProvider") == availableProviders.end()) {
        throw std::runtime_error(
            "ONNX Runtime CUDAExecutionProvider 不可用，已拒绝回退 CPU");
    }

    Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "true_high_speed");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    OrtCUDAProviderOptions cudaOptions{};
    cudaOptions.device_id = 0;
    sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);

    if (progress) {
        progress(0, QStringLiteral("正在加载圆环 ONNX 模型……"));
    }
    Ort::Session ringSession = makeSession(environment, ringModelPath,
                                           sessionOptions);
    Ort::Session associationSession = makeSession(
        environment, associationModelPath, sessionOptions);

    const auto ringInputShape = ringSession.GetInputTypeInfo(0)
                                    .GetTensorTypeAndShapeInfo()
                                    .GetShape();
    if (ringInputShape.size() != 4 || ringInputShape[0] != 1
        || ringInputShape[1] != 3 || ringInputShape[2] <= 0
        || ringInputShape[2] != ringInputShape[3]) {
        throw std::runtime_error("ring_pose.onnx 输入形状不是 [1,3,S,S]");
    }
    const int imageSize = static_cast<int>(ringInputShape[2]);
    CoordinateTracker tracker(associationSession);

    cv::VideoCapture capture(openCvPath(videoPath));
    if (!capture.isOpened()) {
        throw std::runtime_error("OpenCV 无法打开输入视频");
    }
    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    int totalFrames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 0.0) {
        fps = 25.0;
    }
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("视频分辨率无效");
    }

    std::vector<std::vector<Detection>> allDetections;
    if (totalFrames > 0) {
        allDetections.reserve(static_cast<std::size_t>(totalFrames));
    }

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    constexpr std::array<const char *, 1> ringInputNames{"images"};
    constexpr std::array<const char *, 1> ringOutputNames{"output0"};
    const std::array<int64_t, 4> inputShape{1, 3, imageSize, imageSize};
    Ort::RunOptions runOptions;

    cv::Mat frame;
    int detectedFrames = 0;
    while (capture.read(frame)) {
        checkCanceled(isCanceled);
        LetterboxResult prepared = letterbox(frame, imageSize);
        Ort::Value input = Ort::Value::CreateTensor<float>(
            memoryInfo, prepared.tensor.data(), prepared.tensor.size(),
            inputShape.data(), inputShape.size());
        auto output = ringSession.Run(
            runOptions, ringInputNames.data(), &input, 1,
            ringOutputNames.data(), ringOutputNames.size());
        allDetections.push_back(decodeRingOutput(
            output.front(), prepared, width, height));
        ++detectedFrames;

        if (progress && (detectedFrames == 1 || detectedFrames % 10 == 0)) {
            const int percent = totalFrames > 0
                ? std::clamp(detectedFrames * 70 / totalFrames, 0, 70)
                : 0;
            progress(percent,
                     QStringLiteral("第一遍：检测圆环 %1/%2 帧")
                         .arg(detectedFrames)
                         .arg(totalFrames > 0 ? totalFrames : detectedFrames));
        }
    }
    capture.release();
    if (allDetections.empty()) {
        throw std::runtime_error("输入视频没有可读取的帧");
    }
    totalFrames = static_cast<int>(allDetections.size());

    checkCanceled(isCanceled);
    std::vector<FrameFeatures> allFeatures = buildFeatures(
        allDetections, width, height);

    const QFileInfo inputInfo(videoPath);
    const QString outputVideo = QDir(inputInfo.absolutePath()).filePath(
        inputInfo.completeBaseName() + QStringLiteral("_ring_annotated.mp4"));
    const QString outputCsv = QDir(inputInfo.absolutePath()).filePath(
        inputInfo.completeBaseName() + QStringLiteral("_ring_raw.csv"));

    capture.open(openCvPath(videoPath));
    cv::VideoWriter writer(openCvPath(outputVideo),
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                           cv::Size(width, height));
    if (!capture.isOpened() || !writer.isOpened()) {
        throw std::runtime_error("无法重新读取视频或创建圆环标注视频");
    }

    std::vector<FrameRecord> records;
    records.reserve(allDetections.size());
    for (int frameIndex = 0; frameIndex < totalFrames; ++frameIndex) {
        checkCanceled(isCanceled);
        if (!capture.read(frame)) {
            break;
        }

        std::vector<float> context = buildContext(allFeatures, frameIndex);
        const std::vector<int> pointIds = tracker.update(
            allFeatures[frameIndex], context);
        FrameRecord record;
        record.frameIndex = frameIndex;
        record.timestamp = frameIndex / fps;
        record.visiblePoints = drawResults(
            frame, allDetections[frameIndex], pointIds);
        records.push_back(std::move(record));
        writer.write(frame);

        if (progress && (frameIndex == 0 || (frameIndex + 1) % 10 == 0
                         || frameIndex + 1 == totalFrames)) {
            const int percent = 70
                + std::clamp((frameIndex + 1) * 30 / totalFrames, 0, 30);
            progress(percent,
                     QStringLiteral("第二遍：关联圆心 ID %1/%2 帧")
                         .arg(frameIndex + 1)
                         .arg(totalFrames));
        }
    }
    capture.release();
    writer.release();
    checkCanceled(isCanceled);

    writeCsv(outputCsv, videoPath, outputVideo, records, tracker.usedIdCount());
    if (progress) {
        progress(100, QStringLiteral("圆环识别与 ID 关联完成"));
    }
    return {outputVideo, outputCsv};
}
