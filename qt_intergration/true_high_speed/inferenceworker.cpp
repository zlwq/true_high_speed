#include "inferenceworker.h"

#include "locotrackinference.h"
#include "ringinference.h"
#include "tapinference.h"
#include "trackon2inference.h"

#include <QDir>
#include <QFileInfo>

#include <exception>
#include <utility>

InferenceWorker::InferenceWorker(Scheme scheme,
                                 QString videoPath,
                                 QString modelDirectory,
                                 QVector<QPointF> tapPoints,
                                 QObject *parent)
    : QObject(parent)
    , scheme_(scheme)
    , videoPath_(std::move(videoPath))
    , modelDirectory_(std::move(modelDirectory))
    , tapPoints_(std::move(tapPoints))
{
}

void InferenceWorker::requestCancel() noexcept
{
    cancelRequested_.store(true, std::memory_order_relaxed);
}

void InferenceWorker::process()
{
    const auto canceled = [this] {
        return cancelRequested_.load(std::memory_order_relaxed);
    };
    const auto progress = [this](int percent, const QString &message) {
        emit progressChanged(percent, message);
    };

    try {
        if (scheme_ != Scheme::TrackOn2) {
            TrackOn2Inference::clearCachedModel();
        }

        InferenceResult result;
        if (scheme_ == Scheme::Ring) {
            const QString ringModel = QDir(modelDirectory_).filePath(
                QStringLiteral("ring_pose.onnx"));
            const QString associationModel = QDir(modelDirectory_).filePath(
                QStringLiteral("coordinate_association.onnx"));
            result = RingInference::run(videoPath_, ringModel, associationModel,
                                        progress, canceled);
        } else if (scheme_ == Scheme::LocoTrack) {
            const QString locoModelDirectory = QDir(modelDirectory_).filePath(
                QStringLiteral("localtrack"));
            result = LocoTrackInference::run(
                videoPath_, locoModelDirectory, tapPoints_, progress, canceled);
        } else if (scheme_ == Scheme::TrackOn2) {
            const QString trackOn2Model = QDir(modelDirectory_).filePath(
                QStringLiteral("trackon2_dinov2_pure_torchscript.pt"));
            result = TrackOn2Inference::run(
                videoPath_, trackOn2Model, tapPoints_, progress, canceled);
        } else {
            const QString tapModel = QDir(modelDirectory_).filePath(
                QStringLiteral("tapnextpp_stable_512_q4_fp16_ts.pt"));
            result = TapInference::run(videoPath_, tapModel, tapPoints_,
                                       progress, canceled);
        }

        if (canceled()) {
            emit canceled();
            return;
        }

        emit completed(
            tr("处理完成\n标注视频：%1\n原始坐标：%2")
                .arg(QDir::toNativeSeparators(result.annotatedVideoPath),
                     QDir::toNativeSeparators(result.rawCsvPath)));
    } catch (const InferenceCanceled &) {
        emit canceled();
    } catch (const std::exception &error) {
        emit failed(QString::fromUtf8(error.what()));
    } catch (...) {
        emit failed(tr("发生未知推理错误。"));
    }
}
