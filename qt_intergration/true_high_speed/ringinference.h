#ifndef RINGINFERENCE_H
#define RINGINFERENCE_H

#include <QString>

#include <functional>
#include <stdexcept>

struct InferenceResult
{
    QString annotatedVideoPath;
    QString rawCsvPath;
};

class InferenceCanceled final : public std::runtime_error
{
public:
    InferenceCanceled()
        : std::runtime_error("inference canceled")
    {
    }
};

using InferenceProgress = std::function<void(int, const QString &)>;
using InferenceCancelCheck = std::function<bool()>;

class RingInference final
{
public:
    static InferenceResult run(const QString &videoPath,
                               const QString &ringModelPath,
                               const QString &associationModelPath,
                               const InferenceProgress &progress,
                               const InferenceCancelCheck &isCanceled);
};

#endif // RINGINFERENCE_H
