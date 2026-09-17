#ifndef TRACKON2INFERENCE_H
#define TRACKON2INFERENCE_H

#include "ringinference.h"

#include <QPointF>
#include <QString>
#include <QVector>

class TrackOn2Inference final
{
public:
    static void clearCachedModel();

    static InferenceResult run(const QString &videoPath,
                               const QString &modelPath,
                               const QVector<QPointF> &points,
                               const InferenceProgress &progress,
                               const InferenceCancelCheck &isCanceled);
};

#endif // TRACKON2INFERENCE_H
