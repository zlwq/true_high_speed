#ifndef TAPINFERENCE_H
#define TAPINFERENCE_H

#include "ringinference.h"

#include <QPointF>
#include <QString>
#include <QVector>

class TapInference final
{
public:
    static InferenceResult run(const QString &videoPath,
                               const QString &modelPath,
                               const QVector<QPointF> &points,
                               const InferenceProgress &progress,
                               const InferenceCancelCheck &isCanceled);
};

#endif // TAPINFERENCE_H
