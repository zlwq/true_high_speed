#ifndef LOCOTRACKINFERENCE_H
#define LOCOTRACKINFERENCE_H

#include "ringinference.h"

#include <QPointF>
#include <QString>
#include <QVector>

class LocoTrackInference final
{
public:
    static InferenceResult run(const QString &videoPath,
                               const QString &modelDirectory,
                               const QVector<QPointF> &points,
                               const InferenceProgress &progress,
                               const InferenceCancelCheck &isCanceled);
};

#endif // LOCOTRACKINFERENCE_H
