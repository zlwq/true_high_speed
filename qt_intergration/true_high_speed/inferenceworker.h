#ifndef INFERENCEWORKER_H
#define INFERENCEWORKER_H

#include <QObject>
#include <QPointF>
#include <QString>
#include <QVector>

#include <atomic>

class InferenceWorker final : public QObject
{
    Q_OBJECT

public:
    enum class Scheme {
        Ring,
        LocoTrack,
        TrackOn2,
        Tap
    };
    Q_ENUM(Scheme)

    InferenceWorker(Scheme scheme,
                    QString videoPath,
                    QString modelDirectory,
                    QVector<QPointF> tapPoints,
                    QObject *parent = nullptr);

    // 只写原子变量，可以安全地由主线程直接调用。
    void requestCancel() noexcept;

public slots:
    void process();

signals:
    void progressChanged(int percent, const QString &message);
    void completed(const QString &summary);
    void failed(const QString &message);
    void canceled();

private:
    Scheme scheme_;
    QString videoPath_;
    QString modelDirectory_;
    QVector<QPointF> tapPoints_;
    std::atomic_bool cancelRequested_{false};
};

#endif // INFERENCEWORKER_H
