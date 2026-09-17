#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "inferenceworker.h"

#include <QMainWindow>
#include <QPointF>
#include <QPointer>
#include <QVector>

class QCloseEvent;
class QThread;

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow final : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void browseVideo();
    void startRingScheme();
    void startLocoTrackScheme();
    void startTrackOn2Scheme();
    void startTapScheme();
    void startPhysicalAnalysis();
    void startReview();

private:
    QString selectedVideoPath() const;
    QString resolveModelDirectory() const;
    bool validateVideoSelection(QString *videoPath = nullptr);
    bool ensureOpenCvReadableVideo(QString *videoPath);
    bool selectTrackingPoints(const QString &videoPath,
                              const QString &schemeName,
                              int maximumPoints,
                              QVector<QPointF> *points);
    void startWorker(InferenceWorker::Scheme scheme,
                     const QString &videoPath,
                     const QVector<QPointF> &tapPoints = {});
    void setProcessingState(bool processing);
    void finishWorkerUi(const QString &statusText, bool showAsError);
    void stopWorker();

    Ui::MainWindow *ui;
    QThread *workerThread_ = nullptr;
    QPointer<InferenceWorker> worker_;
};

#endif // MAINWINDOW_H
