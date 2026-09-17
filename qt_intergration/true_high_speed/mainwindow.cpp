#include "mainwindow.h"
#include "physicalanalysis.h"
#include "reviewdialog.h"
#include "ui_mainwindow.h"

#include <QApplication>
#include <QBrush>
#include <QCloseEvent>
#include <QColor>
#include <QCoreApplication>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGraphicsEllipseItem>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QGraphicsSimpleTextItem>
#include <QGraphicsView>
#include <QImage>
#include <QLabel>
#include <QLineF>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QPushButton>
#include <QProcess>
#include <QProgressDialog>
#include <QResizeEvent>
#include <QShowEvent>
#include <QStandardPaths>
#include <QStringList>
#include <QThread>
#include <QVBoxLayout>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace {

std::string openCvPath(const QString &path)
{
    // Windows 的 OpenCV/FFmpeg 后端通常按当前系统代码页解释窄字符串。
    // 对中文路径，toLocal8Bit 比直接 toStdString 更可靠。
    return path.toLocal8Bit().toStdString();
}

QImage matToQImage(const cv::Mat &bgr)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows,
                  static_cast<int>(rgb.step), QImage::Format_RGB888)
        .copy();
}

bool canDecodeFirstFrame(const QString &videoPath)
{
    cv::VideoCapture capture(openCvPath(videoPath));
    cv::Mat frame;
    return capture.isOpened() && capture.read(frame) && !frame.empty();
}

QString findFfmpegExecutable()
{
#ifdef _WIN32
    const QString bundled = QDir(QCoreApplication::applicationDirPath())
                                .filePath(QStringLiteral("ffmpeg.exe"));
    if (QFileInfo::exists(bundled)) {
        return bundled;
    }
    return QStandardPaths::findExecutable(QStringLiteral("ffmpeg.exe"));
#else
    const QString bundled = QDir(QCoreApplication::applicationDirPath())
                                .filePath(QStringLiteral("ffmpeg"));
    if (QFileInfo::exists(bundled)) {
        return bundled;
    }
    return QStandardPaths::findExecutable(QStringLiteral("ffmpeg"));
#endif
}

class PointSelectionView final : public QGraphicsView
{
public:
    explicit PointSelectionView(const QImage &image,
                                int maximumPoints,
                                QWidget *parent = nullptr)
        : QGraphicsView(parent)
        , image_(QPixmap::fromImage(image))
        , maximumPoints_(maximumPoints)
    {
        setScene(&scene_);
        setRenderHints(QPainter::Antialiasing
                       | QPainter::SmoothPixmapTransform);
        setDragMode(QGraphicsView::NoDrag);
        setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
        redraw();
    }

    QVector<QPointF> points() const
    {
        return points_;
    }

protected:
    void showEvent(QShowEvent *event) override
    {
        QGraphicsView::showEvent(event);
        fitImage();
    }

    void resizeEvent(QResizeEvent *event) override
    {
        QGraphicsView::resizeEvent(event);
        fitImage();
    }

    void mousePressEvent(QMouseEvent *event) override
    {
        const QPointF position = mapToScene(event->position().toPoint());
        const QRectF imageBounds(QPointF(0.0, 0.0), image_.size());
        if (!imageBounds.contains(position)) {
            QGraphicsView::mousePressEvent(event);
            return;
        }

        if (event->button() == Qt::LeftButton) {
            if (maximumPoints_ > 0 && points_.size() >= maximumPoints_) {
                event->accept();
                return;
            }
            points_.append(position);
            redraw();
            event->accept();
            return;
        }

        if (event->button() == Qt::RightButton && !points_.isEmpty()) {
            int closestIndex = -1;
            qreal closestDistance = std::numeric_limits<qreal>::max();
            for (int index = 0; index < points_.size(); ++index) {
                const QLineF line(position, points_.at(index));
                if (line.length() < closestDistance) {
                    closestDistance = line.length();
                    closestIndex = index;
                }
            }

            const qreal sceneTolerance = 24.0
                / std::max<qreal>(std::abs(transform().m11()), 0.001);
            if (closestIndex >= 0 && closestDistance <= sceneTolerance) {
                points_.removeAt(closestIndex);
                redraw();
            }
            event->accept();
            return;
        }

        QGraphicsView::mousePressEvent(event);
    }

private:
    void fitImage()
    {
        if (!image_.isNull()) {
            fitInView(QRectF(QPointF(0.0, 0.0), image_.size()),
                      Qt::KeepAspectRatio);
        }
    }

    void redraw()
    {
        scene_.clear();
        scene_.addPixmap(image_)->setZValue(0.0);
        scene_.setSceneRect(QRectF(QPointF(0.0, 0.0), image_.size()));

        const QPen outline(Qt::white, 2.0);
        const QBrush fill(QColor(235, 70, 70, 220));
        for (int index = 0; index < points_.size(); ++index) {
            const QPointF &point = points_.at(index);
            auto *marker = scene_.addEllipse(
                QRectF(point.x() - 6.0, point.y() - 6.0, 12.0, 12.0),
                outline, fill);
            marker->setZValue(1.0);

            auto *number = scene_.addSimpleText(QString::number(index + 1));
            number->setBrush(Qt::yellow);
            number->setPen(QPen(Qt::black, 1.0));
            number->setPos(point + QPointF(8.0, -20.0));
            number->setZValue(2.0);
        }
    }

    QGraphicsScene scene_;
    QPixmap image_;
    QVector<QPointF> points_;
    int maximumPoints_ = 0;
};

class PointSelectionDialog final : public QDialog
{
public:
    explicit PointSelectionDialog(const QImage &firstFrame,
                                  double playbackFps,
                                  const QString &schemeName,
                                  int maximumPoints,
                                  QWidget *parent = nullptr)
        : QDialog(parent)
        , view_(new PointSelectionView(firstFrame, maximumPoints, this))
    {
        setWindowTitle(tr("选择 %1 跟踪点").arg(schemeName));
        resize(1000, 720);

        const QString limitText = maximumPoints > 0
            ? tr("　最多选择 %1 个点。").arg(maximumPoints)
            : QString();
        auto *instructions = new QLabel(
            tr("%1　MP4 播放帧率：%2 FPS　左键添加点；右键点击点附近删除；完成后按“开始跟踪”。%3")
                .arg(schemeName)
                .arg(playbackFps, 0, 'f', 3)
                .arg(limitText),
            this);
        instructions->setWordWrap(true);

        auto *buttons = new QDialogButtonBox(this);
        auto *startButton = buttons->addButton(
            tr("开始跟踪"), QDialogButtonBox::AcceptRole);
        buttons->addButton(QDialogButtonBox::Cancel);

        auto *layout = new QVBoxLayout(this);
        layout->addWidget(instructions);
        layout->addWidget(view_, 1);
        layout->addWidget(buttons);

        connect(startButton, &QPushButton::clicked, this, [this] {
            if (view_->points().isEmpty()) {
                QMessageBox::warning(this, tr("尚未选择跟踪点"),
                                     tr("请至少在第一帧上选择一个点。"));
                return;
            }
            accept();
        });
        connect(buttons, &QDialogButtonBox::rejected,
                this, &QDialog::reject);
    }

    QVector<QPointF> points() const
    {
        return view_->points();
    }

private:
    PointSelectionView *view_;
};

} // namespace

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->processingProgressBar->setRange(0, 100);
    ui->processingProgressBar->setValue(0);

    connect(ui->browseButton, &QPushButton::clicked,
            this, &MainWindow::browseVideo);
    connect(ui->ringSchemeButton, &QPushButton::clicked,
            this, &MainWindow::startRingScheme);
    connect(ui->localtrackButton, &QPushButton::clicked,
            this, &MainWindow::startLocoTrackScheme);
    connect(ui->trackonButton, &QPushButton::clicked,
            this, &MainWindow::startTrackOn2Scheme);
    connect(ui->tapSchemeButton, &QPushButton::clicked,
            this, &MainWindow::startTapScheme);
    connect(ui->physicalAnalysisButton, &QPushButton::clicked,
            this, &MainWindow::startPhysicalAnalysis);
    connect(ui->ReviewButton, &QPushButton::clicked,
            this, &MainWindow::startReview);
}

MainWindow::~MainWindow()
{
    stopWorker();
    delete ui;
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    if (workerThread_ && workerThread_->isRunning()) {
        const auto answer = QMessageBox::question(
            this, tr("正在处理视频"),
            tr("推理尚未完成。要取消任务并退出吗？"),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if (answer != QMessageBox::Yes) {
            event->ignore();
            return;
        }
        ui->statusLabel->setText(tr("正在取消，请稍候……"));
        stopWorker();
    }
    event->accept();
}

void MainWindow::browseVideo()
{
    const QString currentPath = selectedVideoPath();
    const QString initialDirectory = currentPath.isEmpty()
        ? QDir::homePath()
        : QFileInfo(currentPath).absolutePath();
    const QString path = QFileDialog::getOpenFileName(
        this, tr("选择高速视频"), initialDirectory,
        tr("视频文件 (*.mp4 *.avi *.mov *.mkv *.m4v);;所有文件 (*.*)"));
    if (path.isEmpty()) {
        return;
    }

    ui->videoPathEdit->setText(QDir::toNativeSeparators(path));
    ui->videoPathEdit->setToolTip(QDir::toNativeSeparators(path));
    ui->statusLabel->setText(tr("视频已选择，请选择处理方案"));
    ui->processingProgressBar->setValue(0);
}

void MainWindow::startPhysicalAnalysis()
{
    if (workerThread_ && workerThread_->isRunning()) {
        QMessageBox::information(this, tr("任务正在运行"),
                                 tr("请等待当前视频处理完成。"));
        return;
    }

    const QString csvPath = QFileDialog::getOpenFileName(
        this, tr("选择原始坐标 CSV"), QDir::homePath(),
        tr("CSV 文件 (*.csv)"));
    if (csvPath.isEmpty()) {
        return;
    }
    openPhysicalAnalysisDialog(csvPath, this);
}

void MainWindow::startReview()
{
    if (workerThread_ && workerThread_->isRunning()) {
        QMessageBox::information(this, tr("任务正在运行"),
                                 tr("请等待当前视频处理完成。"));
        return;
    }

    const QString csvPath = QFileDialog::getOpenFileName(
        this, tr("选择物理信息 CSV"), QDir::homePath(),
        tr("CSV 文件 (*.csv)"));
    if (csvPath.isEmpty()) {
        return;
    }
    openReviewDialog(csvPath, this);
}

void MainWindow::startRingScheme()
{
    QString videoPath;
    if (!validateVideoSelection(&videoPath)) {
        return;
    }
    startWorker(InferenceWorker::Scheme::Ring, videoPath);
}

void MainWindow::startTapScheme()
{
    QString videoPath;
    if (!validateVideoSelection(&videoPath)) {
        return;
    }

    QVector<QPointF> points;
    if (!selectTrackingPoints(videoPath, QStringLiteral("TAPNext++"), 0,
                              &points)) {
        return;
    }
    startWorker(InferenceWorker::Scheme::Tap, videoPath, points);
}

void MainWindow::startLocoTrackScheme()
{
    QString videoPath;
    if (!validateVideoSelection(&videoPath)) {
        return;
    }

    QVector<QPointF> points;
    if (!selectTrackingPoints(videoPath, QStringLiteral("LocoTrack-B"), 64,
                              &points)) {
        return;
    }
    startWorker(InferenceWorker::Scheme::LocoTrack, videoPath, points);
}

void MainWindow::startTrackOn2Scheme()
{
    QString videoPath;
    if (!validateVideoSelection(&videoPath)) {
        return;
    }

    QVector<QPointF> points;
    if (!selectTrackingPoints(videoPath, QStringLiteral("Track-On2"), 0,
                              &points)) {
        return;
    }
    startWorker(InferenceWorker::Scheme::TrackOn2, videoPath, points);
}

QString MainWindow::selectedVideoPath() const
{
    return QDir::fromNativeSeparators(ui->videoPathEdit->text().trimmed());
}

QString MainWindow::resolveModelDirectory() const
{
    QStringList candidates;
    candidates << QDir(QCoreApplication::applicationDirPath())
                      .filePath(QStringLiteral("model"));
    candidates << QDir(QCoreApplication::applicationDirPath())
                      .filePath(QStringLiteral("../model"));
    candidates << QDir::current().filePath(QStringLiteral("model"));
#ifdef THS_SOURCE_MODEL_DIR
    candidates << QString::fromUtf8(THS_SOURCE_MODEL_DIR);
#endif

    for (const QString &candidate : candidates) {
        const QDir directory(QDir::cleanPath(candidate));
        if (directory.exists()) {
            return directory.absolutePath();
        }
    }
    return QDir::cleanPath(candidates.constFirst());
}

bool MainWindow::validateVideoSelection(QString *videoPath)
{
    if (workerThread_ && workerThread_->isRunning()) {
        QMessageBox::information(this, tr("任务正在运行"),
                                 tr("请等待当前视频处理完成。"));
        return false;
    }

    const QString path = selectedVideoPath();
    if (path.isEmpty()) {
        QMessageBox::warning(this, tr("未选择视频"),
                             tr("请先点击“浏览”选择视频文件。"));
        return false;
    }
    const QFileInfo info(path);
    if (!info.isFile() || !info.isReadable()) {
        QMessageBox::critical(this, tr("视频不可用"),
                              tr("视频文件不存在或无法读取：\n%1")
                                  .arg(QDir::toNativeSeparators(path)));
        return false;
    }

    QString preparedPath = info.absoluteFilePath();
    if (!ensureOpenCvReadableVideo(&preparedPath)) {
        return false;
    }

    if (videoPath) {
        *videoPath = preparedPath;
    }
    return true;
}

bool MainWindow::ensureOpenCvReadableVideo(QString *videoPath)
{
    if (canDecodeFirstFrame(*videoPath)) {
        return true;
    }

    const QString ffmpeg = findFfmpegExecutable();
    if (ffmpeg.isEmpty()) {
        QMessageBox::critical(
            this, tr("缺少 FFmpeg"),
            tr("OpenCV 无法解码这个视频，程序需要先自动转换为 H.264。\n\n"
               "请把 ffmpeg.exe 放到程序 exe 旁边，或将 FFmpeg 加入系统 PATH。"));
        return false;
    }

    const QFileInfo inputInfo(*videoPath);
    const QString outputPath = QDir(inputInfo.absolutePath()).filePath(
        inputInfo.completeBaseName() + QStringLiteral("_h264.mp4"));
    const QFileInfo outputInfo(outputPath);
    if (outputInfo.isFile()
        && outputInfo.lastModified() >= inputInfo.lastModified()
        && canDecodeFirstFrame(outputPath)) {
        *videoPath = outputInfo.absoluteFilePath();
        ui->videoPathEdit->setText(QDir::toNativeSeparators(*videoPath));
        ui->videoPathEdit->setToolTip(QDir::toNativeSeparators(*videoPath));
        return true;
    }

    QProgressDialog dialog(tr("当前视频无法由 OpenCV 解码，正在自动转换为 H.264……"),
                           tr("取消"), 0, 0, this);
    dialog.setWindowTitle(tr("转换视频"));
    dialog.setWindowModality(Qt::WindowModal);
    dialog.setMinimumDuration(0);
    dialog.setAutoClose(false);
    dialog.setAutoReset(false);
    dialog.show();

    QProcess process;
    process.setProcessChannelMode(QProcess::SeparateChannels);
    const QStringList arguments{
        QStringLiteral("-nostdin"),
        QStringLiteral("-hide_banner"),
        QStringLiteral("-loglevel"), QStringLiteral("error"),
        QStringLiteral("-y"),
        QStringLiteral("-i"), *videoPath,
        QStringLiteral("-map"), QStringLiteral("0:v:0"),
        QStringLiteral("-an"),
        QStringLiteral("-c:v"), QStringLiteral("libx264"),
        QStringLiteral("-preset"), QStringLiteral("fast"),
        QStringLiteral("-crf"), QStringLiteral("12"),
        QStringLiteral("-pix_fmt"), QStringLiteral("yuv420p"),
        QStringLiteral("-movflags"), QStringLiteral("+faststart"),
        outputPath
    };
    process.start(ffmpeg, arguments);
    if (!process.waitForStarted(5000)) {
        dialog.close();
        QMessageBox::critical(this, tr("转换失败"),
                              tr("无法启动 FFmpeg：\n%1")
                                  .arg(process.errorString()));
        return false;
    }

    while (process.state() != QProcess::NotRunning) {
        process.waitForFinished(100);
        QApplication::processEvents();
        if (dialog.wasCanceled()) {
            process.kill();
            process.waitForFinished();
            QFile::remove(outputPath);
            return false;
        }
    }
    dialog.close();

    if (process.exitStatus() != QProcess::NormalExit
        || process.exitCode() != 0) {
        const QString details = QString::fromUtf8(process.readAllStandardError())
                                    .trimmed().right(2000);
        QFile::remove(outputPath);
        QMessageBox::critical(
            this, tr("转换失败"),
            details.isEmpty()
                ? tr("FFmpeg 未能将视频转换为 H.264。")
                : tr("FFmpeg 转换失败：\n%1").arg(details));
        return false;
    }

    if (!canDecodeFirstFrame(outputPath)) {
        QFile::remove(outputPath);
        QMessageBox::critical(
            this, tr("转换后仍无法读取"),
            tr("视频已经转换为 H.264，但当前 OpenCV 仍无法读取。"
               "请检查 OpenCV 的 FFmpeg 视频后端。"));
        return false;
    }

    *videoPath = QFileInfo(outputPath).absoluteFilePath();
    ui->videoPathEdit->setText(QDir::toNativeSeparators(*videoPath));
    ui->videoPathEdit->setToolTip(QDir::toNativeSeparators(*videoPath));
    ui->statusLabel->setText(tr("已自动转换为 H.264，请选择处理方案"));
    return true;
}

bool MainWindow::selectTrackingPoints(const QString &videoPath,
                                      const QString &schemeName,
                                      int maximumPoints,
                                      QVector<QPointF> *points)
{
    cv::VideoCapture capture(openCvPath(videoPath));
    if (!capture.isOpened()) {
        QMessageBox::critical(this, tr("读取失败"),
                              tr("OpenCV 无法打开所选视频。"));
        return false;
    }

    cv::Mat firstFrame;
    if (!capture.read(firstFrame) || firstFrame.empty()) {
        QMessageBox::critical(this, tr("读取失败"),
                              tr("无法读取视频第一帧。"));
        return false;
    }
    double fps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 0.0) {
        fps = 25.0;
    }

    PointSelectionDialog dialog(matToQImage(firstFrame), fps, schemeName,
                                maximumPoints, this);
    if (dialog.exec() != QDialog::Accepted) {
        return false;
    }
    *points = dialog.points();
    return !points->isEmpty();
}

void MainWindow::startWorker(InferenceWorker::Scheme scheme,
                             const QString &videoPath,
                             const QVector<QPointF> &tapPoints)
{
    const QString modelDirectory = resolveModelDirectory();
    workerThread_ = new QThread(this);
    worker_ = new InferenceWorker(scheme, videoPath, modelDirectory,
                                  tapPoints);
    worker_->moveToThread(workerThread_);

    connect(workerThread_, &QThread::started,
            worker_, &InferenceWorker::process);
    connect(worker_, &InferenceWorker::progressChanged,
            this, [this](int percent, const QString &message) {
                ui->processingProgressBar->setValue(
                    std::clamp(percent, 0, 100));
                ui->statusLabel->setText(message);
            });

    connect(worker_, &InferenceWorker::completed,
            this, [this](const QString &summary) {
                finishWorkerUi(summary, false);
            });
    connect(worker_, &InferenceWorker::failed,
            this, [this](const QString &message) {
                finishWorkerUi(message, true);
            });
    connect(worker_, &InferenceWorker::canceled, this, [this] {
        setProcessingState(false);
        ui->statusLabel->setText(tr("处理已取消"));
        ui->processingProgressBar->setValue(0);
    });

    connect(worker_, &InferenceWorker::completed,
            workerThread_, &QThread::quit);
    connect(worker_, &InferenceWorker::failed,
            workerThread_, &QThread::quit);
    connect(worker_, &InferenceWorker::canceled,
            workerThread_, &QThread::quit);
    connect(worker_, &InferenceWorker::completed,
            worker_, &QObject::deleteLater);
    connect(worker_, &InferenceWorker::failed,
            worker_, &QObject::deleteLater);
    connect(worker_, &InferenceWorker::canceled,
            worker_, &QObject::deleteLater);
    connect(workerThread_, &QThread::finished, this,
            [this, thread = workerThread_] {
                worker_ = nullptr;
                workerThread_ = nullptr;
                thread->deleteLater();
            });

    setProcessingState(true);
    ui->processingProgressBar->setValue(0);
    switch (scheme) {
    case InferenceWorker::Scheme::Ring:
        ui->statusLabel->setText(tr("正在启动圆环识别方案……"));
        break;
    case InferenceWorker::Scheme::LocoTrack:
        ui->statusLabel->setText(tr("正在启动 LocoTrack-B 通用跟踪方案……"));
        break;
    case InferenceWorker::Scheme::TrackOn2:
        ui->statusLabel->setText(tr("正在启动 Track-On2 通用跟踪方案……"));
        break;
    case InferenceWorker::Scheme::Tap:
        ui->statusLabel->setText(tr("正在启动 TAPNext++ 通用跟踪方案……"));
        break;
    }
    workerThread_->start();
}

void MainWindow::setProcessingState(bool processing)
{
    ui->browseButton->setEnabled(!processing);
    ui->ringSchemeButton->setEnabled(!processing);
    ui->localtrackButton->setEnabled(!processing);
    ui->trackonButton->setEnabled(!processing);
    ui->tapSchemeButton->setEnabled(!processing);
    ui->physicalAnalysisButton->setEnabled(!processing);
    ui->ReviewButton->setEnabled(!processing);
}

void MainWindow::finishWorkerUi(const QString &text, bool showAsError)
{
    setProcessingState(false);
    if (showAsError) {
        ui->statusLabel->setText(tr("处理失败"));
        QMessageBox::critical(this, tr("推理失败"), text);
        return;
    }

    ui->processingProgressBar->setValue(100);
    ui->statusLabel->setText(tr("处理完成"));
    QMessageBox::information(this, tr("处理完成"), text);
}

void MainWindow::stopWorker()
{
    if (worker_) {
        worker_->requestCancel();
    }
    if (workerThread_ && workerThread_->isRunning()) {
        workerThread_->requestInterruption();
        workerThread_->quit();
        workerThread_->wait();
    }
}
