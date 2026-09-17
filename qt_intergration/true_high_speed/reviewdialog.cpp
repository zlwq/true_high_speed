#include "reviewdialog.h"

#include <QCheckBox>
#include <QColor>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QElapsedTimer>
#include <QEnterEvent>
#include <QFile>
#include <QFileInfo>
#include <QFontMetrics>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPixmap>
#include <QPushButton>
#include <QRegularExpression>
#include <QResizeEvent>
#include <QScreen>
#include <QScrollArea>
#include <QSplitter>
#include <QStringConverter>
#include <QStringList>
#include <QTextStream>
#include <QTimer>
#include <QToolTip>
#include <QVBoxLayout>
#include <QVector>
#include <QWidget>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using OptionalValue = std::optional<double>;

struct ReviewMetric
{
    QString csvName;
    QString displayName;
    QString unit;
    QColor color;
};

struct ReviewRow
{
    int frameIndex = 0;
    double timestamp = 0.0;
    QVector<OptionalValue> values;
};

struct ReviewCsv
{
    QString sourceVideoPath;
    QString processedVideoPath;
    QVector<ReviewMetric> metrics;
    QVector<ReviewRow> rows;
};

std::string openCvPath(const QString &path)
{
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

bool parseFiniteNumber(const QString &text, double *value)
{
    bool ok = false;
    const double parsed = text.toDouble(&ok);
    if (!ok || !std::isfinite(parsed)) {
        return false;
    }
    *value = parsed;
    return true;
}

bool parsePositiveMetadata(const QString &line,
                           const QString &prefix)
{
    if (!line.startsWith(prefix)) {
        return false;
    }
    double value = 0.0;
    return parseFiniteNumber(line.mid(prefix.size()).trimmed(), &value)
        && value > 0.0;
}

std::optional<ReviewMetric> translateMetric(const QString &name,
                                            int colorIndex)
{
    static const QRegularExpression pointPattern(
        QStringLiteral("^id_(\\d+)_(speed_m_s|acceleration_m_s2)$"));
    static const QRegularExpression segmentPattern(
        QStringLiteral(
            "^segment_\\d+_id_(\\d+)_to_id_(\\d+)_"
            "(linear_speed_m_s|angular_velocity_rad_s|"
            "angular_acceleration_rad_s2)$"));

    QString displayName;
    QString unit;
    const QRegularExpressionMatch pointMatch = pointPattern.match(name);
    if (pointMatch.hasMatch()) {
        const QString id = pointMatch.captured(1);
        if (pointMatch.captured(2) == QStringLiteral("speed_m_s")) {
            displayName = QStringLiteral("ID %1 的线速度").arg(id);
            unit = QStringLiteral("m/s");
        } else {
            displayName = QStringLiteral("ID %1 的加速度").arg(id);
            unit = QStringLiteral("m/s²");
        }
    } else {
        const QRegularExpressionMatch segmentMatch = segmentPattern.match(name);
        if (!segmentMatch.hasMatch()) {
            return std::nullopt;
        }
        const QString firstId = segmentMatch.captured(1);
        const QString secondId = segmentMatch.captured(2);
        const QString type = segmentMatch.captured(3);
        const QString target = QStringLiteral("ID %1–ID %2")
                                   .arg(firstId, secondId);
        if (type == QStringLiteral("linear_speed_m_s")) {
            displayName = target + QStringLiteral(" 的线速度");
            unit = QStringLiteral("m/s");
        } else if (type == QStringLiteral("angular_velocity_rad_s")) {
            displayName = target + QStringLiteral(" 的角速度");
            unit = QStringLiteral("rad/s");
        } else {
            displayName = target + QStringLiteral(" 的角加速度");
            unit = QStringLiteral("rad/s²");
        }
    }

    ReviewMetric metric;
    metric.csvName = name;
    metric.displayName = displayName;
    metric.unit = unit;
    metric.color = QColor::fromHsv((colorIndex * 67) % 360, 190, 210);
    return metric;
}

bool parsePhysicalCsv(const QString &path, ReviewCsv *result, QString *error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        *error = QObject::tr("无法读取 CSV 文件：\n%1")
                     .arg(QDir::toNativeSeparators(path));
        return false;
    }

    QTextStream stream(&file);
    stream.setEncoding(QStringConverter::Utf8);
    QString sourceLine = stream.readLine();
    if (!sourceLine.isEmpty() && sourceLine.front() == QChar::ByteOrderMark) {
        sourceLine.remove(0, 1);
    }
    const QString sourcePrefix = QStringLiteral("#source_video_path=");
    if (!sourceLine.startsWith(sourcePrefix)) {
        *error = QObject::tr("第一行缺少未处理原视频路径。");
        return false;
    }
    result->sourceVideoPath = sourceLine.mid(sourcePrefix.size()).trimmed();

    const QString processedLine = stream.readLine();
    const QString processedPrefix = QStringLiteral("#processed_video_path=");
    if (!processedLine.startsWith(processedPrefix)) {
        *error = QObject::tr("第二行缺少处理后视频路径。");
        return false;
    }
    result->processedVideoPath =
        processedLine.mid(processedPrefix.size()).trimmed();
    if (!QDir::isAbsolutePath(result->sourceVideoPath)
        || !QDir::isAbsolutePath(result->processedVideoPath)) {
        *error = QObject::tr("CSV 中的视频路径必须是绝对路径。");
        return false;
    }

    if (!parsePositiveMetadata(stream.readLine(),
                               QStringLiteral("#pixels_per_meter="))) {
        *error = QObject::tr("第三行像素标定信息无效。");
        return false;
    }
    if (!parsePositiveMetadata(stream.readLine(),
                               QStringLiteral("#frames_per_second="))) {
        *error = QObject::tr("第四行时间标定信息无效。");
        return false;
    }
    if (stream.atEnd()) {
        *error = QObject::tr("CSV 缺少物理信息表头。");
        return false;
    }

    const QStringList header = stream.readLine().split(',', Qt::KeepEmptyParts);
    if (header.size() < 3
        || header.at(0) != QStringLiteral("frame_index")
        || header.at(1) != QStringLiteral("timestamp_sec")) {
        *error = QObject::tr("CSV 不是本程序生成的物理信息文件。");
        return false;
    }

    result->metrics.clear();
    for (int column = 2; column < header.size(); ++column) {
        const auto metric = translateMetric(header.at(column), column - 2);
        if (!metric.has_value()) {
            *error = QObject::tr("无法识别物理信息列：%1")
                         .arg(header.at(column));
            return false;
        }
        result->metrics.append(*metric);
    }

    result->rows.clear();
    int lineNumber = 5;
    int previousFrame = -1;
    double previousTime = -std::numeric_limits<double>::infinity();
    while (!stream.atEnd()) {
        ++lineNumber;
        const QString line = stream.readLine();
        if (line.trimmed().isEmpty()) {
            continue;
        }
        const QStringList fields = line.split(',', Qt::KeepEmptyParts);
        if (fields.size() != header.size()) {
            *error = QObject::tr("CSV 第 %1 行列数不正确。").arg(lineNumber);
            return false;
        }

        bool frameOk = false;
        ReviewRow row;
        row.frameIndex = fields.at(0).toInt(&frameOk);
        if (!frameOk || row.frameIndex < 0
            || (!result->rows.isEmpty()
                && row.frameIndex != previousFrame + 1)
            || !parseFiniteNumber(fields.at(1), &row.timestamp)
            || row.timestamp < previousTime) {
            *error = QObject::tr("CSV 第 %1 行的帧号或时间戳无效。")
                         .arg(lineNumber);
            return false;
        }
        previousFrame = row.frameIndex;
        previousTime = row.timestamp;

        row.values.resize(result->metrics.size());
        for (int metric = 0; metric < result->metrics.size(); ++metric) {
            const QString text = fields.at(metric + 2).trimmed();
            if (text.isEmpty()) {
                continue;
            }
            double value = 0.0;
            if (!parseFiniteNumber(text, &value)) {
                *error = QObject::tr("CSV 第 %1 行的 %2 数值无效。")
                             .arg(lineNumber)
                             .arg(result->metrics.at(metric).displayName);
                return false;
            }
            row.values[metric] = value;
        }
        result->rows.append(std::move(row));
    }

    if (result->rows.isEmpty()
        || result->rows.constFirst().frameIndex != 0) {
        *error = QObject::tr("CSV 必须包含从第 0 帧开始的物理信息数据。");
        return false;
    }
    return true;
}

QVector<int> selectMetrics(const ReviewCsv &csv, QWidget *parent)
{
    QDialog dialog(parent);
    dialog.setWindowTitle(QObject::tr("选择要显示的物理信息"));
    dialog.resize(560, 460);

    auto *instructions = new QLabel(
        QObject::tr("请选择需要同时查看的曲线："), &dialog);
    auto *content = new QWidget(&dialog);
    auto *contentLayout = new QVBoxLayout(content);
    QVector<QCheckBox *> checkBoxes;
    checkBoxes.reserve(csv.metrics.size());
    for (const ReviewMetric &metric : csv.metrics) {
        auto *checkBox = new QCheckBox(
            QStringLiteral("%1（%2）")
                .arg(metric.displayName, metric.unit), content);
        contentLayout->addWidget(checkBox);
        checkBoxes.append(checkBox);
    }
    contentLayout->addStretch();

    auto *scrollArea = new QScrollArea(&dialog);
    scrollArea->setWidgetResizable(true);
    scrollArea->setWidget(content);
    auto *buttons = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    auto *layout = new QVBoxLayout(&dialog);
    layout->addWidget(instructions);
    layout->addWidget(scrollArea, 1);
    layout->addWidget(buttons);

    QObject::connect(buttons, &QDialogButtonBox::accepted,
                     &dialog, [&dialog, &checkBoxes] {
        const bool anyChecked = std::any_of(
            checkBoxes.cbegin(), checkBoxes.cend(),
            [](const QCheckBox *checkBox) { return checkBox->isChecked(); });
        if (!anyChecked) {
            QMessageBox::warning(&dialog, QObject::tr("尚未选择"),
                                 QObject::tr("请至少选择一项物理信息。"));
            return;
        }
        dialog.accept();
    });
    QObject::connect(buttons, &QDialogButtonBox::rejected,
                     &dialog, &QDialog::reject);

    if (dialog.exec() != QDialog::Accepted) {
        return {};
    }
    QVector<int> selected;
    for (int index = 0; index < checkBoxes.size(); ++index) {
        if (checkBoxes.at(index)->isChecked()) {
            selected.append(index);
        }
    }
    return selected;
}

class PhysicalPlotWidget final : public QWidget
{
public:
    PhysicalPlotWidget(const ReviewCsv &csv,
                       QVector<int> selectedMetrics,
                       QWidget *parent = nullptr)
        : QWidget(parent)
        , csv_(csv)
        , selectedMetrics_(std::move(selectedMetrics))
    {
        setMouseTracking(true);
        setMinimumWidth(320);
        setMinimumHeight(std::max(
            320, static_cast<int>(selectedMetrics_.size()) * 150 + 50));
        cursorTime_ = csv_.rows.constFirst().timestamp;
    }

    void setPlaying(bool playing, bool unlockWhenPaused = false)
    {
        playing_ = playing;
        if (playing_) {
            locked_ = true;
        } else if (unlockWhenPaused) {
            locked_ = false;
        }
        update();
    }

    void setCurrentFrame(int frameIndex)
    {
        const int row = nearestRowForFrame(frameIndex);
        if (row >= 0) {
            cursorTime_ = csv_.rows.at(row).timestamp;
            update();
        }
    }

    std::function<void(int)> frameClicked;

protected:
    void paintEvent(QPaintEvent *) override
    {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.fillRect(rect(), QColor(250, 251, 253));

        const QRectF entirePlot = plotArea();
        if (entirePlot.width() <= 1.0 || entirePlot.height() <= 1.0) {
            return;
        }
        const int count = selectedMetrics_.size();
        const qreal gap = 12.0;
        const qreal plotHeight =
            (entirePlot.height() - gap * (count - 1)) / count;

        for (int selected = 0; selected < count; ++selected) {
            const int metricIndex = selectedMetrics_.at(selected);
            const ReviewMetric &metric = csv_.metrics.at(metricIndex);
            const QRectF area(entirePlot.left(),
                              entirePlot.top() + selected * (plotHeight + gap),
                              entirePlot.width(), plotHeight);
            drawMetric(painter, area, metricIndex, metric);
        }

        const qreal x = xForTime(cursorTime_, entirePlot);
        painter.setPen(QPen(QColor(220, 45, 55), 2.0));
        painter.drawLine(QPointF(x, entirePlot.top()),
                         QPointF(x, entirePlot.bottom()));

        painter.setPen(QColor(65, 70, 80));
        painter.drawText(QRectF(entirePlot.left(), entirePlot.bottom() + 8.0,
                                entirePlot.width(), 24.0),
                         Qt::AlignCenter,
                         QObject::tr("时间（秒）"));
        painter.drawText(QRectF(entirePlot.left() - 35.0,
                                entirePlot.bottom() + 5.0, 70.0, 20.0),
                         Qt::AlignCenter,
                         QString::number(firstTime(), 'f', 3));
        painter.drawText(QRectF(entirePlot.right() - 35.0,
                                entirePlot.bottom() + 5.0, 70.0, 20.0),
                         Qt::AlignCenter,
                         QString::number(lastTime(), 'f', 3));
    }

    void mouseMoveEvent(QMouseEvent *event) override
    {
        const QRectF area = plotArea();
        const int row = rowAtX(event->position().x(), area);
        if (row < 0) {
            QToolTip::hideText();
            return;
        }
        if (!playing_ && !locked_) {
            cursorTime_ = csv_.rows.at(row).timestamp;
            update();
        }

        QStringList values;
        for (int metricIndex : selectedMetrics_) {
            const OptionalValue &value = csv_.rows.at(row).values.at(metricIndex);
            if (value.has_value()) {
                const ReviewMetric &metric = csv_.metrics.at(metricIndex);
                values << QStringLiteral("%1：%2 %3")
                              .arg(metric.displayName,
                                   QString::number(*value, 'g', 9), metric.unit);
            }
        }
        if (values.isEmpty()) {
            QToolTip::hideText();
        } else {
            values.prepend(QObject::tr("第 %1 帧，%2 秒")
                               .arg(csv_.rows.at(row).frameIndex)
                               .arg(csv_.rows.at(row).timestamp, 0, 'f', 6));
            QToolTip::showText(event->globalPosition().toPoint(),
                               values.join('\n'), this);
        }
    }

    void enterEvent(QEnterEvent *event) override
    {
        if (!playing_) {
            locked_ = false;
            const int row = rowAtX(event->position().x(), plotArea());
            if (row >= 0) {
                cursorTime_ = csv_.rows.at(row).timestamp;
                update();
            }
        }
        QWidget::enterEvent(event);
    }

    void mousePressEvent(QMouseEvent *event) override
    {
        if (event->button() != Qt::LeftButton) {
            QWidget::mousePressEvent(event);
            return;
        }
        const int row = rowAtX(event->position().x(), plotArea());
        if (row < 0) {
            return;
        }
        locked_ = true;
        cursorTime_ = csv_.rows.at(row).timestamp;
        update();
        if (frameClicked) {
            frameClicked(csv_.rows.at(row).frameIndex);
        }
    }

    void leaveEvent(QEvent *event) override
    {
        QToolTip::hideText();
        QWidget::leaveEvent(event);
    }

private:
    QRectF plotArea() const
    {
        return QRectF(230.0, 20.0,
                      std::max(1, width() - 255),
                      std::max(1, height() - 70));
    }

    double firstTime() const
    {
        return csv_.rows.constFirst().timestamp;
    }

    double lastTime() const
    {
        const double last = csv_.rows.constLast().timestamp;
        return last > firstTime() ? last : firstTime() + 1.0;
    }

    qreal xForTime(double time, const QRectF &area) const
    {
        const double ratio = std::clamp(
            (time - firstTime()) / (lastTime() - firstTime()), 0.0, 1.0);
        return area.left() + ratio * area.width();
    }

    int rowAtX(qreal x, const QRectF &area) const
    {
        if (x < area.left() || x > area.right()) {
            return -1;
        }
        const double ratio = (x - area.left()) / area.width();
        const double time = firstTime() + ratio * (lastTime() - firstTime());
        auto found = std::lower_bound(
            csv_.rows.cbegin(), csv_.rows.cend(), time,
            [](const ReviewRow &row, double target) {
                return row.timestamp < target;
            });
        if (found == csv_.rows.cbegin()) {
            return 0;
        }
        if (found == csv_.rows.cend()) {
            return csv_.rows.size() - 1;
        }
        const int upper = static_cast<int>(found - csv_.rows.cbegin());
        const int lower = upper - 1;
        return std::abs(csv_.rows.at(lower).timestamp - time)
                    <= std::abs(csv_.rows.at(upper).timestamp - time)
            ? lower : upper;
    }

    int nearestRowForFrame(int frameIndex) const
    {
        auto found = std::lower_bound(
            csv_.rows.cbegin(), csv_.rows.cend(), frameIndex,
            [](const ReviewRow &row, int target) {
                return row.frameIndex < target;
            });
        if (found == csv_.rows.cend()) {
            return csv_.rows.size() - 1;
        }
        return static_cast<int>(found - csv_.rows.cbegin());
    }

    void drawMetric(QPainter &painter,
                    const QRectF &area,
                    int metricIndex,
                    const ReviewMetric &metric)
    {
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();
        for (const ReviewRow &row : csv_.rows) {
            const OptionalValue &value = row.values.at(metricIndex);
            if (value.has_value()) {
                minimum = std::min(minimum, *value);
                maximum = std::max(maximum, *value);
            }
        }

        painter.setPen(QPen(QColor(205, 210, 220), 1.0));
        painter.setBrush(Qt::white);
        painter.drawRect(area);
        const QString label = QStringLiteral("%1（%2）")
                                  .arg(metric.displayName, metric.unit);
        painter.setPen(metric.color);
        const QString elided = painter.fontMetrics().elidedText(
            label, Qt::ElideRight, static_cast<int>(area.left() - 16.0));
        painter.drawText(QRectF(4.0, area.top(), area.left() - 12.0,
                                area.height()),
                         Qt::AlignVCenter | Qt::AlignRight, elided);

        if (!std::isfinite(minimum) || !std::isfinite(maximum)) {
            painter.setPen(QColor(130, 135, 145));
            painter.drawText(area, Qt::AlignCenter,
                             QObject::tr("没有有效数据"));
            return;
        }
        if (std::abs(maximum - minimum) < 1.0e-12) {
            const double padding = std::max(1.0, std::abs(maximum) * 0.1);
            minimum -= padding;
            maximum += padding;
        } else {
            const double padding = (maximum - minimum) * 0.08;
            minimum -= padding;
            maximum += padding;
        }

        painter.setPen(QColor(105, 110, 120));
        painter.drawText(QRectF(area.left() + 4.0, area.top() + 2.0,
                                100.0, 18.0),
                         Qt::AlignLeft,
                         QString::number(maximum, 'g', 5));
        painter.drawText(QRectF(area.left() + 4.0, area.bottom() - 20.0,
                                100.0, 18.0),
                         Qt::AlignLeft,
                         QString::number(minimum, 'g', 5));

        QPainterPath path;
        bool continuing = false;
        for (const ReviewRow &row : csv_.rows) {
            const OptionalValue &value = row.values.at(metricIndex);
            if (!value.has_value()) {
                continuing = false;
                continue;
            }
            const qreal x = xForTime(row.timestamp, area);
            const qreal ratio = (*value - minimum) / (maximum - minimum);
            const qreal y = area.bottom() - ratio * area.height();
            if (continuing) {
                path.lineTo(x, y);
            } else {
                path.moveTo(x, y);
                continuing = true;
            }
        }
        painter.setClipRect(area.adjusted(1.0, 1.0, -1.0, -1.0));
        painter.setPen(QPen(metric.color, 2.0));
        painter.setBrush(Qt::NoBrush);
        painter.drawPath(path);
        painter.setClipping(false);
    }

    const ReviewCsv &csv_;
    QVector<int> selectedMetrics_;
    double cursorTime_ = 0.0;
    bool playing_ = false;
    bool locked_ = false;
};

class ReviewDialog final : public QDialog
{
public:
    ReviewDialog(ReviewCsv csv,
                 QVector<int> selectedMetrics,
                 QWidget *parent = nullptr)
        : QDialog(parent)
        , csv_(std::move(csv))
        , selectedMetrics_(std::move(selectedMetrics))
    {
        setWindowTitle(tr("视频帧与物理信息曲线"));

        if (!capture_.open(openCvPath(csv_.processedVideoPath))) {
            throw std::runtime_error("OpenCV 无法打开处理后视频");
        }
        fps_ = capture_.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps_) || fps_ <= 0.0) {
            fps_ = 25.0;
        }
        frameCount_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_COUNT));
        if (frameCount_ <= 0) {
            frameCount_ = csv_.rows.constLast().frameIndex + 1;
        }

        videoLabel_ = new QLabel(this);
        videoLabel_->setAlignment(Qt::AlignCenter);
        videoLabel_->setMinimumSize(320, 180);
        videoLabel_->setStyleSheet(QStringLiteral("background: black;"));
        playButton_ = new QPushButton(tr("播放"), this);
        frameLabel_ = new QLabel(this);

        auto *controls = new QHBoxLayout;
        controls->addWidget(playButton_);
        controls->addWidget(frameLabel_, 1);
        auto *videoPanel = new QWidget(this);
        auto *videoLayout = new QVBoxLayout(videoPanel);
        videoLayout->setContentsMargins(0, 0, 0, 0);
        videoLayout->addWidget(videoLabel_, 1);
        videoLayout->addLayout(controls);

        plot_ = new PhysicalPlotWidget(csv_, selectedMetrics_, this);
        auto *plotScroll = new QScrollArea(this);
        plotScroll->setWidgetResizable(true);
        plotScroll->setWidget(plot_);

        auto *splitter = new QSplitter(Qt::Vertical, this);
        splitter->addWidget(videoPanel);
        splitter->addWidget(plotScroll);
        splitter->setStretchFactor(0, 1);
        splitter->setStretchFactor(1, 1);

        auto *layout = new QVBoxLayout(this);
        layout->addWidget(splitter);

        timer_.setTimerType(Qt::PreciseTimer);
        timer_.setInterval(15);
        connect(&timer_, &QTimer::timeout, this, [this] { playbackTick(); });
        connect(playButton_, &QPushButton::clicked,
                this, [this] { togglePlayback(); });
        plot_->frameClicked = [this](int frameIndex) {
            pausePlayback(false);
            showFrame(frameIndex, false);
        };

        if (!showFrame(0, false)) {
            throw std::runtime_error("无法读取处理后视频第一帧");
        }

        QScreen *targetScreen = parent != nullptr
            ? parent->screen()
            : QGuiApplication::primaryScreen();
        if (targetScreen != nullptr) {
            const QRect screenGeometry = targetScreen->geometry();
            const int dialogWidth = screenGeometry.width() / 2;
            const int dialogHeight = screenGeometry.height() / 2;
            setGeometry(screenGeometry.x()
                            + (screenGeometry.width() - dialogWidth) / 2,
                        screenGeometry.y()
                            + (screenGeometry.height() - dialogHeight) / 2,
                        dialogWidth,
                        dialogHeight);
        } else {
            resize(960, 540);
        }
    }

    ~ReviewDialog() override
    {
        timer_.stop();
        capture_.release();
    }

protected:
    void resizeEvent(QResizeEvent *event) override
    {
        QDialog::resizeEvent(event);
        renderCurrentImage();
    }

private:
    void togglePlayback()
    {
        if (playing_) {
            pausePlayback(true);
            return;
        }
        if (currentFrame_ >= frameCount_ - 1) {
            if (!showFrame(0, false)) {
                return;
            }
        }
        playing_ = true;
        playbackStartFrame_ = currentFrame_;
        playbackClock_.restart();
        playButton_->setText(tr("暂停"));
        plot_->setPlaying(true);
        timer_.start();
    }

    void pausePlayback(bool unlockPlot)
    {
        if (!playing_) {
            plot_->setPlaying(false, unlockPlot);
            return;
        }
        playing_ = false;
        timer_.stop();
        playButton_->setText(tr("播放"));
        plot_->setPlaying(false, unlockPlot);
    }

    void playbackTick()
    {
        const qint64 elapsed = playbackClock_.elapsed();
        int targetFrame = playbackStartFrame_
            + static_cast<int>(std::floor(elapsed * fps_ / 1000.0));
        if (targetFrame >= frameCount_) {
            targetFrame = frameCount_ - 1;
        }
        if (targetFrame != currentFrame_
            && !showFrame(targetFrame, true)) {
            pausePlayback(true);
            return;
        }
        if (currentFrame_ >= frameCount_ - 1) {
            pausePlayback(true);
        }
    }

    bool showFrame(int frameIndex, bool allowSequentialRead)
    {
        frameIndex = std::clamp(frameIndex, 0, frameCount_ - 1);
        cv::Mat frame;
        const int distance = frameIndex - currentFrame_;
        if (allowSequentialRead && distance > 0 && distance <= 3) {
            for (int step = 0; step < distance; ++step) {
                if (!capture_.read(frame) || frame.empty()) {
                    return false;
                }
            }
        } else {
            capture_.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
            if (!capture_.read(frame) || frame.empty()) {
                return false;
            }
        }

        currentFrame_ = frameIndex;
        currentImage_ = matToQImage(frame);
        renderCurrentImage();
        plot_->setCurrentFrame(currentFrame_);
        frameLabel_->setText(
            tr("第 %1 / %2 帧　%3 秒")
                .arg(currentFrame_)
                .arg(frameCount_ - 1)
                .arg(currentFrame_ / fps_, 0, 'f', 6));
        return true;
    }

    void renderCurrentImage()
    {
        if (currentImage_.isNull() || !videoLabel_) {
            return;
        }
        videoLabel_->setPixmap(QPixmap::fromImage(currentImage_).scaled(
            videoLabel_->size(), Qt::KeepAspectRatio,
            Qt::SmoothTransformation));
    }

    ReviewCsv csv_;
    QVector<int> selectedMetrics_;
    cv::VideoCapture capture_;
    double fps_ = 25.0;
    int frameCount_ = 0;
    int currentFrame_ = -1;
    int playbackStartFrame_ = 0;
    bool playing_ = false;
    QImage currentImage_;
    QLabel *videoLabel_ = nullptr;
    QLabel *frameLabel_ = nullptr;
    QPushButton *playButton_ = nullptr;
    PhysicalPlotWidget *plot_ = nullptr;
    QTimer timer_;
    QElapsedTimer playbackClock_;
};

} // namespace

void openReviewDialog(const QString &csvPath, QWidget *parent)
{
    ReviewCsv csv;
    QString error;
    if (!parsePhysicalCsv(csvPath, &csv, &error)) {
        QMessageBox::warning(parent, QObject::tr("CSV 格式不符合要求"), error);
        return;
    }

    const QFileInfo processedVideo(csv.processedVideoPath);
    if (!processedVideo.isFile() || !processedVideo.isReadable()) {
        QMessageBox::warning(
            parent, QObject::tr("找不到处理后视频"),
            QObject::tr("找不到 CSV 第二行记录的处理后视频：\n%1")
                .arg(QDir::toNativeSeparators(csv.processedVideoPath)));
        return;
    }

    const QVector<int> selectedMetrics = selectMetrics(csv, parent);
    if (selectedMetrics.isEmpty()) {
        return;
    }

    try {
        ReviewDialog dialog(std::move(csv), selectedMetrics, parent);
        dialog.exec();
    } catch (const std::exception &exception) {
        QMessageBox::critical(
            parent, QObject::tr("无法打开回看窗口"),
            QObject::tr("%1").arg(QString::fromUtf8(exception.what())));
    }
}
