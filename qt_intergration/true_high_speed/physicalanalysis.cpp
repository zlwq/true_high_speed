#include "physicalanalysis.h"

#include "csvmetadata.h"

#include <QApplication>
#include <QBrush>
#include <QColor>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QDoubleValidator>
#include <QFile>
#include <QFileInfo>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QGraphicsSimpleTextItem>
#include <QGraphicsView>
#include <QGridLayout>
#include <QHash>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QLineF>
#include <QLineEdit>
#include <QList>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPair>
#include <QPen>
#include <QPointF>
#include <QPixmap>
#include <QPushButton>
#include <QRegularExpression>
#include <QResizeEvent>
#include <QSet>
#include <QShowEvent>
#include <QStringConverter>
#include <QStringList>
#include <QTextStream>
#include <QVBoxLayout>
#include <QVector>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace {

using OptionalPoint = std::optional<QPointF>;
using OptionalValue = std::optional<double>;

struct CsvRow {
    int frameIndex = 0;
    double sourceTimestamp = 0.0;
    QVector<OptionalPoint> points;
};

struct TrackingCsv {
    QString sourceVideoPath;
    QString processedVideoPath;
    QVector<int> pointIds;
    QVector<CsvRow> rows;
};

struct Marker {
    int id = 0;
    QPointF position;
};

struct SegmentAnalysis {
    int firstId = 0;
    int secondId = 0;
    QVector<OptionalPoint> midpoint;
    QVector<OptionalPoint> midpointVelocity;
    QVector<OptionalValue> angle;
    QVector<OptionalValue> angularVelocity;
    QVector<OptionalValue> angularAcceleration;
};

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

QImage matToQImage(const cv::Mat &bgr)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows,
                  static_cast<int>(rgb.step), QImage::Format_RGB888)
        .copy();
}

bool parseNumber(const QString &text, double *value)
{
    bool ok = false;
    const double result = text.toDouble(&ok);
    if (!ok || !std::isfinite(result)) {
        return false;
    }
    *value = result;
    return true;
}

bool parseTrackingCsv(const QString &path, TrackingCsv *result,
                      QString *error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        *error = QObject::tr("无法读取 CSV 文件：\n%1")
                     .arg(QDir::toNativeSeparators(path));
        return false;
    }

    QTextStream stream(&file);
    stream.setEncoding(QStringConverter::Utf8);
    QString metadata = stream.readLine();
    if (!metadata.isEmpty() && metadata.front() == QChar::ByteOrderMark) {
        metadata.remove(0, 1);
    }
    const QString metadataPrefix = QStringLiteral("#source_video_path=");
    if (!metadata.startsWith(metadataPrefix)) {
        *error = QObject::tr(
            "CSV 缺少源视频全局路径，不是当前版本生成的坐标文件。");
        return false;
    }
    result->sourceVideoPath = metadata.mid(metadataPrefix.size()).trimmed();
    if (result->sourceVideoPath.isEmpty()
        || !QDir::isAbsolutePath(result->sourceVideoPath)) {
        *error = QObject::tr("CSV 中的源视频路径不是有效的全局路径。");
        return false;
    }

    if (stream.atEnd()) {
        *error = QObject::tr("CSV 缺少处理后视频路径。");
        return false;
    }
    QString processedMetadata = stream.readLine();
    const QString processedPrefix = QStringLiteral("#processed_video_path=");
    if (!processedMetadata.startsWith(processedPrefix)) {
        *error = QObject::tr("CSV 缺少处理后视频路径，不是当前版本生成的坐标文件。");
        return false;
    }
    result->processedVideoPath =
        processedMetadata.mid(processedPrefix.size()).trimmed();
    if (result->processedVideoPath.isEmpty()
        || !QDir::isAbsolutePath(result->processedVideoPath)) {
        *error = QObject::tr("CSV 中的处理后视频路径不是有效的全局路径。");
        return false;
    }

    if (stream.atEnd()) {
        *error = QObject::tr("CSV 缺少坐标表头。");
        return false;
    }
    const QStringList header = stream.readLine().split(',', Qt::KeepEmptyParts);
    if (header.size() < 4 || header.size() % 2 != 0
        || header.at(0) != QStringLiteral("frame_index")
        || (header.at(1) != QStringLiteral("timestamp_sec")
            && header.at(1) != QStringLiteral("video_timestamp_sec"))) {
        *error = QObject::tr("CSV 表头格式不符合坐标文件要求。");
        return false;
    }

    const QRegularExpression xPattern(QStringLiteral("^id_(\\d+)_x_px$"));
    result->pointIds.clear();
    for (int column = 2; column < header.size(); column += 2) {
        const QRegularExpressionMatch match = xPattern.match(header.at(column));
        if (!match.hasMatch()) {
            *error = QObject::tr("CSV 第 %1 列不是目标点 X 坐标。")
                         .arg(column + 1);
            return false;
        }
        bool idOk = false;
        const int id = match.captured(1).toInt(&idOk);
        const QString expectedY = QStringLiteral("id_%1_y_px").arg(id);
        if (!idOk || id <= 0 || header.at(column + 1) != expectedY
            || result->pointIds.contains(id)) {
            *error = QObject::tr("CSV 中 ID %1 的 X/Y 坐标列不成对。")
                         .arg(id);
            return false;
        }
        result->pointIds.append(id);
    }

    result->rows.clear();
    int lineNumber = 3;
    int previousFrame = std::numeric_limits<int>::min();
    while (!stream.atEnd()) {
        ++lineNumber;
        const QString line = stream.readLine();
        if (line.trimmed().isEmpty()) {
            continue;
        }
        const QStringList fields = line.split(',', Qt::KeepEmptyParts);
        if (fields.size() != header.size()) {
            *error = QObject::tr("CSV 第 %1 行的列数不正确。").arg(lineNumber);
            return false;
        }

        bool frameOk = false;
        CsvRow row;
        row.frameIndex = fields.at(0).toInt(&frameOk);
        if (!frameOk
            || (!result->rows.isEmpty()
                && row.frameIndex != previousFrame + 1)
            || !parseNumber(fields.at(1), &row.sourceTimestamp)) {
            *error = QObject::tr("CSV 第 %1 行的帧号或时间戳无效。")
                         .arg(lineNumber);
            return false;
        }
        previousFrame = row.frameIndex;
        row.points.resize(result->pointIds.size());
        for (int point = 0; point < result->pointIds.size(); ++point) {
            const QString xText = fields.at(2 + point * 2).trimmed();
            const QString yText = fields.at(3 + point * 2).trimmed();
            if (xText.isEmpty() && yText.isEmpty()) {
                continue;
            }
            double x = 0.0;
            double y = 0.0;
            if (xText.isEmpty() || yText.isEmpty()
                || !parseNumber(xText, &x) || !parseNumber(yText, &y)) {
                *error = QObject::tr("CSV 第 %1 行中 ID %2 的坐标无效。")
                             .arg(lineNumber)
                             .arg(result->pointIds.at(point));
                return false;
            }
            row.points[point] = QPointF(x, y);
        }
        result->rows.append(std::move(row));
    }

    if (result->rows.isEmpty()) {
        *error = QObject::tr("CSV 中没有坐标数据。");
        return false;
    }
    if (result->rows.constFirst().frameIndex != 0) {
        *error = QObject::tr("CSV 数据必须从第 0 帧开始。");
        return false;
    }
    return true;
}

QVector<OptionalPoint> interpolateShortGaps(QVector<OptionalPoint> values,
                                             const QVector<double> &times)
{
    int index = 0;
    while (index < values.size()) {
        if (values.at(index).has_value()) {
            ++index;
            continue;
        }
        const int gapStart = index;
        while (index < values.size() && !values.at(index).has_value()) {
            ++index;
        }
        const int gapEnd = index;
        const int gapLength = gapEnd - gapStart;
        if (gapLength > 3 || gapStart == 0 || gapEnd >= values.size()
            || !values.at(gapStart - 1).has_value()
            || !values.at(gapEnd).has_value()) {
            continue;
        }

        const double startTime = times.at(gapStart - 1);
        const double endTime = times.at(gapEnd);
        const double duration = endTime - startTime;
        if (!(duration > 0.0)) {
            continue;
        }
        const QPointF startPoint = *values.at(gapStart - 1);
        const QPointF endPoint = *values.at(gapEnd);
        for (int fill = gapStart; fill < gapEnd; ++fill) {
            const double ratio = (times.at(fill) - startTime) / duration;
            values[fill] = startPoint + (endPoint - startPoint) * ratio;
        }
    }
    return values;
}

QVector<OptionalPoint> pointDerivative(const QVector<OptionalPoint> &values,
                                        const QVector<double> &times)
{
    QVector<OptionalPoint> derivative(values.size());
    for (int index = 0; index < values.size(); ++index) {
        if (!values.at(index).has_value()) {
            continue;
        }
        int first = -1;
        int second = -1;
        if (index > 0 && index + 1 < values.size()
            && values.at(index - 1).has_value()
            && values.at(index + 1).has_value()) {
            first = index - 1;
            second = index + 1;
        } else if (index + 1 < values.size()
                   && values.at(index + 1).has_value()) {
            first = index;
            second = index + 1;
        } else if (index > 0 && values.at(index - 1).has_value()) {
            first = index - 1;
            second = index;
        }
        if (first < 0) {
            continue;
        }
        const double dt = times.at(second) - times.at(first);
        if (dt > 0.0) {
            derivative[index] = (*values.at(second) - *values.at(first)) / dt;
        }
    }
    return derivative;
}

QVector<OptionalValue> scalarDerivative(const QVector<OptionalValue> &values,
                                         const QVector<double> &times)
{
    QVector<OptionalValue> derivative(values.size());
    for (int index = 0; index < values.size(); ++index) {
        if (!values.at(index).has_value()) {
            continue;
        }
        int first = -1;
        int second = -1;
        if (index > 0 && index + 1 < values.size()
            && values.at(index - 1).has_value()
            && values.at(index + 1).has_value()) {
            first = index - 1;
            second = index + 1;
        } else if (index + 1 < values.size()
                   && values.at(index + 1).has_value()) {
            first = index;
            second = index + 1;
        } else if (index > 0 && values.at(index - 1).has_value()) {
            first = index - 1;
            second = index;
        }
        if (first < 0) {
            continue;
        }
        const double dt = times.at(second) - times.at(first);
        if (dt > 0.0) {
            derivative[index] =
                (*values.at(second) - *values.at(first)) / dt;
        }
    }
    return derivative;
}

QVector<OptionalPoint> pointSecondDerivative(
    const QVector<OptionalPoint> &values,
    const QVector<double> &times)
{
    QVector<OptionalPoint> derivative(values.size());
    for (int index = 0; index < values.size(); ++index) {
        if (!values.at(index).has_value()) {
            continue;
        }
        int first = -1;
        int middle = -1;
        int last = -1;
        if (index > 0 && index + 1 < values.size()
            && values.at(index - 1).has_value()
            && values.at(index + 1).has_value()) {
            first = index - 1;
            middle = index;
            last = index + 1;
        } else if (index + 2 < values.size()
                   && values.at(index + 1).has_value()
                   && values.at(index + 2).has_value()) {
            first = index;
            middle = index + 1;
            last = index + 2;
        } else if (index >= 2 && values.at(index - 1).has_value()
                   && values.at(index - 2).has_value()) {
            first = index - 2;
            middle = index - 1;
            last = index;
        }
        if (first < 0) {
            continue;
        }
        const double firstDt = times.at(middle) - times.at(first);
        const double secondDt = times.at(last) - times.at(middle);
        const double totalDt = times.at(last) - times.at(first);
        if (firstDt > 0.0 && secondDt > 0.0 && totalDt > 0.0) {
            const QPointF firstSlope =
                (*values.at(middle) - *values.at(first)) / firstDt;
            const QPointF secondSlope =
                (*values.at(last) - *values.at(middle)) / secondDt;
            derivative[index] = (secondSlope - firstSlope) * (2.0 / totalDt);
        }
    }
    return derivative;
}

QVector<OptionalValue> scalarSecondDerivative(
    const QVector<OptionalValue> &values,
    const QVector<double> &times)
{
    QVector<OptionalValue> derivative(values.size());
    for (int index = 0; index < values.size(); ++index) {
        if (!values.at(index).has_value()) {
            continue;
        }
        int first = -1;
        int middle = -1;
        int last = -1;
        if (index > 0 && index + 1 < values.size()
            && values.at(index - 1).has_value()
            && values.at(index + 1).has_value()) {
            first = index - 1;
            middle = index;
            last = index + 1;
        } else if (index + 2 < values.size()
                   && values.at(index + 1).has_value()
                   && values.at(index + 2).has_value()) {
            first = index;
            middle = index + 1;
            last = index + 2;
        } else if (index >= 2 && values.at(index - 1).has_value()
                   && values.at(index - 2).has_value()) {
            first = index - 2;
            middle = index - 1;
            last = index;
        }
        if (first < 0) {
            continue;
        }
        const double firstDt = times.at(middle) - times.at(first);
        const double secondDt = times.at(last) - times.at(middle);
        const double totalDt = times.at(last) - times.at(first);
        if (firstDt > 0.0 && secondDt > 0.0 && totalDt > 0.0) {
            const double firstSlope =
                (*values.at(middle) - *values.at(first)) / firstDt;
            const double secondSlope =
                (*values.at(last) - *values.at(middle)) / secondDt;
            derivative[index] =
                (secondSlope - firstSlope) * (2.0 / totalDt);
        }
    }
    return derivative;
}

void writeValue(std::ofstream &stream, const OptionalValue &value)
{
    stream << ',';
    if (value.has_value() && std::isfinite(*value)) {
        stream << *value;
    }
}

bool generatePhysicalCsv(const QString &outputPath,
                         const TrackingCsv &csv,
                         const QList<int> &selectedIds,
                         const QVector<QPair<int, int>> &segments,
                         double pixelsPerMeter,
                         double framesPerSecond,
                         QString *error)
{
    const int rowCount = csv.rows.size();
    QVector<double> times(rowCount);
    for (int row = 0; row < rowCount; ++row) {
        times[row] = csv.rows.at(row).frameIndex / framesPerSecond;
    }

    QSet<int> requiredIds;
    for (int id : selectedIds) {
        requiredIds.insert(id);
    }
    for (const auto &segment : segments) {
        requiredIds.insert(segment.first);
        requiredIds.insert(segment.second);
    }

    QHash<int, QVector<OptionalPoint>> tracks;
    for (int id : requiredIds) {
        const int pointIndex = csv.pointIds.indexOf(id);
        if (pointIndex < 0) {
            *error = QObject::tr("CSV 中找不到 ID %1。").arg(id);
            return false;
        }
        QVector<OptionalPoint> values(rowCount);
        for (int row = 0; row < rowCount; ++row) {
            values[row] = csv.rows.at(row).points.at(pointIndex);
        }
        tracks.insert(id, interpolateShortGaps(std::move(values), times));
    }

    QHash<int, QVector<OptionalPoint>> velocities;
    QHash<int, QVector<OptionalPoint>> accelerations;
    for (int id : selectedIds) {
        const QVector<OptionalPoint> velocity = pointDerivative(tracks[id], times);
        velocities.insert(id, velocity);
        accelerations.insert(id, pointSecondDerivative(tracks[id], times));
    }

    QVector<SegmentAnalysis> segmentResults;
    segmentResults.reserve(segments.size());
    constexpr double pi = 3.14159265358979323846;
    for (const auto &segment : segments) {
        SegmentAnalysis result;
        result.firstId = segment.first;
        result.secondId = segment.second;
        result.midpoint.resize(rowCount);
        result.angle.resize(rowCount);

        bool hasPreviousAngle = false;
        double previousAngle = 0.0;
        const auto &firstTrack = tracks[segment.first];
        const auto &secondTrack = tracks[segment.second];
        for (int row = 0; row < rowCount; ++row) {
            if (!firstTrack.at(row).has_value()
                || !secondTrack.at(row).has_value()) {
                hasPreviousAngle = false;
                continue;
            }
            const QPointF difference =
                *secondTrack.at(row) - *firstTrack.at(row);
            result.midpoint[row] =
                (*firstTrack.at(row) + *secondTrack.at(row)) / 2.0;
            if (std::hypot(difference.x(), difference.y()) <= 1.0e-9) {
                hasPreviousAngle = false;
                continue;
            }
            double angle = std::atan2(difference.y(), difference.x());
            if (hasPreviousAngle) {
                while (angle - previousAngle > pi) {
                    angle -= 2.0 * pi;
                }
                while (angle - previousAngle < -pi) {
                    angle += 2.0 * pi;
                }
            }
            result.angle[row] = angle;
            previousAngle = angle;
            hasPreviousAngle = true;
        }
        result.midpointVelocity = pointDerivative(result.midpoint, times);
        result.angularVelocity = scalarDerivative(result.angle, times);
        result.angularAcceleration =
            scalarSecondDerivative(result.angle, times);
        segmentResults.append(std::move(result));
    }

    std::ofstream stream(fileSystemPath(outputPath),
                         std::ios::binary | std::ios::trunc);
    if (!stream) {
        *error = QObject::tr("无法创建物理信息 CSV：\n%1")
                     .arg(QDir::toNativeSeparators(outputPath));
        return false;
    }
    writeCsvSourceVideoPath(stream, csv.sourceVideoPath);
    writeCsvProcessedVideoPath(stream, csv.processedVideoPath);
    stream << std::fixed << std::setprecision(9)
           << "#pixels_per_meter=" << pixelsPerMeter << '\n'
           << "#frames_per_second=" << framesPerSecond << '\n';
    stream << "frame_index,timestamp_sec";
    for (int id : selectedIds) {
        stream << ",id_" << id << "_speed_m_s"
               << ",id_" << id << "_acceleration_m_s2";
    }
    for (int index = 0; index < segmentResults.size(); ++index) {
        const auto &segment = segmentResults.at(index);
        const std::string prefix =
            "segment_" + std::to_string(index + 1)
            + "_id_" + std::to_string(segment.firstId)
            + "_to_id_" + std::to_string(segment.secondId);
        stream << ',' << prefix << "_linear_speed_m_s"
               << ',' << prefix << "_angular_velocity_rad_s"
               << ',' << prefix << "_angular_acceleration_rad_s2";
    }
    stream << '\n';

    const double meterScale = 1.0 / pixelsPerMeter;
    for (int row = 0; row < rowCount; ++row) {
        stream << csv.rows.at(row).frameIndex << ',' << times.at(row);
        for (int id : selectedIds) {
            const OptionalPoint &velocity = velocities[id].at(row);
            writeValue(stream, velocity.has_value()
                                   ? OptionalValue(std::hypot(velocity->x(),
                                                              velocity->y())
                                                   * meterScale)
                                   : std::nullopt);

            const OptionalPoint &acceleration = accelerations[id].at(row);
            writeValue(stream, acceleration.has_value()
                                   ? OptionalValue(std::hypot(acceleration->x(),
                                                              acceleration->y())
                                                   * meterScale)
                                   : std::nullopt);
        }

        for (const SegmentAnalysis &segment : segmentResults) {
            const OptionalPoint &velocity = segment.midpointVelocity.at(row);
            writeValue(stream, velocity.has_value()
                                   ? OptionalValue(std::hypot(velocity->x(),
                                                              velocity->y())
                                                   * meterScale)
                                   : std::nullopt);
            writeValue(stream, segment.angularVelocity.at(row));
            writeValue(stream, segment.angularAcceleration.at(row));
        }
        stream << '\n';
    }

    if (!stream) {
        *error = QObject::tr("写入物理信息 CSV 时发生错误。");
        return false;
    }
    return true;
}

class PhysicalSelectionView final : public QGraphicsView
{
public:
    explicit PhysicalSelectionView(const QImage &image,
                                   QVector<Marker> markers,
                                   QWidget *parent = nullptr)
        : QGraphicsView(parent)
        , image_(QPixmap::fromImage(image))
        , markers_(std::move(markers))
    {
        setScene(&scene_);
        setRenderHints(QPainter::Antialiasing
                       | QPainter::SmoothPixmapTransform);
        setDragMode(QGraphicsView::NoDrag);
        redraw();
    }

    QList<int> selectedIds() const
    {
        QList<int> ids = selectedIds_.values();
        std::sort(ids.begin(), ids.end());
        return ids;
    }

    QVector<QPair<int, int>> segments() const
    {
        return segments_;
    }

    void startPixelCalibration()
    {
        pixelCalibrationMode_ = true;
        mouseDown_ = false;
        calibrationStart_.reset();
        calibrationCurrent_.reset();
        redraw();
    }

    std::function<void(double)> calibrationCompleted;
    std::function<void()> selectionChanged;

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
        if (event->button() != Qt::LeftButton || !insideImage(position)) {
            QGraphicsView::mousePressEvent(event);
            return;
        }

        mouseDown_ = true;
        if (pixelCalibrationMode_) {
            calibrationStart_ = position;
            calibrationCurrent_ = position;
        } else {
            pressedId_ = nearestMarker(position);
            pressedPosition_ = position;
        }
        event->accept();
    }

    void mouseMoveEvent(QMouseEvent *event) override
    {
        if (mouseDown_ && pixelCalibrationMode_
            && calibrationStart_.has_value()) {
            const QPointF position = mapToScene(event->position().toPoint());
            calibrationCurrent_ = position;
            redraw();
            event->accept();
            return;
        }
        QGraphicsView::mouseMoveEvent(event);
    }

    void mouseReleaseEvent(QMouseEvent *event) override
    {
        if (event->button() != Qt::LeftButton || !mouseDown_) {
            QGraphicsView::mouseReleaseEvent(event);
            return;
        }
        mouseDown_ = false;
        const QPointF position = mapToScene(event->position().toPoint());

        if (pixelCalibrationMode_) {
            pixelCalibrationMode_ = false;
            const double length = calibrationStart_.has_value()
                    && insideImage(position)
                ? QLineF(*calibrationStart_, position).length()
                : 0.0;
            calibrationStart_.reset();
            calibrationCurrent_.reset();
            redraw();
            if (length >= 2.0 && calibrationCompleted) {
                calibrationCompleted(length);
            }
            event->accept();
            return;
        }

        const int releasedId = nearestMarker(position);
        if (pressedId_ > 0 && releasedId > 0) {
            const double dragLength = QLineF(pressedPosition_, position).length();
            if (pressedId_ != releasedId && dragLength >= sceneTolerance() / 2.0) {
                bool removedSameDirection = false;
                for (int index = segments_.size() - 1; index >= 0; --index) {
                    const auto &segment = segments_.at(index);
                    if (segment.first == pressedId_
                        && segment.second == releasedId) {
                        removedSameDirection = true;
                        segments_.removeAt(index);
                    } else if (segment.first == releasedId
                               && segment.second == pressedId_) {
                        segments_.removeAt(index);
                    }
                }
                if (!removedSameDirection) {
                    segments_.append({pressedId_, releasedId});
                }
            } else if (pressedId_ == releasedId) {
                if (selectedIds_.contains(pressedId_)) {
                    selectedIds_.remove(pressedId_);
                } else {
                    selectedIds_.insert(pressedId_);
                }
            }
            redraw();
            if (selectionChanged) {
                selectionChanged();
            }
        }
        pressedId_ = -1;
        event->accept();
    }

private:
    bool insideImage(const QPointF &position) const
    {
        return QRectF(QPointF(0.0, 0.0), image_.size()).contains(position);
    }

    qreal sceneTolerance() const
    {
        return 22.0 / std::max<qreal>(std::abs(transform().m11()), 0.001);
    }

    int nearestMarker(const QPointF &position) const
    {
        int id = -1;
        qreal distance = std::numeric_limits<qreal>::max();
        for (const Marker &marker : markers_) {
            const qreal candidate = QLineF(position, marker.position).length();
            if (candidate < distance) {
                distance = candidate;
                id = marker.id;
            }
        }
        return distance <= sceneTolerance() ? id : -1;
    }

    std::optional<QPointF> markerPosition(int id) const
    {
        for (const Marker &marker : markers_) {
            if (marker.id == id) {
                return marker.position;
            }
        }
        return std::nullopt;
    }

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

        const QPen segmentPen(QColor(40, 210, 120), 4.0);
        for (int index = 0; index < segments_.size(); ++index) {
            const auto first = markerPosition(segments_.at(index).first);
            const auto second = markerPosition(segments_.at(index).second);
            if (!first.has_value() || !second.has_value()) {
                continue;
            }
            auto *line = scene_.addLine(QLineF(*first, *second), segmentPen);
            line->setZValue(1.0);
            auto *label = scene_.addSimpleText(QStringLiteral("L%1").arg(index + 1));
            label->setBrush(Qt::yellow);
            label->setPen(QPen(Qt::black, 1.0));
            label->setPos((*first + *second) / 2.0 + QPointF(5.0, 5.0));
            label->setZValue(3.0);
        }

        if (calibrationStart_.has_value()
            && calibrationCurrent_.has_value()) {
            QPen calibrationPen(QColor(30, 220, 255), 3.0, Qt::DashLine);
            auto *line = scene_.addLine(
                QLineF(*calibrationStart_, *calibrationCurrent_),
                calibrationPen);
            line->setZValue(2.0);
        }

        const QPen outline(Qt::white, 2.0);
        for (const Marker &markerData : markers_) {
            const QColor color = selectedIds_.contains(markerData.id)
                ? QColor(235, 70, 70, 230)
                : QColor(60, 130, 240, 210);
            auto *marker = scene_.addEllipse(
                QRectF(markerData.position.x() - 6.0,
                       markerData.position.y() - 6.0, 12.0, 12.0),
                outline, QBrush(color));
            marker->setZValue(4.0);
            auto *number = scene_.addSimpleText(
                QStringLiteral("ID %1").arg(markerData.id));
            number->setBrush(Qt::yellow);
            number->setPen(QPen(Qt::black, 1.0));
            number->setPos(markerData.position + QPointF(8.0, -20.0));
            number->setZValue(5.0);
        }
    }

    QGraphicsScene scene_;
    QPixmap image_;
    QVector<Marker> markers_;
    QSet<int> selectedIds_;
    QVector<QPair<int, int>> segments_;
    bool pixelCalibrationMode_ = false;
    bool mouseDown_ = false;
    int pressedId_ = -1;
    QPointF pressedPosition_;
    std::optional<QPointF> calibrationStart_;
    std::optional<QPointF> calibrationCurrent_;
};

class PhysicalAnalysisDialog final : public QDialog
{
public:
    PhysicalAnalysisDialog(QString csvPath, TrackingCsv csv,
                           const QImage &firstFrame, double sourceFps,
                           int videoFrameCount, QWidget *parent)
        : QDialog(parent)
        , csvPath_(std::move(csvPath))
        , csv_(std::move(csv))
        , sourceFps_(sourceFps)
    {
        setWindowTitle(tr("标定并生成物理信息"));
        resize(1050, 780);

        const CsvRow &firstRow = csv_.rows.constFirst();
        QVector<Marker> markers;
        for (int point = 0; point < csv_.pointIds.size(); ++point) {
            if (firstRow.points.at(point).has_value()) {
                markers.append({csv_.pointIds.at(point),
                                *firstRow.points.at(point)});
            }
        }
        view_ = new PhysicalSelectionView(firstFrame, markers, this);

        auto *videoInformation = new QLabel(
            tr("原视频总帧数：%1　原视频帧率：%2 FPS")
                .arg(videoFrameCount)
                .arg(sourceFps_, 0, 'f', 6),
            this);
        auto *instructions = new QLabel(
            tr("单击目标点可选择或取消；从目标点 A 按住拖到目标点 B 后松开可建立线段。"
               "再次按相同方向拖过同一线段可删除。短缺失（连续不超过 3 帧）"
               "会先进行时间线性插值。"),
            this);
        instructions->setWordWrap(true);

        pixelScaleEdit_ = new QLineEdit(this);
        pixelScaleEdit_->setPlaceholderText(tr("必须填写，例如 1250"));
        timeScaleEdit_ = new QLineEdit(
            QString::number(sourceFps_, 'g', 12), this);
        auto *scaleValidator = new QDoubleValidator(
            0.000000001, 1000000000000.0, 9, this);
        scaleValidator->setNotation(QDoubleValidator::StandardNotation);
        pixelScaleEdit_->setValidator(scaleValidator);
        auto *timeValidator = new QDoubleValidator(
            0.000000001, 1000000000000.0, 9, this);
        timeValidator->setNotation(QDoubleValidator::StandardNotation);
        timeScaleEdit_->setValidator(timeValidator);

        auto *calibrationButton = new QPushButton(tr("辅助像素标定"), this);
        selectionLabel_ = new QLabel(this);
        updateSelectionLabel();

        auto *form = new QGridLayout;
        form->addWidget(new QLabel(tr("像素标定（像素/米）："), this), 0, 0);
        form->addWidget(pixelScaleEdit_, 0, 1);
        form->addWidget(calibrationButton, 0, 2);
        form->addWidget(new QLabel(tr("时间标定（帧/秒）："), this), 1, 0);
        form->addWidget(timeScaleEdit_, 1, 1);
        form->addWidget(selectionLabel_, 1, 2);
        form->setColumnStretch(1, 1);

        auto *buttons = new QDialogButtonBox(this);
        auto *generateButton = buttons->addButton(
            tr("标定并生成 CSV"), QDialogButtonBox::AcceptRole);
        buttons->addButton(QDialogButtonBox::Cancel);

        auto *layout = new QVBoxLayout(this);
        layout->addWidget(videoInformation);
        layout->addWidget(instructions);
        layout->addLayout(form);
        layout->addWidget(view_, 1);
        layout->addWidget(buttons);

        connect(calibrationButton, &QPushButton::clicked, this, [this] {
            QMessageBox::information(
                this, tr("辅助像素标定"),
                tr("请在画面中任意位置按下鼠标左键，拖到已知距离的另一端后松开。"));
            view_->startPixelCalibration();
        });
        view_->calibrationCompleted = [this](double pixels) {
            acceptCalibrationLength(pixels);
        };
        view_->selectionChanged = [this] {
            updateSelectionLabel();
        };
        connect(generateButton, &QPushButton::clicked, this, [this] {
            generate();
        });
        connect(buttons, &QDialogButtonBox::rejected,
                this, &QDialog::reject);
    }

private:
    void updateSelectionLabel()
    {
        selectionLabel_->setText(
            tr("已选 %1 个点，%2 条线段")
                .arg(view_->selectedIds().size())
                .arg(view_->segments().size()));
    }

    void acceptCalibrationLength(double pixels)
    {
        QDialog dialog(this);
        dialog.setWindowTitle(tr("输入实际距离"));
        auto *pixelLabel = new QLabel(
            tr("所画线段的像素长度：%1 px").arg(pixels, 0, 'f', 3),
            &dialog);
        auto *distance = new QDoubleSpinBox(&dialog);
        distance->setDecimals(9);
        distance->setRange(0.000000001, 1000000000.0);
        distance->setValue(1.0);
        auto *unit = new QComboBox(&dialog);
        unit->addItem(tr("米"));
        unit->addItem(tr("毫米"));
        auto *entryLayout = new QHBoxLayout;
        entryLayout->addWidget(distance, 1);
        entryLayout->addWidget(unit);
        auto *buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
        auto *layout = new QVBoxLayout(&dialog);
        layout->addWidget(pixelLabel);
        layout->addLayout(entryLayout);
        layout->addWidget(buttons);
        connect(buttons, &QDialogButtonBox::accepted,
                &dialog, &QDialog::accept);
        connect(buttons, &QDialogButtonBox::rejected,
                &dialog, &QDialog::reject);
        if (dialog.exec() != QDialog::Accepted) {
            return;
        }

        const double meters = unit->currentIndex() == 0
            ? distance->value()
            : distance->value() / 1000.0;
        pixelScaleEdit_->setText(
            QString::number(pixels / meters, 'g', 12));
    }

    void generate()
    {
        bool pixelOk = false;
        bool timeOk = false;
        const double pixelsPerMeter = pixelScaleEdit_->text().toDouble(&pixelOk);
        const double framesPerSecond = timeScaleEdit_->text().toDouble(&timeOk);
        if (!pixelOk || !std::isfinite(pixelsPerMeter)
            || pixelsPerMeter <= 0.0) {
            QMessageBox::warning(this, tr("像素标定无效"),
                                 tr("请填写大于 0 的“一米对应像素数”，"
                                    "或使用辅助像素标定。"));
            return;
        }
        if (!timeOk || !std::isfinite(framesPerSecond)
            || framesPerSecond <= 0.0) {
            QMessageBox::warning(this, tr("时间标定无效"),
                                 tr("请填写大于 0 的“一秒对应帧数”。"));
            return;
        }

        const QList<int> selectedIds = view_->selectedIds();
        const QVector<QPair<int, int>> segments = view_->segments();
        if (selectedIds.isEmpty() && segments.isEmpty()) {
            QMessageBox::warning(this, tr("尚未选择目标"),
                                 tr("请至少选择一个目标点或建立一条线段。"));
            return;
        }

        const QFileInfo inputInfo(csvPath_);
        const QString outputPath = QDir(inputInfo.absolutePath()).filePath(
            inputInfo.completeBaseName() + QStringLiteral("_physical.csv"));
        if (QFileInfo::exists(outputPath)) {
            const auto answer = QMessageBox::question(
                this, tr("覆盖已有文件"),
                tr("文件已存在：\n%1\n\n是否覆盖？")
                    .arg(QDir::toNativeSeparators(outputPath)),
                QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
            if (answer != QMessageBox::Yes) {
                return;
            }
        }

        QApplication::setOverrideCursor(Qt::WaitCursor);
        QString error;
        bool ok = false;
        try {
            ok = generatePhysicalCsv(
                outputPath, csv_, selectedIds, segments, pixelsPerMeter,
                framesPerSecond, &error);
        } catch (const std::exception &exception) {
            error = tr("计算物理信息时发生错误：%1")
                        .arg(QString::fromUtf8(exception.what()));
        } catch (...) {
            error = tr("计算物理信息时发生未知错误。");
        }
        QApplication::restoreOverrideCursor();
        if (!ok) {
            QMessageBox::critical(this, tr("生成失败"), error);
            return;
        }

        QMessageBox::information(
            this, tr("生成完成"),
            tr("物理信息 CSV 已生成：\n%1")
                .arg(QDir::toNativeSeparators(outputPath)));
        accept();
    }

    QString csvPath_;
    TrackingCsv csv_;
    double sourceFps_ = 0.0;
    PhysicalSelectionView *view_ = nullptr;
    QLineEdit *pixelScaleEdit_ = nullptr;
    QLineEdit *timeScaleEdit_ = nullptr;
    QLabel *selectionLabel_ = nullptr;
};

} // namespace

void openPhysicalAnalysisDialog(const QString &csvPath, QWidget *parent)
{
    TrackingCsv csv;
    QString error;
    if (!parseTrackingCsv(csvPath, &csv, &error)) {
        QMessageBox::warning(parent, QObject::tr("CSV 格式不符合要求"), error);
        return;
    }

    const QFileInfo videoInfo(csv.sourceVideoPath);
    if (!videoInfo.isFile()) {
        QMessageBox::warning(
            parent, QObject::tr("源视频不存在"),
            QObject::tr("CSV 记录的源视频文件不存在：\n%1")
                .arg(QDir::toNativeSeparators(csv.sourceVideoPath)));
        return;
    }

    cv::VideoCapture capture(openCvPath(csv.sourceVideoPath));
    if (!capture.isOpened()) {
        QMessageBox::warning(parent, QObject::tr("源视频无法打开"),
                             QObject::tr("OpenCV 无法打开 CSV 对应的源视频。"));
        return;
    }
    cv::Mat firstFrame;
    if (!capture.read(firstFrame) || firstFrame.empty()) {
        QMessageBox::warning(parent, QObject::tr("源视频无法读取"),
                             QObject::tr("无法读取源视频第一帧。"));
        return;
    }
    double fps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 0.0) {
        fps = 25.0;
    }
    int frameCount = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    if (frameCount <= 0) {
        frameCount = csv.rows.size();
    }
    capture.release();

    bool hasFirstFramePoint = false;
    for (const OptionalPoint &point : csv.rows.constFirst().points) {
        if (point.has_value()) {
            hasFirstFramePoint = true;
            break;
        }
    }
    if (!hasFirstFramePoint) {
        QMessageBox::warning(
            parent, QObject::tr("首帧没有目标点"),
            QObject::tr("CSV 第一帧的所有目标点坐标均为空，无法在首帧中选择目标。"));
        return;
    }

    PhysicalAnalysisDialog dialog(csvPath, std::move(csv),
                                  matToQImage(firstFrame), fps,
                                  frameCount, parent);
    dialog.exec();
}
