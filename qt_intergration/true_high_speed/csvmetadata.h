#ifndef CSVMETADATA_H
#define CSVMETADATA_H

#include <QByteArray>
#include <QFileInfo>
#include <QString>

#include <fstream>

inline void writeCsvSourceVideoPath(std::ofstream &stream,
                                    const QString &videoPath)
{
    const QByteArray path = QFileInfo(videoPath).absoluteFilePath().toUtf8();
    stream << "\xEF\xBB\xBF#source_video_path=";
    stream.write(path.constData(), path.size());
    stream << '\n';
}

inline void writeCsvProcessedVideoPath(std::ofstream &stream,
                                       const QString &videoPath)
{
    const QByteArray path = QFileInfo(videoPath).absoluteFilePath().toUtf8();
    stream << "#processed_video_path=";
    stream.write(path.constData(), path.size());
    stream << '\n';
}

#endif // CSVMETADATA_H
