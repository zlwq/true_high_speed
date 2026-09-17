#include "mainwindow.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication application(argc, argv);
    QApplication::setApplicationName(QStringLiteral("高速视频运动分析"));
    QApplication::setOrganizationName(QStringLiteral("true_high_speed"));

    MainWindow window;
    window.show();
    return QApplication::exec();
}
