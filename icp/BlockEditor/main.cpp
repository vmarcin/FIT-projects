/**
* @file main.cpp
* @brief Main
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    MainWindow w;
    w.setWindowState(Qt::WindowFullScreen);
    //w.setWindowState(Qt::WindowMaximized);
    w.show();

    return a.exec();
}
