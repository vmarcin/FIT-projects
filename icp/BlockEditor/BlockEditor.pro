#-------------------------------------------------
#
# Project created by QtCreator 2018-04-24T00:10:34
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = BlockEditor
TEMPLATE = app
# CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        mainwindow.cpp \
    schemeitem.cpp \
    schemescene.cpp \
    link.cpp \
    block.cpp \
    getreal.cpp \
    getimaginary.cpp \
    arithmblock.cpp \
    polarform.cpp \
    port.cpp \
    scheme.cpp \
    adder.cpp \
    multiplier.cpp \
    subtractor.cpp \
    divider.cpp \
    connection.cpp \
    createcomplex.cpp \
    polartocomplex.cpp

HEADERS += \
        mainwindow.h \
    schemeitem.h \
    schemescene.h \
    link.h \
    block.h \
    port.h \
    polarform.h \
    getreal.h \
    getimaginary.h \
    scheme.h \
    connection.h \
    subtractor.h \
    multiplier.h \
    divider.h \
    adder.h \
    arithmblock.h \
    createcomplex.h \
    polartocomplex.h

RESOURCES += \
    schemescene.qrc
