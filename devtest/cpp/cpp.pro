TEMPLATE = app
CONFIG -= app_bundle
QT += core gui widgets opengl
TARGET = cpp
INCLUDEPATH += .

# Input
HEADERS += ImageWidget.h GV.h GS.h
SOURCES += ImageWidget.cpp GV.cpp GS.cpp main.cpp
