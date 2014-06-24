TEMPLATE = lib
LANGUAGE = C++
QT += core gui widgets
CONFIG += static c++11 exceptions rtti stl thread
CONFIG -= app_bundle
TARGET = bughunt
INCLUDEPATH += /Library/Frameworks/Python.framework/Versions/3.4/include/python3.4m /usr/local/boost/include/boost-1_55
CFLAGS += -fPIC -fno-omit-frame-pointer -march=native

SOURCES += bughunt.cpp

HEADERS += bughunt.h
