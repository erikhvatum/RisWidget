#!/usr/bin/env bash

set -x

cd /home/ehvatum/zplrepo/ris_widget && \
rm -fv sipris_*.cpp sipAPIris_widget.h ris_widget.so ris_widget.sbf && \
sip -c . -b ris_widget.sbf -I /usr/share/sip/PyQt5 -x VendorID -t WS_X11 -t Qt_5_2_0 -g ris_widget.sip && \
g++ -O2 -fPIC -std=c++0x -march=native -Wall -W -D_REENTRANT -DQT_NO_DEBUG -DQT_WIDGETS_LIB -DQT_GUI_LIB -DQT_CORE_LIB -I/usr/lib64/qt5/mkspecs/linux-g++ -I. -I/usr/include/qt5 -I/usr/include/qt5/QtWidgets -I/usr/include/qt5/QtGui -I/usr/include/qt5/QtCore -I/usr/include/qt5/QtOpenGL -I/usr/include/python3.3 -I/usr/lib64/python3.3/site-packages/numpy/core/include -I/usr/local/glm -Wl,-O1 -std=c++0x -shared -lQt5Widgets -lQt5Gui -lQt5Core -lQt5OpenGL -lGL -lpthread -lboost_python-3.3 -L/usr/local/lib -lboost_numpy *.cpp cpp/*.o -o ris_widget.so
