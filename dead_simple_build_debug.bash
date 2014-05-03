#!/usr/bin/env bash

set -x

cd /home/ehvatum/zplrepo/ris_widget && \
rm -fv sipris_*.cpp sipAPIris_widget.h _ris_widget.so ris_widget.sbf && \
sip -c . -b ris_widget.sbf -I /usr/share/sip/PyQt5 -x VendorID -t WS_X11 -t Qt_5_2_0 -g ris_widget.sip && \
g++ -Og -ggdb -fPIC -std=c++0x -march=native -Wall -W -D_REENTRANT -DQT_NO_DEBUG -DQT_WIDGETS_LIB -DQT_GUI_LIB -DQT_CORE_LIB -I/usr/lib64/qt5/mkspecs/linux-g++ -I. -I/usr/include/qt5 -I/usr/include/qt5/QtWidgets -I/usr/include/qt5/QtGui -I/usr/include/qt5/QtCore -I/usr/include/qt5/QtOpenGL -I/usr/local/cpython_debug_build/include/python3.4dm -I/home/ehvatum/cpython_debug_venv/lib/python3.4/site-packages/numpy/core/include -I/usr/local/boost_for_cpython_debug_build/include -I/usr/local/glm -Wl,-O1 -std=c++0x -shared -lQt5Widgets -lQt5Gui -lQt5Core -lQt5OpenGL -lGL -lpthread -lboost_python3 -L/usr/local/cpython_debug_build/lib -L/usr/local/boost_for_cpython_debug_build/lib -lboost_numpy *.cpp cpp/*.o -o _ris_widget.so
