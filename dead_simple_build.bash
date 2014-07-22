#!/usr/bin/env bash

unamestr=`uname`

if [[ "$unamestr" == 'Darwin' ]]; then

    set -x

    cd /Users/ehvatum/zplrepo/ris_widget && \
    rm -fv sip_ris_*.cpp sipAPI_ris_widget.h _ris_widget.so ris_widget.sbf && \
    sip -c . -b ris_widget.sbf -I /Library/Frameworks/Python.framework/Versions/3.4/share/sip/PyQt5 -x VendorID -t WS_MACX -t Qt_5_3_1 -g ris_widget.sip && \
    clang++ -v -O0 -g -fno-omit-frame-pointer -fPIC -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk -std=c++11 -stdlib=libc++ -mmacosx-version-min=10.7 -Wall -W -DQT_NO_DEBUG -DQT_OPENGL_LIB -DQT_WIDGETS_LIB -DQT_GUI_LIB -DQT_CORE_LIB -I/usr/local/Cellar/qt5/5.3.1/mkspecs/macx-clang -I. -I/Library/Frameworks/Python.framework/Versions/3.4/include/python3.4m -I/usr/local/include -I/Library/Frameworks/Python.framework/Versions/3.4/lib/python3.4/site-packages/numpy/core/include/numpy -I/usr/local/qt5/5.3/clang_64/lib/QtOpenGL.framework/Versions/5/Headers -I/usr/local/qt5/5.3/clang_64/lib/QtWidgets.framework/Versions/5/Headers -I/usr/local/qt5/5.3/clang_64/lib/QtGui.framework/Versions/5/Headers -I/usr/local/qt5/5.3/clang_64/lib/QtCore.framework/Versions/5/Headers -I. -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk/System/Library/Frameworks/OpenGL.framework/Versions/A/Headers -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk/System/Library/Frameworks/AGL.framework/Headers -I. -F/usr/local/qt5/5.3/clang_64/lib -std=c++0x -shared -framework QtCore -framework QtGui -framework QtWidgets -framework OpenCL -framework OpenGL -framework CoreFoundation -lpthread *.cpp cpp/Debug/libRisWidget.a -o _ris_widget.so -lpython3.4m -L/Library/Frameworks/Python.framework/Versions/3.4/lib/python3.4/config-3.4m

else

    set -x

    cd /home/ehvatum/zplrepo/ris_widget && \
    rm -fv sip_ris_*.cpp sipAPI_ris_widget.h _ris_widget.so ris_widget.sbf && \
    sip -c . -b ris_widget.sbf -I /usr/share/sip/PyQt5 -x VendorID -t WS_X11 -t Qt_5_3_0 -g ris_widget.sip && \
    g++ -O2 -fno-omit-frame-pointer -fPIC -std=c++0x -march=native -Wall -W -D_REENTRANT -DQT_NO_DEBUG -DQT_WIDGETS_LIB -DQT_GUI_LIB -DQT_CORE_LIB -I/usr/lib64/qt5/mkspecs/linux-g++ -I. -I/usr/include/qt5 -I/usr/include/qt5/QtWidgets -I/usr/include/qt5/QtGui -I/usr/include/qt5/QtCore -I/usr/include/qt5/QtOpenGL -I/usr/include/python3.4 -I/usr/lib64/python3.4/site-packages/numpy/core/include -I/usr/local/glm -Wl,-O1 -std=c++0x -shared -lQt5Widgets -lQt5Gui -lQt5Core -lQt5OpenGL -lGL -lpthread *.cpp cpp/*.o -o _ris_widget.so

fi
