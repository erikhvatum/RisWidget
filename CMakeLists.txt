# The MIT License (MIT)
#
# Copyright (c) 2014 Erik Hvatum
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# CMake 2.8.11 introduced the target_link_libraries command, a huge time saver
cmake_minimum_required(VERSION 2.8.11)

project(ris_widget)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_definitions(-DSTAND_ALONE_EXECUTABLE)

include_directories(SYSTEM /usr/local/include
                           /usr/local/glm
                           /usr/include/python3.4
                           /usr/lib64/python3.4/site-packages/numpy/core/include)

find_package(OpenGL REQUIRED)
find_package(Qt5Core)
find_package(Qt5Gui)
find_package(Qt5OpenGL)
find_package(Qt5Widgets)

add_executable(ris_widget source/cpp/RisWidget.cpp
                          source/cpp/GilStateScopeOperators.cpp
                          source/cpp/GlslProg.cpp
                          source/cpp/GlxFunctions.c
                          source/cpp/HistoDrawProg.cpp
                          source/cpp/HistogramWidget.cpp
                          source/cpp/HistogramView.cpp
                          source/cpp/ImageDrawProg.cpp
                          source/cpp/ImageWidget.cpp
                          source/cpp/ImageWidgetViewScroller.cpp
                          source/cpp/ImageView.cpp
                          source/cpp/Renderer.cpp
                          source/cpp/RisWidgetException.cpp
                          source/cpp/ShowCheckerDialog.cpp
                          source/cpp/View.cpp
                          source/cpp/ViewWidget.cpp
                          ${UI_HEADERS}
                          ${UI_RESOURCES})

qt5_wrap_ui(UI_HEADERS source/cpp/HistogramWidget.ui
                       source/cpp/RisWidget.ui
                       source/cpp/ShowCheckerDialog.ui
                       source/cpp/ImageWidget.ui)

qt5_add_resources(UI_RESOURCES source/cpp/RisWidget.qrc)

target_link_libraries(ris_widget Qt5::Core
                                 Qt5::Gui
                                 Qt5::OpenGL
                                 Qt5::Widgets)
