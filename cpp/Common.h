// The MIT License (MIT)
//
// Copyright (c) 2014 Erik Hvatum
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef __RIS_WIDGET__COMMON_H_01936724387909345__
#define __RIS_WIDGET__COMMON_H_01936724387909345__

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#define GLM_FORCE_CXX11
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#pragma GCC diagnostic pop

#include <atomic>
#include <boost/numpy.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/import.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/module.hpp>
#include <boost/python/manage_new_object.hpp>
#include <cmath>
#include <cstdint>
// Need cstring include for memcpy(..).  Note: this is the C++ version of string.h; it has nothing to do with
// Microsoft's CString.
#include <cstring>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#ifdef _DEBUG
 #undef _DEBUG
 #include <Python.h>
 #define _DEBUG
#else
 #include <Python.h>
#endif
#include <QAbstractScrollArea>
#include <QComboBox>
#include <QDoubleValidator>
#include <QFile>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QMainWindow>
#include <QMessageBox>
#include <QMutex>
#include <QMutexLocker>
#ifdef ENABLE_GL_DEBUG_LOGGING
 #include <QOpenGLDebugLogger>
#endif
#define GL_GLEXT_PROTOTYPES
#include <QOpenGLFunctions_4_3_Core>
// Note: QPointers act as weak references to QObject derived class instances.  See here:
// http://qt-project.org/doc/qt-5/qpointer.html
#include <QPointer>
#include <QResizeEvent>
#include <QScreen>
#include <QScrollBar>
#include <QSharedPointer>
#include <QString>
#include <QSurface>
#include <QThread>
#include <QTimer>
#include <QToolBar>
#include <QVector>
#include <QWindow>
#include <string>
#include <utility>
#include <vector>

#include "GilScopedRelease.h"
#include "LockedRef.h"
#include "RisWidgetException.h"

#endif

