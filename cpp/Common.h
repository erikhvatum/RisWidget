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

#ifdef __cplusplus
#ifndef __RIS_WIDGET__COMMON_H_01936724387909345__
#define __RIS_WIDGET__COMMON_H_01936724387909345__

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
//#define GLM_FORCE_CXX11
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#pragma GCC diagnostic pop

#include <atomic>
#include <chrono>
#include <cmath>
#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
 #include <OpenCL/opencl.h>
#else
 #include <CL/cl.h>
#endif
#undef CL_VERSION_1_2
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include "cl.hpp"
#include <cstdint>
// Need cstring include for memcpy(..).  Note: this is the C++ version of string.h; it has nothing to do with
// Microsoft's CString.
#include <cstring>
#include <FreeImagePlus.h>
#include <iostream>
#include <fstream>
#include <future>
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
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Some terrible environments #define min and max (Windows, for example).  How is std::numeric_limits<...>::max going to
// work if max expands to (( ( ( a )  >  ( b ) ) ? ( a ) : ( b ) )) ?!  BY NOT WORKING!  THAT'S HOW!  Anyway, Qt uses
// std::numeric_limits::min/max in its headers, so these defines have to go away before we include Qt.
#ifdef min
 #undef min
#endif
#ifdef max
 #undef max
#endif

#ifdef ENABLE_GL_DEBUG_LOGGING
 #include <QOpenGLDebugLogger>
#endif
#define GL_GLEXT_PROTOTYPES
#include <QOpenGLFunctions_4_1_Core>
// Note: QPointers act as weak references to QObject derived class instances.  See here:
// http://qt-project.org/doc/qt-5/qpointer.html
#include <QPointer>
#include <QtCore>
#include <QtWidgets>

#include "GilStateScopeOperators.h"
#include "LockedRef.h"
#include "RisWidgetException.h"

#endif
#endif
