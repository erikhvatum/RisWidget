# The MIT License (MIT)
#
# Copyright (c) 2014 WUSTL ZPLAB
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
#
# Authors: Erik Hvatum <ice.rikh@gmail.com>

# CMake 2.8.11 introduced the target_link_libraries command, a huge time saver
cmake_minimum_required(VERSION 2.8.11)

project(ris_widget)

set(AS_PYTHON_MODULE TRUE CACHE BOOL "Build as a native Python extension (for example, a .so file on Linux) using PyQt5 and SIP.  Otherwise, ris_widget is built as a stand-alone executable.")
set(ENABLE_GL_DEBUG_LOGGING FALSE CACHE BOOL "Enable OpenGL debug output.")
set(ENABLE_CL_PROFILING FALSE CACHE BOOL "Enable OpenCL profiling and output of kernel execution times.")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules")
set(Python_ADDITIONAL_VERSIONS ${Python_ADDITIONAL_VERSIONS} "3.4")
include_directories(SYSTEM /usr/local/include)
include_directories(source/cpp)

find_package(OpenGL REQUIRED)
find_package(OpenCL REQUIRED)
find_package(FreeImage REQUIRED)
find_package(FreeImagePlus REQUIRED)
find_package(GLM REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(PythonInterp 3.4 REQUIRED)
find_package(PythonLibs 3.4 REQUIRED)
if(AS_PYTHON_MODULE)
    find_package(SIP REQUIRED)
    find_package(PyQt5 REQUIRED)
endif()

include_directories(SYSTEM ${GLM_INCLUDE_DIR})
include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})
include_directories(SYSTEM ${FREEIMAGEPLUS_INCLUDE_PATH})
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy;print(numpy.get_include())"
                RESULT_VARIABLE _NUMPY_INCLUDE_DIR_FOUND
                ERROR_VARIABLE _NUMPY_INCLUDE_DIR_ERROR
                OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _NUMPY_INCLUDE_DIR_FOUND MATCHES 0)
    message(FATAL_ERROR "Failed to import numpy (which must be done in order to locate the numpy C API header directory):\n${_NUMPY_INCLUDE_DIR_ERROR}")
    return()
endif()
include_directories(SYSTEM ${NUMPY_INCLUDE_DIR})

add_subdirectory(source/cpp output/cpp)
