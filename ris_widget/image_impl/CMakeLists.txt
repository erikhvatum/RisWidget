# The MIT License (MIT)
#
# Copyright (c) 2016 WUSTL ZPLAB
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

include(SIPMacros)

set(SOURCES
        _CppImage.cpp
        _CppImage.h
        Gil.h
        Gil.cpp)

set(SIPSOURCE _cpp_image.sip)

set(LIBRARIES
        ${OPENGL_LIBRARIES}
        ${FREEIMAGE_LIBRARY}
#        ${FREEIMAGEPLUS_LIBRARY}
        ${PYTHON_LIBRARY}
        Qt5::Core
        Qt5::Gui
        Qt5::OpenGL
        Qt5::Widgets)

add_library(_cpp_image_impl STATIC ${SOURCES})
#set_target_properties(_cpp_image_impl PROPERTIES COMPILE_OPTIONS "${_cpp_image_impl_defs}")

# Obviously the static lib, ris_widget_gui.a, as such is not going to be linked against any libraries.
# target_link_libraries does more than set -llibs, however.  It also adds the #defines and other compiler
# flags needed, including (vitally) -fPIC.  Going about this any other way causes CMake to officiously
# add a -fPIE flag because this is a static lib and static libs are only ever linked to executables,
# by CMake's logic.  But, THIS static lib is NOT being linked into an executable!  It is being linked
# into a .so file, and that means we must have -fPIC.
target_link_libraries(_cpp_image_impl ${LIBRARIES})
#set(SIP_INCLUDES ${SIP_INCLUDES} ${PYQT5_SIP_DIR})
#set(SIP_EXTRA_OPTIONS ${SIP_EXTRA_OPTIONS} ${PYQT5_SIP_FLAGS} "-e")

add_sip_python_module(_cpp_image ${SIPSOURCE} _cpp_image_impl ${LIBRARIES})

# TODO: Remove this cheap hack, which is, at the moment, very helpful
# for running directly out of a build directory.
cmake_policy(SET CMP0026 OLD)

get_target_property(_PYTHON_MODULE_LOCATION python_module__cpp_image LOCATION)
if(WIN32)
    string(REPLACE "/" "\\" _PYTHON_MODULE_LOCATION ${_PYTHON_MODULE_LOCATION})
    string(REPLACE "/" "\\" _PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR})
    add_custom_command(TARGET python_module__cpp_image
            POST_BUILD
            COMMAND copy /Y ${_PYTHON_MODULE_LOCATION} "${_PROJECT_SOURCE_DIR}\\ris_widget\\image_impl\\_cpp_image.pyd")
    unset(_PROJECT_SOURCE_DIR)
else()
    add_custom_command(TARGET python_module__cpp_image
            POST_BUILD
            COMMAND cp -fv ${_PYTHON_MODULE_LOCATION} "${PROJECT_SOURCE_DIR}/ris_widget/image_impl/_cpp_image.so")
endif()
unset(_PYTHON_MODULE_LOCATION)