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

include(SIPMacros)

if(NOT AS_PYTHON_MODULE)
    set(_RIS_WIDGET_GUI_DEFS ${_RIS_WIDGET_GUI_DEFS} "-DSTAND_ALONE_EXECUTABLE")
endif()

if(ENABLE_GL_DEBUG_LOGGING)
    set(_RIS_WIDGET_GUI_DEFS ${_RIS_WIDGET_GUI_DEFS} "-DENABLE_GL_DEBUG_LOGGING")
endif()

if(ENABLE_CL_PROFILING)
    set(_RIS_WIDGET_GUI_DEFS ${_RIS_WIDGET_GUI_DEFS} "-DENABLE_CL_PROFILING")
endif()

qt5_wrap_ui(UI_HEADERS Flipper/Flipper.ui
                       HistogramWidget.ui
                       ImageWidget.ui
                       RisWidget.ui
                       ShowCheckerDialog.ui)

qt5_add_resources(UI_RESOURCES RisWidget.qrc)

set(SOURCES RisWidget.cpp
            Flipper/DiskFlipper.cpp
            Flipper/Flipper.cpp
            Flipper/RamFlipper.cpp
            GilStateScopeOperators.cpp
            GlslProg.cpp
            GlxFunctions.c
            HistoDrawProg.cpp
            HistogramWidget.cpp
            HistogramView.cpp
            ImageDrawProg.cpp
            ImageWidget.cpp
            ImageWidgetViewScroller.cpp
            ImageView.cpp
            Renderer.cpp
            RisWidgetException.cpp
            ShowCheckerDialog.cpp
            View.cpp
            ViewWidget.cpp
            ${UI_HEADERS}
            ${UI_RESOURCES})

set(LIBRARIES ${OPENGL_LIBRARIES}
              ${FREEIMAGEPLUS_LIBRARY}
              ${PYTHON_LIBRARY}
              ${OPENCL_LIBRARIES}
              Qt5::Core
              Qt5::Gui
              Qt5::OpenGL
              Qt5::Widgets)

if(AS_PYTHON_MODULE)
    add_library(ris_widget_gui STATIC ${SOURCES})
    set_target_properties(ris_widget_gui PROPERTIES COMPILE_OPTIONS "${_RIS_WIDGET_GUI_DEFS}")
    # Obviously the static lib, ris_widget_gui.a, as such is not going to be linked against any libraries.
    # target_link_libraries does more than set -llibs, however.  It also adds the #defines and other compiler
    # flags needed, including (vitally) -fPIC.  Going about this any other way causes CMake to officiously
    # add a -fPIE flag because this is a static lib and static libs are only ever linked to executables,
    # by CMake's logic.  But, THIS static lib is NOT being linked into an executable!  It is being linked
    # into a .so file, and that means we must have -fPIC.
    target_link_libraries(ris_widget_gui ${LIBRARIES})
    set(SIP_INCLUDES ${SIP_INCLUDES} ${PYQT5_SIP_DIR})
    set(SIP_EXTRA_OPTIONS ${SIP_EXTRA_OPTIONS} ${PYQT5_SIP_FLAGS} "-e")
    # NB: this creates a target named "python_module__ris_widget"
    add_sip_python_module(_ris_widget ris_widget.sip ris_widget_gui ${LIBRARIES})

    get_target_property(_PYTHON_MODULE_LOCATION python_module__ris_widget LOCATION)
    message(STATUS "***************************${_PYTHON_MODULE_LOCATION}****************************")
    if(WIN32)
        add_custom_command(TARGET python_module__ris_widget
                           POST_BUILD
                           COMMAND copy /Y ${_PYTHON_MODULE_LOCATION} "${PROJECT_SOURCE_DIR}/_ris_widget.pyd")
    else()
        add_custom_command(TARGET python_module__ris_widget
                           POST_BUILD
                           COMMAND cp -fv ${_PYTHON_MODULE_LOCATION} "${PROJECT_SOURCE_DIR}/_ris_widget.so")
    endif()
    unset(_PYTHON_MODULE_LOCATION)
else()
    add_executable(ris_widget ${SOURCES})
    set_target_properties(ris_widget PROPERTIES COMPILE_OPTIONS "${_RIS_WIDGET_GUI_DEFS}")
    target_link_libraries(ris_widget ${LIBRARIES})
endif()
