# The MIT License (MIT)
#
# Copyright (c) 2014-2015 WUSTL ZPLAB
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

from contextlib import ExitStack
import numpy
from PyQt5 import Qt
import sys

_NEXT_QGRAPHICSITEM_USERTYPE = Qt.QGraphicsItem.UserType + 1

def UNIQUE_QGRAPHICSITEM_TYPE():
    """Returns a value to return from QGraphicsItem.type() overrides (which help
    Qt and PyQt return objects of the right type from any call returning QGraphicsItem
    references; for details see http://www.riverbankcomputing.com/pipermail/pyqt/2015-January/035302.html
    and https://bugreports.qt.io/browse/QTBUG-45064)

    This function will not return the same value twice and should be
    used to generate type values for all custom item classes that may
    have instances in the same scene."""
    global _NEXT_QGRAPHICSITEM_USERTYPE
    ret = _NEXT_QGRAPHICSITEM_USERTYPE
    _NEXT_QGRAPHICSITEM_USERTYPE += 1
    return ret

_NEXT_QLISTWIDGETITEM_USERTYPE = Qt.QListWidgetItem.UserType + 1

def UNIQUE_QLISTWIDGETITEM_TYPE():
    global _NEXT_QLISTWIDGETITEM_USERTYPE
    ret = _NEXT_QLISTWIDGETITEM_USERTYPE
    _NEXT_QLISTWIDGETITEM_USERTYPE += 1
    return ret

_NV_PATH_RENDERING_AVAILABLE = None

def NV_PATH_RENDERING_AVAILABLE():
    global _NV_PATH_RENDERING_AVAILABLE
    if _NV_PATH_RENDERING_AVAILABLE is None:
        try:
            with ExitStack() as estack:
                glw = Qt.QOpenGLWidget()
                estack.callback(glw.deleteLater)
                glf = Qt.QSurfaceFormat()
                glf.setRenderableType(Qt.QSurfaceFormat.OpenGL)
                glf.setVersion(2, 1)
                glf.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
                glf.setSwapBehavior(Qt.QSurfaceFormat.DoubleBuffer)
                glf.setStereo(False)
                glf.setSwapInterval(1)
                glw.setFormat(glf)
                glw.show()
                estack.callback(glw.hide)
                if glw.context().hasExtension('GL_NV_path_rendering'):
                    print('Detected GL_NV_path_rendering support...')
                    _NV_PATH_RENDERING_AVAILABLE = True
                else:
                    print('No GL_NV_path_rendering support...')
                    _NV_PATH_RENDERING_AVAILABLE = False
        except:
            print('An error occurred while attempting to determine whether the GL_NV_path_rendering extension is supported.', sys.stderr)
            _NV_PATH_RENDERING_AVAILABLE = False
    return _NV_PATH_RENDERING_AVAILABLE

_GL_QSURFACE_FORMAT = None

def GL_QSURFACE_FORMAT(msaa_sample_count=None):
    global _GL_QSURFACE_FORMAT
    if _GL_QSURFACE_FORMAT is None:
        _GL_QSURFACE_FORMAT = Qt.QSurfaceFormat()
        _GL_QSURFACE_FORMAT.setRenderableType(Qt.QSurfaceFormat.OpenGL)
        _GL_QSURFACE_FORMAT.setVersion(2, 1)
        _GL_QSURFACE_FORMAT.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
        _GL_QSURFACE_FORMAT.setSwapBehavior(Qt.QSurfaceFormat.DoubleBuffer)
        _GL_QSURFACE_FORMAT.setStereo(False)
        _GL_QSURFACE_FORMAT.setSwapInterval(1)
        if msaa_sample_count is not None:
            _GL_QSURFACE_FORMAT.setSamples(msaa_sample_count)
        if NV_PATH_RENDERING_AVAILABLE():
            _GL_QSURFACE_FORMAT.setStencilBufferSize(4)
        _GL_QSURFACE_FORMAT.setRedBufferSize(8)
        _GL_QSURFACE_FORMAT.setGreenBufferSize(8)
        _GL_QSURFACE_FORMAT.setBlueBufferSize(8)
        _GL_QSURFACE_FORMAT.setAlphaBufferSize(8)
    return _GL_QSURFACE_FORMAT

_freeimage = None

def FREEIMAGE(show_messagebox_on_error=False, error_messagebox_owner=None):
    """If show_messagebox_on_error is true and importing freeimage fails with an exception, a modal QMessageBox is displayed
    describing the error and None is returned.  If show_messagebox_on_error is false and importing freeimage fails with
    and exception, the exception is allowed to propagate."""
    global _freeimage
    if _freeimage is None:
        if show_messagebox_on_error:
            try:
                import freeimage
                _freeimage = freeimage
            except ImportError:
                Qt.QMessageBox.information(error_messagebox_owner,
                                           'freeimage.py Not Found',
                                           "Zach's freeimage module is required for loading drag & dropped image files.")
                return
            except RuntimeError as e:
                estr = '\n'.join(("freeimage.py was found, but an error occurred while importing it " + \
                                  "(likely because freeimage.so/dylib/dll could not be found):\n",) + e.args)
                Qt.QMessageBox.information(error_messagebox_owner, 'Error While Importing freeimage Module', estr)
                return
        else:
            import freeimage
            _freeimage = freeimage
    return _freeimage
