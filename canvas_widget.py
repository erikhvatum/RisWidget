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

import numpy
from pathlib import Path
from PyQt5 import Qt
import sys

class _CanvasGLWidget(Qt.QOpenGLWidget):
    """In order to obtain a QGraphicsView instance that renders into an OpenGL 2.1
    compatibility context (OS X does not support OpenGL 3.0+ compatibility profile,
    and Qt 5.4.1 QPainter support relies upon legacy calls, limiting us to 2.1 on
    OS X and everywhere else as a practicality), I know of two supported methods:

    * Call the static method Qt.QSurfaceFormat.setDefaultFormat(our_format).  All
    widgets created thereafter will render into OpenGL 2.1 compat contexts.  That can
    possibly be avoided, by attempting to stash and restore the value of
    Qt.QSurfaceFormat.defaultFormat(), but this should must be done with care.  On
    your particular platform, when does QGraphicsView read
    Qt.QSurfaceFormat.defaultFormat()?  When you show() the widget?  During the next
    iteration of the event loop after you show() the widget?  Also, you don't want
    to interfere with intervening instantiation of a 3rd party widget (eg a,
    matplotlib chart, which may want the software rasterizer).

    * Make a QGLWidget (old-school) or QOpenGLWidget (new school) and feed it to
    QGraphicsView's method setViewport.  _CanvasGLWidget is that QOpenGLWidget.
    _CanvasGLWidget doesn't really do anything except set up an OpenGL context.
    All relevant QEvents are filtered by the QGraphicsView-derived instance.
    _CanvasGLWidget has paintGL and resizeGL methods only because PyQt5 verifies
    that these exist in order to enforce the no-class-instances-with-pure-virtual-methods
    C++ contract."""

    _QSURFACE_FORMAT = None

    def __init__(self, parent=None):
        super().__init__(parent)
        if _CanvasGLWidget._QSURFACE_FORMAT is None:
            qsurface_format = Qt.QSurfaceFormat()
            qsurface_format.setRenderableType(Qt.QSurfaceFormat.OpenGL)
            qsurface_format.setVersion(2, 1)
            qsurface_format.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
            qsurface_format.setSwapBehavior(Qt.QSurfaceFormat.DoubleBuffer)
            qsurface_format.setStereo(False)
            qsurface_format.setSwapInterval(1)
            _CanvasGLWidget._QSURFACE_FORMAT = qsurface_format
        self.setFormat(_CanvasGLWidget._QSURFACE_FORMAT)

    def initializeGL(self):
        # PyQt5 provides access to OpenGL functions up to OpenGL 2.0, but we have made a 2.1
        # context.  QOpenGLContext.versionFunctions(..) will, by default, attempt to return
        # a wrapper around QOpenGLFunctions2_1, which will fail, as there is no
        # PyQt5._QOpenGLFunctions_2_1 implementation.  Therefore, we explicitly request 2.0
        # functions, and any 2.1 calls that we want to make can not occur through self.glfs.
        vp = Qt.QOpenGLVersionProfile()
        vp.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
        vp.setVersion(2, 0)
        self._glfs = self.context().versionFunctions(vp)
        if not self._glfs:
            raise RuntimeError('Failed to retrieve OpenGL function bundle.')
        if not self._glfs.initializeOpenGLFunctions():
            raise RuntimeError('Failed to initialize OpenGL function bundle.')

    def paintGL(self):
        pass

    def resizeGL(self, x, y):
        pass

class CanvasWidget(Qt.QGraphicsView):
    _NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE = {
        numpy.uint8  : Qt.QOpenGLTexture.UInt8,
        numpy.uint16 : Qt.QOpenGLTexture.UInt16,
        numpy.float32: Qt.QOpenGLTexture.Float32}
    _IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT = {
        'g'   : Qt.QOpenGLTexture.R32F,
        'ga'  : Qt.QOpenGLTexture.RG32F,
        'rgb' : Qt.QOpenGLTexture.RGB32F,
        'rgba': Qt.QOpenGLTexture.RGBA32F}
    _IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT = {
        'g'   : Qt.QOpenGLTexture.Red,
        'ga'  : Qt.QOpenGLTexture.RG,
        'rgb' : Qt.QOpenGLTexture.RGB,
        'rgba': Qt.QOpenGLTexture.RGBA}

    request_mouseover_info_status_text_change = Qt.pyqtSignal(object)

    def __init__(self, parent, qsurface_format):
        super().__init__(parent)
        self.setFormat(qsurface_format)
        self.setMouseTracking(True)
        self._glfs = None
        self._quad_vao = None
        self._quad_buffer = None

    def event(self, event):
        if event.type() == Qt.QEvent.Leave:
            self.request_mouseover_info_status_text_change.emit(None)
        return super().event(event)

    def _init_glfs(self):
        # PyQt5 provides access to OpenGL functions up to OpenGL 2.0, but we have made a 2.1
        # context.  QOpenGLContext.versionFunctions(..) will, by default, attempt to return
        # a wrapper around QOpenGLFunctions2_1, which will fail, as there is no
        # PyQt5._QOpenGLFunctions_2_1 implementation.  Therefore, we explicitly request 2.0
        # functions, and any 2.1 calls that we want to make can not occur through self.glfs.
        vp = Qt.QOpenGLVersionProfile()
        vp.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
        vp.setVersion(2, 0)
        self._glfs = self.context().versionFunctions(vp)
        if not self._glfs:
            raise RuntimeError('Failed to retrieve OpenGL function bundle.')
        if not self._glfs.initializeOpenGLFunctions():
            raise RuntimeError('Failed to initialize OpenGL function bundle.')

    def _build_shader_prog(self, desc, vert_fn, frag_fn):
        source_dpath = Path(__file__).parent / 'shaders'
        prog = Qt.QOpenGLShaderProgram(self)
        if not prog.addShaderFromSourceFile(Qt.QOpenGLShader.Vertex, str(source_dpath / vert_fn)):
            raise RuntimeError('Failed to compile vertex shader "{}" for {} {} shader program.'.format(vert_fn, type(self).__name__, desc))
        if not prog.addShaderFromSourceFile(Qt.QOpenGLShader.Fragment, str(source_dpath / frag_fn)):
            raise RuntimeError('Failed to compile fragment shader "{}" for {} {} shader program.'.format(frag_fn, type(self).__name__, desc))
        if not prog.link():
            raise RuntimeError('Failed to link {} {} shader program.'.format(type(self).__name__, desc))
        return prog

    def _make_quad_vao(self):
        self._quad_vao = Qt.QOpenGLVertexArrayObject()
        self._quad_vao.create()
        quad_vao_binder = Qt.QOpenGLVertexArrayObject.Binder(self._quad_vao)
        quad = numpy.array([1.1, -1.1,
                            -1.1, -1.1,
                            -1.1, 1.1,
                            1.1, 1.1], dtype=numpy.float32)
        self._quad_buffer = Qt.QOpenGLBuffer(Qt.QOpenGLBuffer.VertexBuffer)
        self._quad_buffer.create()
        self._quad_buffer.bind()
        self._quad_buffer.setUsagePattern(Qt.QOpenGLBuffer.StaticDraw)
        self._quad_buffer.allocate(quad.ctypes.data, quad.nbytes)
