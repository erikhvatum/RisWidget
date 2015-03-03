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

from contextlib import contextmanager
import numpy
from pathlib import Path
from PyQt5 import Qt
import sys

NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE = {
    numpy.uint8  : Qt.QOpenGLTexture.UInt8,
    numpy.uint16 : Qt.QOpenGLTexture.UInt16,
    numpy.float32: Qt.QOpenGLTexture.Float32}
IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT = {
    'g'   : Qt.QOpenGLTexture.R32F,
    'ga'  : Qt.QOpenGLTexture.RG32F,
    'rgb' : Qt.QOpenGLTexture.RGB32F,
    'rgba': Qt.QOpenGLTexture.RGBA32F}
IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT = {
    'g'   : Qt.QOpenGLTexture.Red,
    'ga'  : Qt.QOpenGLTexture.RG,
    'rgb' : Qt.QOpenGLTexture.RGB,
    'rgba': Qt.QOpenGLTexture.RGBA}

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
    gl_initializing = Qt.pyqtSignal()

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
            # Specifically enabling alpha channel is not sufficient for enabling QPainter composition modes that
            # use destination alpha (ie, nothing drawn in CompositionMode_DestinationOver will be visible in
            # a painGL widget).
#           qsurface_format.setRedBufferSize(8)
#           qsurface_format.setGreenBufferSize(8)
#           qsurface_format.setBlueBufferSize(8)
#           qsurface_format.setAlphaBufferSize(8)
            _CanvasGLWidget._QSURFACE_FORMAT = qsurface_format
        self.setFormat(_CanvasGLWidget._QSURFACE_FORMAT)
        self.view = None

    def initializeGL(self):
        # PyQt5 provides access to OpenGL functions up to OpenGL 2.0, but we have made a 2.1
        # context.  QOpenGLContext.versionFunctions(..) will, by default, attempt to return
        # a wrapper around QOpenGLFunctions2_1, which may fail, as there is not necessarily
        # a PyQt5._QOpenGLFunctions_2_1 implementation.  Therefore, we explicitly request 2.0
        # functions, and any 2.1 calls that we want to make can not occur through self.glfs.
        vp = Qt.QOpenGLVersionProfile()
        vp.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
        vp.setVersion(2, 0)
        self.glfs = self.context().versionFunctions(vp)
        if not self.glfs:
            raise RuntimeError('Failed to retrieve OpenGL function bundle.')
        if not self.glfs.initializeOpenGLFunctions():
            raise RuntimeError('Failed to initialize OpenGL function bundle.')
        self.gl_initializing.emit()

    def paintGL(self):
        print('paintGL(self)')

    def resizeGL(self, w, h):
        print('resizeGL(self, w, h)')

class CanvasView(Qt.QGraphicsView):
    def __init__(self, canvas_scene, parent):
        super().__init__(canvas_scene, parent)
#       self.setMouseTracking(True)
        glw = _CanvasGLWidget()
        glw.view = self
        glw.gl_initializing.connect(self._on_gl_initializing)
        self._glw = glw
        self.setViewport(glw)
        self.destroyed.connect(self._release_view_resources)

    def _release_view_resources(self):
        """Delete, release, or otherwise destroy GL resources associated with this CanvasView instance."""
        if self.scene() is not None and \
           self.viewport() is not None and \
           self.viewport().context() is not None and \
           self.viewport().context().isValid():
            with canvas_widget_gl_context(self):
                self.quad_vao.destroy()
                self.quad_buffer.destroy()
                for item in self.scene().items():
                    if issubclass(type(item), CanvasGLItem):
                        item.release_resources_for_view(self)

#   def event(self, event):
#       if event.type() == Qt.QEvent.Leave:
#           self.request_mouseover_info_status_text_change.emit(None)
#       return super().event(event)

    def _on_gl_initializing(self):
        self.glfs = self.viewport().glfs
        self._make_quad_vao()

    def _make_quad_vao(self):
        self.quad_vao = Qt.QOpenGLVertexArrayObject()
        self.quad_vao.create()
        quad_vao_binder = Qt.QOpenGLVertexArrayObject.Binder(self.quad_vao)
        quad = numpy.array([1.1, -1.1,
                            -1.1, -1.1,
                            -1.1, 1.1,
                            1.1, 1.1], dtype=numpy.float32)
        self.quad_buffer = Qt.QOpenGLBuffer(Qt.QOpenGLBuffer.VertexBuffer)
        self.quad_buffer.create()
        self.quad_buffer.bind()
        try:
            self.quad_buffer.setUsagePattern(Qt.QOpenGLBuffer.StaticDraw)
            self.quad_buffer.allocate(quad.ctypes.data, quad.nbytes)
        finally:
            # Note: the following release call is essential.  Without it, QPainter will never work for
            # this widget again!
            self.quad_buffer.release()

    def drawBackground(self, p, rect):
        p.beginNativePainting()
        gl = self.glfs
        gl.glClearColor(0,0,0,1)
        gl.glClearDepth(1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        p.endNativePainting()

@contextmanager
def canvas_widget_gl_context(canvas_widget):
    canvas_widget.viewport().makeCurrent()
    try:
        yield
    finally:
        canvas_widget.viewport().doneCurrent()

@contextmanager
def native_painting(qpainter):
    qpainter.beginNativePainting()
    try:
        yield
    finally:
        qpainter.endNativePainting()

@contextmanager
def bind(resource):
    resource.bind()
    try:
        yield
    finally:
        resource.release()

class CanvasScene(Qt.QGraphicsScene):
    request_mouseover_info_status_text_change = Qt.pyqtSignal(object)

class CanvasGLItem(Qt.QGraphicsItem):
    def __init__(self, parent_item=None):
        super().__init__(parent_item)
        self._view_resources = {}

    def build_shader_prog(self, desc, vert_fn, frag_fn, canvas_view):
        source_dpath = Path(__file__).parent / 'shaders'
        prog = Qt.QOpenGLShaderProgram(canvas_view)
        if not prog.addShaderFromSourceFile(Qt.QOpenGLShader.Vertex, str(source_dpath / vert_fn)):
            raise RuntimeError('Failed to compile vertex shader "{}" for {} {} shader program.'.format(vert_fn, type(self).__name__, desc))
        if not prog.addShaderFromSourceFile(Qt.QOpenGLShader.Fragment, str(source_dpath / frag_fn)):
            raise RuntimeError('Failed to compile fragment shader "{}" for {} {} shader program.'.format(frag_fn, type(self).__name__, desc))
        if not prog.link():
            raise RuntimeError('Failed to link {} {} shader program.'.format(type(self).__name__, desc))
        vrs = self._view_resources[canvas_view]
        if 'progs' not in vrs:
            vrs['progs'] = {desc : prog}
        else:
            vrs['progs'][desc] = prog

    def release_resources_for_view(self, canvas_view):
        for item in self.childItems():
            if issubclass(type(item), CanvasGLItem):
                item.release_resources_for_view(canvas_view)
        if canvas_view in self._view_resources:
            vrs = self._view_resources[canvas_view]
            if 'progs' in vrs:
                for prog in vrs['progs'].values():
                    prog.removeAllShaders()
            del self._view_resources[canvas_view]

    def _normalize_min_max(self, min_max):
        r = self._image.range
        min_max -= r[0]
        min_max /= r[1] - r[0]
