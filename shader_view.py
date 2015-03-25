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

from .gl_resources import GL_QSURFACE_FORMAT, GL
#from .shader_view_overlay_scene import ShaderViewOverlayView
import numpy
from PyQt5 import Qt

class ShaderView(Qt.QGraphicsView):
    resized = Qt.pyqtSignal(Qt.QSize)

    def __init__(self, shader_scene, overlay_scene, parent):
        super().__init__(shader_scene, parent)
        self.overlay_scene = overlay_scene
        self.setMouseTracking(True)
        glw = _ShaderViewGLViewport(self)
        # It seems necessary to retain this reference.  It is available via self.viewport() after
        # the setViewport call completes, suggesting that PyQt keeps a reference to it, but this 
        # reference is evidentally weak or perhaps just a pointer.
        self._glw = glw
        self.setViewport(glw)
        self.destroyed.connect(self._free_shader_view_resources)
        self.overlay_scene.changed.connect(self.scene().invalidate)
#       self.overlay_scene.changed.connect(self._on_invalidate)
#       self.overlay_view = ShaderViewOverlayView(overlay_scene, glw)
#       self.overlay_view.show()

#   def _on_invalidate(self):
#       print('ShaderView._on_invalidate')
#       self.scene().invalidate()

    def _free_shader_view_resources(self):
        """Delete, release, or otherwise destroy GL resources associated with this ShaderView instance."""
        if self.scene() is not None:
            viewport = self.viewport()
            if viewport is not None and viewport.context() is not None and viewport.context().isValid():
                viewport.makeCurrent()
                try:
                    self.quad_vao.destroy()
                    self.quad_buffer.destroy()
                    for item in self.scene().items():
                        if issubclass(type(item), CanvasGLItem):
                            item.free_shader_view_resources(self)
                finally:
                    viewport.doneCurrent()

    def _on_gl_initializing(self):
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        size = event.size()
        self.overlay_scene.setSceneRect(0, 0, size.width(), size.height())
        self.resized.emit(size)

    def drawBackground(self, p, rect):
        p.beginNativePainting()
        gl = GL()
        gl.glClearColor(0,0,0,1)
        gl.glClearDepth(1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        p.endNativePainting()

    def drawForeground(self, p, rect):
        p.save()
        p.resetTransform()
        try:
            self.overlay_scene.render(p)
        finally:
            p.restore()

class _ShaderViewGLViewport(Qt.QOpenGLWidget):
    """In order to obtain a QGraphicsView instance that renders into an OpenGL 2.1
    compatibility context (OS X does not support OpenGL 3.0+ compatibility profile,
    and Qt 5.4.1 QPainter support relies upon legacy calls, limiting us to 2.1 on
    OS X - and everywhere else as a practicality), I know of two supported methods:

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
    QGraphicsView's setViewport method.  _ShaderViewGLViewport is that QOpenGLWidget.
    _ShaderViewGLViewport doesn't really do anything except set up an OpenGL context.
    All relevant QEvents are filtered by the QGraphicsView-derived instance.
    _ShaderViewGLViewport has paintGL and resizeGL methods only because older PyQt5
    versions verify that these exist in order to enforce the
    no-class-instances-with-pure-virtual-methods C++ contract."""

    def __init__(self, view):
        super().__init__()
        self.setFormat(GL_QSURFACE_FORMAT())
        self.view = view

    def initializeGL(self):
        self.view._on_gl_initializing()

    def paintGL(self):
        raise NotImplementedError(_ShaderViewGLViewport._DONT_CALL_ME_ERROR)

    def resizeGL(self, w, h):
        raise NotImplementedError(_ShaderViewGLViewport._DONT_CALL_ME_ERROR)

    _DONT_CALL_ME_ERROR = 'This method should not be called; any event or signal that '
    _DONT_CALL_ME_ERROR+= 'could potentially result in this method executing should have '
    _DONT_CALL_ME_ERROR+= 'been intercepted by the Viewport owning this _ShaderViewGLViewport '
    _DONT_CALL_ME_ERROR+= 'instance.'
