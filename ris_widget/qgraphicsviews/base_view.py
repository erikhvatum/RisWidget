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

import numpy
from PyQt5 import Qt
from ..shared_resources import QGL, GL_QSURFACE_FORMAT

class BaseView(Qt.QGraphicsView):
    """Updates to things depending directly on the view's size (eg, in many cases, the view's own transformation), if any,
    are initiated by the BaseView subclass's resizeEvent handler.

    Updates to things depending directly on the view's transformation and/or the region of the scene visible in the view
    (eg, the position of the scene's context_info_item relative to the top left of its view) occur in response to the
    scene_region_changed signal."""
    scene_region_changed = Qt.pyqtSignal(Qt.QGraphicsView)

    def __init__(self, base_scene, parent):
        super().__init__(base_scene, parent)
        self.setMouseTracking(True)
        gl_widget = _ShaderViewGLViewport(self)
        # It seems necessary to retain this reference.  It is available via self.viewport() after
        # the setViewport call completes, suggesting that PyQt keeps a reference to it, but this 
        # reference is evidentally weak or perhaps just a pointer.
        self.gl_widget = gl_widget
        self.setViewport(gl_widget)
        if GL_QSURFACE_FORMAT().samples() > 0:
            self.setRenderHint(Qt.QPainter.Antialiasing)
        self.scene_region_changed.connect(base_scene.contextual_info_item.return_to_fixed_position)
        self.scene_region_changed.emit(self)

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

    def scrollContentsBy(self, dx, dy):
        """This function is never actually called for HistogramView as HistogramView always displays
        a unit-square view into HistogramScene.  However, if zooming and panning and whatnot are ever
        implemented for HistogramView, then this function will swing into action as it does for GeneralView,
        and HistogramView's add_contextual_info_item's resize signal's disconnect call should be removed."""
        super().scrollContentsBy(dx, dy)
        # In the case of scrollContentsBy(..) execution in response to view resize, self.resizeEvent(..)
        # has not yet had a chance to do its thing, meaning that self.transform() may differ from
        # the value obtained during painting.  However, self.on_resize_done(..) also emits
        # scene_view_rect_changed, at which point self.transform() does return the correct value.
        # Both happen during the same event loop iteration, and no repaint will occur until the next
        # iteration, so any incorrect position possibly set in response to scene_view_rect_change emission
        # here will be corrected in response to resizeEvent(..)'s scene_view_rect_changed emission
        # before the next repaint.  Thus, nothing repositioned in response to our emission should be
        # visible to the user in an incorrect position.
        self.scene_region_changed.emit(self)

    def _on_resize(self, size):
        """_on_resize is called after self.size has been updated and before scene_region_changed is emitted,
        providing an opportunity for subclasses to modify view transform in response to view resize without
        causing incorrect positioning of view-relative items."""
        pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._on_resize(event.size())
        self.scene_region_changed.emit(self)

    def drawBackground(self, p, rect):
        p.beginNativePainting()
        GL = QGL()
        GL.glClearColor(0,0,0,1)
        GL.glClearDepth(1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        p.endNativePainting()

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
    _ShaderViewGLViewport has paintGL and resizeGL methods only so that we can
    detect their being called."""

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

    _DONT_CALL_ME_ERROR =\
        'This method should not be called; any event or signal that ' \
        'could potentially result in this method executing should have ' \
        'been intercepted by the Viewport owning this _ShaderViewGLViewport ' \
        'instance.'
