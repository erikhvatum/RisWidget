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

from .shared_resources import GL_QSURFACE_FORMAT, GL
from .shader_scene import MouseoverTextItem
import numpy
from PyQt5 import Qt

class ShaderView(Qt.QGraphicsView):
    # scene_view_rect_changed is emitted immediately after the boundries of the scene region framed
    # by the view rect change (including the portions of the scene extending beyond the ImageItem,
    # in the case where the view is larger than the ImageItem given the current zoom level).  Item
    # view coordinates may be held constant by updating item scene position in response to this
    # signal (in the case of shader_scene.MouseoverTextItem, for example).
    scene_view_rect_changed = Qt.pyqtSignal()

    def __init__(self, shader_scene, parent):
        super().__init__(shader_scene, parent)
        self.view_items = []
        self.setMouseTracking(True)
        glw = _ShaderViewGLViewport(self)
        # It seems necessary to retain this reference.  It is available via self.viewport() after
        # the setViewport call completes, suggesting that PyQt keeps a reference to it, but this 
        # reference is evidentally weak or perhaps just a pointer.
        self._glw = glw
        self.setViewport(glw)
        self.add_mouseover_info_item()
        self.destroyed.connect(self._free_shader_view_resources)

    def add_mouseover_info_item(self):
        f = Qt.QFont('Courier', pointSize=14, weight=Qt.QFont.Bold)
        f.setKerning(False)
        f.setStyleHint(Qt.QFont.Monospace, Qt.QFont.OpenGLCompatible | Qt.QFont.PreferQuality)
        self.mouseover_text_item = MouseoverTextItem(self)
        self.mouseover_text_item.setAcceptHoverEvents(False)
        self.mouseover_text_item.setAcceptedMouseButtons(Qt.Qt.NoButton)
        scene = self.scene()
        scene.addItem(self.mouseover_text_item)
        self.view_items.append(self.mouseover_text_item)
        self.mouseover_text_item.setFont(f)
        c = Qt.QColor(45,255,70,255)
        self.mouseover_text_item.setDefaultTextColor(c)
        scene.update_mouseover_info_signal.connect(self.on_update_mouseover_info)
        self.scene_view_rect_changed.connect(self.mouseover_text_item.on_shader_view_scene_rect_changed)

    def on_update_mouseover_info(self, string, is_html):
        if is_html:
            self.mouseover_text_item.setHtml(string)
        else:
            self.mouseover_text_item.setPlainText(string)

    @property
    def mouseover_info_color(self):
        """(r,g,b,a) tuple, with elements in the range [0,255].  The alpha channel value (4th element of the 
        tuple) defaults to 255 and may be omitted when setting this property."""
        c = self.mouseover_text_item.defaultTextColor()
        return c.red(), c.green(), c.blue(), c.alpha()

    @mouseover_info_color.setter
    def mouseover_info_color(self, rgb_a):
        rgb_a = tuple(map(int, rgb_a))
        if len(rgb_a) == 3:
            rgb_a = rgb_a + (255,)
        elif len(rgb_a) != 4:
            raise ValueError('Value supplied for mouseover_info_color must be a 3 or 4 element iterable.')
        self.mouseover_text_item.setDefaultTextColor(Qt.QColor(*rgb_a))

    def _free_shader_view_resources(self):
        """Delete, release, or otherwise destroy GL resources associated with this ShaderView instance."""
        scene = self.scene()
        if scene is not None:
            for view_item in self.view_items:
                scene.removeItem(view_item)
            viewport = self.viewport()
            if viewport is not None and viewport.context() is not None and viewport.context().isValid():
                viewport.makeCurrent()
                try:
                    self.quad_vao.destroy()
                    self.quad_buffer.destroy()
                    for item in scene.items():
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

    def scrollContentsBy(self, dx, dy):
        """This function is never actually called for HistogramView as HistogramView always displays
        a unit-square view into HistogramScene.  However, if zooming and panning and whatnot are ever
        implemented for HistogramView, then this function will swing into action as it does for ImageView,
        and HistogramView's add_mouseover_info_item's resize signal's disconnect call should be removed."""
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
        self.scene_view_rect_changed.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        size = event.size()
        self.on_resize(size)
        self.on_resize_done(size)

    def on_image_changing(self, image):
        self.scene_view_rect_changed.emit()

    def on_resize(self, size):
        """Adjust view transform in response to view resize."""
        pass

    def on_resize_done(self, size):
        """Adjust scene contents in response to modification of view transform caused by view resize."""
        self.scene_view_rect_changed.emit()

    def drawBackground(self, p, rect):
        p.beginNativePainting()
        gl = GL()
        gl.glClearColor(0,0,0,1)
        gl.glClearDepth(1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
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

    _DONT_CALL_ME_ERROR = 'This method should not be called; any event or signal that '
    _DONT_CALL_ME_ERROR+= 'could potentially result in this method executing should have '
    _DONT_CALL_ME_ERROR+= 'been intercepted by the Viewport owning this _ShaderViewGLViewport '
    _DONT_CALL_ME_ERROR+= 'instance.'
