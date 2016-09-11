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
import OpenGL
import OpenGL.GL.ARB.texture_float
from PyQt5 import Qt
from ..shared_resources import QGL, GL_LOGGER, GL_QSURFACE_FORMAT
from ..image import Image

class BaseView(Qt.QGraphicsView):
    """Instances of BaseView and its subclasses have a .viewport_rect_item attribute, which is an instance of
    ViewportRectItem, an invisible graphics item.  A BaseView (or subclass) instance resizes and repositions
    its .viewport_rect_item to exactly fill its viewport.  This can be seen by assigning True to
    .viewport_rect_item.is_visible.

    So, if you wish for a scene element to remain fixed in scale with respect to the viewport and fixed in
    position with respect to the top-left corner of the viewport, simply parent the item in question to
    .viewport_rect_item (.scene().contextual_info_item does this, for example).  To make item placement relative
    to a viewport anchor that varies with viewport size, such as the bottom-right corner, it must be
    repositioned in response to emission of the .viewport_rect_item.size_changed signal."""

    def __init__(self, base_scene, parent):
        super().__init__(base_scene, parent)
        self.setMouseTracking(True)
        self._background_color = (0.0, 0.0, 0.0)
        gl_widget = _ShaderViewGLViewport(self)
        # It seems necessary to retain this reference.  It is available via self.viewport() after
        # the setViewport call completes, suggesting that PyQt keeps a reference to it, but this 
        # reference is evidentally weak or perhaps just a pointer.
        self.gl_widget = gl_widget
        self.setViewport(gl_widget)
        if GL_QSURFACE_FORMAT().samples() > 0:
            self.setRenderHint(Qt.QPainter.Antialiasing)
        self._update_viewport_rect_item()

    def scrollContentsBy(self, dx, dy):
        """This function is never actually called for HistogramView as HistogramView always displays
        a unit-square view into HistogramScene.  However, if zooming and panning and whatnot are ever
        implemented for HistogramView, then this function will swing into action as it does for GeneralView,
        and HistogramView's add_contextual_info_item's resize signal's disconnect call should be removed."""
        super().scrollContentsBy(dx, dy)
        # In the case of scrollContentsBy(..) execution in response to view resize, self.resizeEvent(..)
        # has not yet had a chance to do its thing, meaning that self.transform() may differ from
        # the value obtained during painting.  However, self.on_resize_done(..) also calls
        # _update_viewport_rect_item, at which point self.transform() does return the correct value.
        # Both happen during the same event loop iteration, and no repaint will occur until the next
        # iteration, so any incorrect position possibly set in response to _update_viewport_rect_item
        # here will be corrected in response to resizeEvent(..)'s scene_view_rect_changed emission
        # before the next repaint.  Thus, nothing repositioned in response to our call should be
        # visible to the user in an incorrect position.
        self._update_viewport_rect_item()

    def _on_resize(self, size):
        """_on_resize is called after self.size has been updated and before ._update_viewport_rect_item is
        called, providing an opportunity for subclasses to modify view transform in response to view resize
        without causing incorrect positioning of view-relative items."""
        pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._on_resize(event.size())
        self._update_viewport_rect_item()

    def _update_viewport_rect_item(self):
        try:
            i = self.viewport_rect_item
        except AttributeError:
            return
        i.size = self.size()
        p = self.mapToScene(0,0)
        if i.pos() != p:
            i.setPos(p)

    def drawBackground(self, p, rect):
        p.beginNativePainting()
        GL = QGL()
        GL.glClearColor(*self._background_color, 1.0)
        GL.glClearDepth(1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        p.endNativePainting()

    # def paintEvent(self, event):
    #     print('paintEvent')
    #     pass

    @property
    def viewport_rect_item(self):
        return self.scene().viewport_rect_item

    @property
    def background_color(self):
        """V.background_color = (R, G, B)
        where R, G, and B, are floating point values in the interval [0, 1]."""
        return self._background_color

    @background_color.setter
    def background_color(self, v):
        v = tuple(map(float, v))
        if len(v) != 3 or not all(map(lambda v_: 0 <= v_ <= 1, v)):
            raise ValueError('The iteraterable assigned to .background_color must represent 3 real numbers in the interval [0, 1].')
        self._background_color = v
        s = self.scene()
        if s:
            s.invalidate()

    def snapshot(self, scene_rect=None, size=None, msaa_sample_count=16):
        scene = self.scene()
        gl_widget = self.gl_widget
        if None in (gl_widget, scene):
            return
        if scene_rect is None:
            scene_rect = self.sceneRect()
        dpi_ratio = gl_widget.devicePixelRatio()
        if size is None:
            size = self.gl_widget.size()
        if scene_rect.isEmpty() or not scene_rect.isValid() or size.width() <= 0 or size.height() <= 0:
            return
        if dpi_ratio != 1:
            # This is an idiotic workaround, but work it does
            size = Qt.QSize(size.width() * dpi_ratio, size.height() * dpi_ratio)
        with ExitStack() as estack:
            if hasattr(self.scene(), 'contextual_info_item') and self.scene().contextual_info_item.isVisible():
                self.scene().contextual_info_item.hide()
                estack.callback(self.scene().contextual_info_item.show)
            gl_widget.makeCurrent()
            estack.callback(gl_widget.doneCurrent)
            GL = QGL()
            fbo_format = Qt.QOpenGLFramebufferObjectFormat()
            fbo_format.setInternalTextureFormat(GL.GL_RGBA8)
            fbo_format.setSamples(msaa_sample_count)
            fbo_format.setAttachment(Qt.QOpenGLFramebufferObject.CombinedDepthStencil)
            fbo = Qt.QOpenGLFramebufferObject(size, fbo_format)
            fbo.bind()
            estack.callback(fbo.release)
            GL.glClearColor(*self._background_color)
            GL.glClearDepth(1)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            glpd = Qt.QOpenGLPaintDevice(size)
            p = Qt.QPainter()
            p.begin(glpd)
            estack.callback(p.end)
            p.setRenderHints(Qt.QPainter.Antialiasing | Qt.QPainter.HighQualityAntialiasing)
            scene.render(p, Qt.QRectF(0,0,size.width(),size.height()), scene_rect)
            qimage = fbo.toImage()
        return Image.from_qimage(qimage).data

class _ShaderViewGLViewport(Qt.QOpenGLWidget):
    context_about_to_change = Qt.pyqtSignal(Qt.QOpenGLWidget)
    context_changed = Qt.pyqtSignal(Qt.QOpenGLWidget)

    def __init__(self, view):
        super().__init__()
        self.setFormat(GL_QSURFACE_FORMAT())
        self.view = view
        self.makeCurrent()
        OpenGL.GL.ARB.texture_float.glInitTextureFloatARB()

    def _check_current(self, estack):
        if Qt.QOpenGLContext.currentContext() is not self.context():
            self.makeCurrent()
            estack.callback(self.doneCurrent)

    def start_logging(self):
        if hasattr(self, 'logger'):
            return
        with ExitStack() as estack:
            self._check_current(estack)
            self.logger = GL_LOGGER()

    def stop_logging(self):
        if not hasattr(self, 'logger'):
            return
        with ExitStack() as estack:
            self._check_current(estack)
            self.logger.stopLogging()
            del self.logger

    def event(self, e):
        assert isinstance(e, Qt.QEvent)
        if e.type() == 215:
            # QEvent::WindowChangeInternal, an enum value equal to 215, is used internally by Qt and is not exposed by
            # PyQt5 (there is no Qt.QEvent.WindowChangeInternal, but simply comparing against the value it would have
            # works).  Upon receipt of a WindowChangeInternal event, QOpenGLWidget releases its C++ smart pointer 
            # reference to its context, causing the smart pointer's atomic reference counter to decrement.  If the count
            # has reached 0, the context is destroyed, and this is typically the case - but not always, and there is
            # no way to ensure that it will be in any particular instance (the atomic counter value could be incremented
            # by another thread in the interval between the query and actual smart pointer reset call).  So, QOpenGLWidget
            # can't know if it ought to make the context current before releasing the context's smart pointer, although
            # doing so would enable cleanup of GL resources.  Furthermore, QContext's destructor can not make itself
            # current - doing so requires a QSurface, and QContext has no knowledge of any QSurface instances.
            # 
            # So, to get around all this nonsense, we intercept the WindowChangeInternal event, make our context current,
            # emit the context_about_to_change signal to cause any cleanup that requires the old context to be current,
            # make no context current, and then, finally, we allow QOpenGLWidget to respond to the event.
            self.makeCurrent()
            had_logger = hasattr(self, 'logger')
            try:
                if had_logger:
                    self.stop_logging()
                self.context_about_to_change.emit(self)
            except Exception as e:
                Qt.qDebug('Exception of type {} in response to context_about_to_change signal: {}'.format(type(e), str(e)))
            self.doneCurrent()
            r = super().event(e)
            self.makeCurrent()
            if had_logger:
                self.start_logging()
            self.context_changed.emit(self)
            self.doneCurrent()
            return r
        return super().event(e)

    def paintGL(self):
        raise NotImplementedError(_ShaderViewGLViewport._DONT_CALL_ME_ERROR)

    def resizeGL(self, w, h):
        raise NotImplementedError(_ShaderViewGLViewport._DONT_CALL_ME_ERROR)

    _DONT_CALL_ME_ERROR =\
        'This method should not be called; any event or signal that ' \
        'could potentially result in this method executing should have ' \
        'been intercepted by the Viewport owning this _ShaderViewGLViewport ' \
        'instance.'
