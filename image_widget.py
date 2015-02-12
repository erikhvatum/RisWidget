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

from .canvas_widget import CanvasWidget
import math
import numpy
from PyQt5 import Qt

class ImageWidgetScroller(Qt.QAbstractScrollArea):
    # It is necessary to derive this class rather than using QAbstractScrollArea directly (which would
    # be consistent with how we make the frame and container dialog for HistogramWidget) because
    # scrollContentsBy is a virtual function that must be overridden, not a signal that can be
    # connected to any arbitrary function (such as ImageWidget._scroll_contents_by).
    def __init__(self, parent, qsurface_format):
        super().__init__(parent)
        self.setFrameShape(Qt.QFrame.StyledPanel)
        self.setFrameShadow(Qt.QFrame.Raised)
        self.image_widget = ImageWidget(self, qsurface_format)
        self.image_widget.show()

    def scrollContentsBy(self, dx, dy):
        self.image_widget._scroll_contents_by(dx, dy)

    def resizeEvent(self, event):
        s = event.size()
        self.image_widget.setGeometry(0, 0, s.width(), s.height())

class ImageWidget(CanvasWidget):
    _ZOOM_PRESETS = numpy.array((10, 5, 2, 1.5, 1, .75, .5, .25, .1), dtype=numpy.float64)
    _ZOOM_MIN_MAX = (.01, 10000.0)
    _ZOOM_DEFAULT_PRESET_IDX = 4
    _ZOOM_CLICK_SCALE_FACTOR = .25

    zoom_changed = Qt.pyqtSignal(int, float)

    def __init__(self, scroller, qsurface_format):
        super().__init__(scroller, qsurface_format)
        self.histogram_widget = None
        self._scroller = scroller
        self._image = None
        self._image_aspect_ratio = None
        self._glsl_prog_g = None
        self._glsl_prog_ga = None
        self._glsl_prog_rgb = None
        self._glsl_prog_rgba = None
        self._image_type_to_glsl_prog = None
        self._tex = None
        self._frag_to_tex = Qt.QTransform()
        self.setMinimumSize(Qt.QSize(100,100))
        self._zoom_preset_idx = self._ZOOM_DEFAULT_PRESET_IDX
        self._custom_zoom = 0
        self._zoom_to_fit = False
        self._pan = Qt.QPoint()

    def initializeGL(self):
        self._init_glfs()
        self._glfs.glClearColor(0,0,0,1)
        self._glfs.glClearDepth(1)
        self._glsl_prog_g = self._build_shader_prog('g',
                                                    'image_widget_vertex_shader.glsl',
                                                    'image_widget_fragment_shader_g.glsl')
        self._glsl_prog_ga = self._build_shader_prog('ga',
                                                     'image_widget_vertex_shader.glsl',
                                                     'image_widget_fragment_shader_ga.glsl')
        self._glsl_prog_rgb = self._build_shader_prog('rgb',
                                                      'image_widget_vertex_shader.glsl',
                                                      'image_widget_fragment_shader_rgb.glsl')
        self._glsl_prog_rgba = self._build_shader_prog('rgba',
                                                       'image_widget_vertex_shader.glsl',
                                                       'image_widget_fragment_shader_rgba.glsl')
        self._image_type_to_glsl_prog = {'g'   : self._glsl_prog_g,
                                         'ga'  : self._glsl_prog_ga,
                                         'rgb' : self._glsl_prog_rgb,
                                         'rgba': self._glsl_prog_rgba}
        self._make_quad_buffer()

    def paintGL(self):
        self._glfs.glClear(self._glfs.GL_COLOR_BUFFER_BIT | self._glfs.GL_DEPTH_BUFFER_BIT)
        if self._image is not None:
            prog = self._image_type_to_glsl_prog[self._image.type]
            prog.bind()
            self._quad_buffer.bind()
            self._tex.bind()
            vert_coord_loc = prog.attributeLocation('vert_coord')
            quad_vao_binder = Qt.QOpenGLVertexArrayObject.Binder(self._quad_vao)
            prog.enableAttributeArray(vert_coord_loc)
            prog.setAttributeBuffer(vert_coord_loc, self._glfs.GL_FLOAT, 0, 2, 0)
            prog.setUniformValue('tex', 0)
            prog.setUniformValue('frag_to_tex', self._frag_to_tex)
            if self._image.is_grayscale:
                if self.histogram_widget.rescale_enabled:
                    gamma = self.histogram_widget.gamma
                    min_max = numpy.array((self.histogram_widget.min, self.histogram_widget.max))
                else:
                    gamma = 1
                    min_max = self._image.range
                prog.setUniformValue('gamma', gamma)
                self._normalize_min_max(min_max)
                prog.setUniformValue('intensity_rescale_min', min_max[0])
                prog.setUniformValue('intensity_rescale_range', min_max[1] - min_max[0])
            else:
                if self.histogram_widget.rescale_enabled:
                    gammas = (self.histogram_widget.gamma_red, self.histogram_widget.gamma_green, self.histogram_widget.gamma_blue)
                    min_maxs = numpy.array(((self.histogram_widget.min_red, self.histogram_widget.min_green, self.histogram_widget.min_blue),
                                            (self.histogram_widget.max_red, self.histogram_widget.max_green, self.histogram_widget.max_blue)))
                else:
                    gammas = (1,1,1)
                    min_max = self._image.range
                    min_maxs = numpy.array((min_max,)*3).T
                prog.setUniformValue('gammas', *gammas)
                self._normalize_min_max(min_maxs)
                prog.setUniformValue('intensity_rescale_mins', *min_maxs[0])
                prog.setUniformValue('intensity_rescale_ranges', *(min_maxs[1]-min_maxs[0]))
            self._glfs.glEnableClientState(self._glfs.GL_VERTEX_ARRAY)
            self._glfs.glDrawArrays(self._glfs.GL_TRIANGLE_FAN, 0, 4)
            self._tex.release()
            self._quad_buffer.release()
            prog.release()

    def resizeGL(self, x, y):
        self._update_scroller_ranges()

    def mouseMoveEvent(self, event):
        if self._image is not None:
            pos = event.pos()
            x_in_image = pos.x() < self._image.size.width() and pos.x() >= 0
            y_in_image = pos.y() < self._image.size.height() and pos.y() >= 0
            mst = 'x:{} y:{} '.format(pos.x() if x_in_image else '-', pos.y() if y_in_image else '-')
            image_type = self._image.type
            vt = '(' + ' '.join((c + ':{}' for c in image_type)) + ')'
            if x_in_image and y_in_image:
                if len(image_type) == 1:
                    vt = vt.format(self._image.data[pos.y(), pos.x()])
                else:
                    vt = vt.format(*self._image.data[pos.y(), pos.x()])
                mst += vt
            else:
                mst += vt.format(*('-' * len(image_type)))
            self.request_mouseover_info_status_text_change.emit(mst)

    def _scroll_contents_by(self, dx, dy):
        self._pan.setX(self._scroller.horizontalScrollBar().value())
        self._pan.setY(self._scroller.verticalScrollBar().value())
        if self._image is not None:
            self._update_frag_to_tex()
        self.update()

    def _update_scroller_ranges(self):
        if self._zoom_to_fit:
            self._scroller.horizontalScrollBar().setRange(0,0)
            self._scroller.verticalScrollBar().setRange(0,0)
        else:
            z = self._custom_zoom if self._zoom_preset_idx == -1 else self._ZOOM_PRESETS[self._zoom_preset_idx]
            def do_axis(i, w, s):
                i *= z
                r = math.ceil(i - w)
                if r < 0:
                    r = 0
                s.setRange(0, r)
                s.setPageStep(w)
            im_sz = Qt.QSize() if self._image is None else self._image.size
            v_sz = self.size()
            do_axis(im_sz.width(), v_sz.width(), self._scroller.horizontalScrollBar())
            do_axis(im_sz.height(), v_sz.height(), self._scroller.verticalScrollBar())
        if self._image is not None:
            self._update_frag_to_tex()

    def _update_frag_to_tex(self):
        view_size = self.size()
        image_size = self._image.size
        # Desired image rect in terms of Qt local widget coordinates is t applied to r
        r = Qt.QPolygonF((Qt.QPointF(0, 0),
                          Qt.QPointF(image_size.width(), 0),
                          Qt.QPointF(image_size.width(), image_size.height()),
                          Qt.QPointF(0, image_size.height())))
        t = Qt.QTransform()
        if self._zoom_to_fit:
            view_aspect_ratio = view_size.width() / view_size.height()
            image_to_view_ratio = self._image_aspect_ratio / view_aspect_ratio
            if image_to_view_ratio <= 1:
                # Image is proportionally taller than the viewport and will be scaled such that the image
                # fills the viewport vertically and is centered horizontally
                zoom_factor = view_size.height() / image_size.height()
                t.translate((view_size.width() - zoom_factor*image_size.width()) / 2, 0)
            else:
                # Image is proportionally wider than the viewport and will be scaled such that the image
                # fills the viewport horizontally and is centered vertically
                zoom_factor = view_size.width() / image_size.width()
                t.translate(0, (view_size.height() - zoom_factor*image_size.height()) / 2)
        else:
            zoom_factor = self._custom_zoom if self._zoom_preset_idx == -1 else ImageWidget._ZOOM_PRESETS[self._zoom_preset_idx]
            t.translate(-self._pan.x(), -(self._scroller.verticalScrollBar().maximum()-self._pan.y()))
            centering = numpy.array((image_size.width(), image_size.height()), dtype=numpy.float64)
            centering *= zoom_factor
            centering = numpy.array((view_size.width(), view_size.height())) - centering
            centering[centering < 0] = 0
            centering /= 2
            t.translate(*centering)
        t.scale(zoom_factor, zoom_factor)
        if not Qt.QTransform.quadToSquare(t.map(r), self._frag_to_tex):
            raise RuntimeError('Failed to compute gl_FragCoord to texture coordinate transformation matrix.')

    def _on_image_changed(self, image):
        try:
            self.makeCurrent()
            if self._image is not None and (image is None or self._image.size != image.size):
                self._tex = None
                self._image = None
            if image is not None:
                desired_texture_format = ImageWidget._IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT[image.type]
                if self._tex is None or not self._tex.isCreated() or self._tex.format() != desired_texture_format:
                    self._tex = Qt.QOpenGLTexture(Qt.QOpenGLTexture.Target2D)
                    self._tex.setFormat(desired_texture_format)
                    self._tex.setWrapMode(Qt.QOpenGLTexture.ClampToEdge)
                    self._tex.setAutoMipMapGenerationEnabled(True)
                    self._tex.setSize(image.size.width(), image.size.height(), 1)
                    self._tex.setMipLevels(4)
                    self._tex.allocateStorage()
                self._tex.setMinMagFilters(Qt.QOpenGLTexture.LinearMipMapLinear, Qt.QOpenGLTexture.Nearest)
                self._tex.bind()
                pixel_transfer_opts = Qt.QOpenGLPixelTransferOptions()
                pixel_transfer_opts.setAlignment(1)
                self._tex.setData(ImageWidget._IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT[image.type],
                                  ImageWidget._NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE[image.dtype],
                                  image.data.ctypes.data,
                                  pixel_transfer_opts)
                self._tex.release()
                self._image = image
                self._image_aspect_ratio = image.size.width() / image.size.height()
            self._update_scroller_ranges()
            self.update()
        finally:
            self.doneCurrent()

    def _normalize_min_max(self, min_max):
        if self._image.dtype != numpy.float32:
            r = self._image.range
            min_max -= r[0]
            min_max /= r[1] - r[0]

    @property
    def zoom_to_fit(self):
        return self._zoom_to_fit

    @zoom_to_fit.setter
    def zoom_to_fit(self, zoom_to_fit):
        self._zoom_to_fit = zoom_to_fit
        self._update_scroller_ranges()
        self.update()

    @property
    def custom_zoom(self):
        return self._custom_zoom

    @custom_zoom.setter
    def custom_zoom(self, custom_zoom):
        self._custom_zoom = custom_zoom
        self._zoom_preset_idx = -1
        self._update_scroller_ranges()
        self.zoom_changed.emit(self._zoom_preset_idx, self._custom_zoom)
        self.update()

    @property
    def zoom_preset_idx(self):
        return self._zoom_preset_idx

    @zoom_preset_idx.setter
    def zoom_preset_idx(self, idx):
        if idx < 0 or idx >= ImageWidget._ZOOM_PRESETS.shape[0]:
            raise ValueError('idx must be in the range [0, {}).'.format(ImageWidget._ZOOM_PRESETS.shape[0]))
        self._update_scroller_ranges()
        self.zoom_changed.emit(self._zoom_preset_idx, self._custom_zoom)
        self.update()
