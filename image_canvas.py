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

from . import canvas
from contextlib import ExitStack
import math
import numpy
from PyQt5 import Qt

class ImageView(canvas.CanvasView):
    _ZOOM_PRESETS = numpy.array((10, 5, 2, 1.5, 1, .75, .5, .25, .1), dtype=numpy.float64)
    _ZOOM_MIN_MAX = (.01, 10000.0)
    _ZOOM_DEFAULT_PRESET_IDX = 4
    _ZOOM_CLICK_SCALE_FACTOR = .25

    zoom_changed = Qt.pyqtSignal(int, float)
    zoom_to_fit_changed = Qt.pyqtSignal(bool)

    def __init__(self, canvas_scene, parent):
        super().__init__(canvas_scene, parent)
        self.histogram_view = None
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
        self.setDragMode(Qt.QGraphicsView.ScrollHandDrag)

    def _on_image_changed(self, image):
        if self._zoom_to_fit:
            self.fitInView(self.scene().image_item, Qt.Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        if self._zoom_to_fit:
            self.fitInView(self.scene().image_item, Qt.Qt.KeepAspectRatio)
        else:
            super().resizeEvent(event)

    @property
    def zoom_to_fit(self):
        return self._zoom_to_fit

    @zoom_to_fit.setter
    def zoom_to_fit(self, zoom_to_fit):
        self._zoom_to_fit = zoom_to_fit
        self._zoom()

    @property
    def custom_zoom(self):
        return self._custom_zoom

    @custom_zoom.setter
    def custom_zoom(self, custom_zoom):
        if self._custom_zoom < ImageView._ZOOM_MIN_MAX[0] or self._custom_zoom > ImageView._ZOOM_MIN_MAX[1]:
            raise ValueError('Value must be in the range [{}, {}].'.format(*ImageView._ZOOM_MIN_MAX))
        self._custom_zoom = custom_zoom
        self._zoom_preset_idx = -1
        self._zoom()

    @property
    def zoom_preset_idx(self):
        return self._zoom_preset_idx

    @zoom_preset_idx.setter
    def zoom_preset_idx(self, idx):
        if idx < 0 or idx >= ImageView._ZOOM_PRESETS.shape[0]:
            raise ValueError('idx must be in the range [0, {}).'.format(ImageView._ZOOM_PRESETS.shape[0]))
        self._zoom_preset_idx = idx
        self._custom_zoom = 0
        self._zoom()

    def _zoom(self, zoom_to_fit_changed=False):
        if self._zoom_to_fit:
            self.fitInView(self.scene().image_item, Qt.Qt.KeepAspectRatio)
        else:
            zoom_factor = self._custom_zoom if self._zoom_preset_idx == -1 else ImageView._ZOOM_PRESETS[self._zoom_preset_idx]
            old_transform = Qt.QTransform(self.transform())
            self.resetTransform()
            self.translate(old_transform.dx(), old_transform.dy())
            self.scale(zoom_factor, zoom_factor)
        if zoom_to_fit_changed:
            self.zoom_to_fit_changed.emit(self._zoom_to_fit)
        else:
            self.zoom_changed.emit(self._zoom_preset_idx, self._custom_zoom)

class ImageItem(canvas.CanvasGLItem):
    def __init__(self, graphics_item_parent=None):
        super().__init__(graphics_item_parent)
        self._image = None
        self._image_id = 0

    def boundingRect(self):
        return Qt.QRectF() if self._image is None else Qt.QRectF(Qt.QPointF(), Qt.QSizeF(self._image.size))

    def paint(self, qpainter, option, widget):
        if widget is None:
            print('WARNING: image_view.ImageItem.paint called with widget=None.  Ensure that view caching is disabled.')
        elif self._image is None:
            if widget.view in self._view_resources:
                vrs = self._view_resources[widget.view]
                if 'tex' in vrs:
#                   with canvas.native_painting(qpainter):
#                       vrs['tex'].destroy()
                    vrs['tex'][0].destroy()
                    del vrs['tex']
        else:
            image = self._image
            desired_texture_format = canvas.IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT[image.type]
            view = widget.view
            with ExitStack() as stack:
                qpainter.beginNativePainting()
                stack.callback(qpainter.endNativePainting)
                if view in self._view_resources:
                    vrs = self._view_resources[view]
                else:
                    self._view_resources[view] = vrs = {}
                    self.build_shader_prog('g',
                                           'image_widget_vertex_shader.glsl',
                                           'image_widget_fragment_shader_g.glsl',
                                           view)
                    self.build_shader_prog('ga',
                                           'image_widget_vertex_shader.glsl',
                                           'image_widget_fragment_shader_ga.glsl',
                                           view)
                    self.build_shader_prog('rgb',
                                           'image_widget_vertex_shader.glsl',
                                           'image_widget_fragment_shader_rgb.glsl',
                                           view)
                    self.build_shader_prog('rgba',
                                           'image_widget_vertex_shader.glsl',
                                           'image_widget_fragment_shader_rgba.glsl',
                                           view)
                if 'tex' in vrs:
                    tex, tex_image_id = vrs['tex']
                    if image.size != Qt.QSize(tex.width(), tex.height()) or tex.format() != desired_texture_format:
                        tex.destroy()
                        del vrs['tex']
                if 'tex' not in vrs:
                    tex = Qt.QOpenGLTexture(Qt.QOpenGLTexture.Target2D)
                    tex.setFormat(desired_texture_format)
                    tex.setWrapMode(Qt.QOpenGLTexture.ClampToEdge)
                    tex.setAutoMipMapGenerationEnabled(True)
                    tex.setSize(image.size.width(), image.size.height(), 1)
                    tex.setMipLevels(4)
                    tex.allocateStorage()
                    tex.setMinMagFilters(Qt.QOpenGLTexture.LinearMipMapLinear, Qt.QOpenGLTexture.Nearest)
                    tex_image_id = -1
                tex.bind()
                stack.callback(lambda: tex.release(0))
                if tex_image_id != self._image_id:
                    pixel_transfer_opts = Qt.QOpenGLPixelTransferOptions()
                    pixel_transfer_opts.setAlignment(1)
                    tex.setData(canvas.IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT[image.type],
                                canvas.NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE[image.dtype],
                                image.data.ctypes.data,
                                pixel_transfer_opts)
                    vrs['tex'] = tex, self._image_id
                prog = vrs['progs'][self._image.type]
                prog.bind()
                stack.callback(prog.release)
                view.quad_buffer.bind()
                stack.callback(view.quad_buffer.release)
                view.quad_vao.bind()
                stack.callback(view.quad_vao.release)
                gl = view.glfs
                vert_coord_loc = prog.attributeLocation('vert_coord')
                prog.enableAttributeArray(vert_coord_loc)
                prog.setAttributeBuffer(vert_coord_loc, gl.GL_FLOAT, 0, 2, 0)
                prog.setUniformValue('tex', 0)
                frag_to_tex = Qt.QTransform()
                frame = Qt.QPolygonF(view.mapFromScene(self.boundingRect()))
                if not qpainter.transform().quadToSquare(frame, frag_to_tex):
                    raise RuntimeError('Failed to compute gl_FragCoord to texture coordinate transformation matrix.')
                prog.setUniformValue('frag_to_tex', frag_to_tex)
                prog.setUniformValue('viewport_height', float(widget.size().height()))
                histogram_view = view.histogram_view
                if self._image.is_grayscale:
                    if histogram_view.rescale_enabled:
                        gamma = histogram_view.gamma
                        min_max = numpy.array((histogram_view.min, histogram_view.max), dtype=float)
                        self._normalize_min_max(min_max)
                    else:
                        gamma = 1
                        min_max = numpy.array((0,1), dtype=float)
                    prog.setUniformValue('gamma', gamma)
                    prog.setUniformValue('intensity_rescale_min', min_max[0])
                    prog.setUniformValue('intensity_rescale_range', min_max[1] - min_max[0])
                else:
                    if histogram_view.rescale_enabled:
                        gammas = (histogram_view.gamma_red, histogram_view.gamma_green, histogram_view.gamma_blue)
                        min_maxs = numpy.array(((histogram_view.min_red, histogram_view.min_green, histogram_view.min_blue),
                                                (histogram_view.max_red, histogram_view.max_green, histogram_view.max_blue)))
                    else:
                        gammas = (1,1,1)
                        min_max = self._image.range
                        min_maxs = numpy.array((min_max,)*3).T
                    prog.setUniformValue('gammas', *gammas)
                    self._normalize_min_max(min_maxs)
                    prog.setUniformValue('intensity_rescale_mins', *min_maxs[0])
                    prog.setUniformValue('intensity_rescale_ranges', *(min_maxs[1]-min_maxs[0]))
                gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
                gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)
                    
    def on_image_changed(self, image):
        if self._image is None and image is not None or \
           self._image is not None and (image is None or self._image.size != image.size):
            self.prepareGeometryChange()
        self._image = image
        self._image_id += 1
        self.update()

    def release_resources_for_view(self, canvas_view):
        if canvas_view in self._view_resources:
            vrs = self._view_resources[canvas_view]
            if 'tex' in vrs:
                vrs['tex'][0].destroy()
                del vrs['tex']
        super().release_resources_for_view(canvas_view)

    def _normalize_min_max(self, min_max):
        if self._image.dtype != numpy.float32:
            r = self._image.range
            min_max -= r[0]
            min_max /= r[1] - r[0]

class ImageScene(canvas.CanvasScene):
    def __init__(self, parent):
        super().__init__(parent)
        self.image_item = ImageItem()
        self.addItem(self.image_item)
#       color = Qt.QColor(Qt.Qt.blue)
#       color.setAlphaF(0.5)
#       brush = Qt.QBrush(color)
#       color2 = Qt.QColor(Qt.Qt.green)
#       color2.setAlphaF(0.8)
#       pen = Qt.QPen(color2)
#       pen.setWidth(5)
#       self.foo_item = self.addRect(20,10,200,100,pen,brush)

    def _on_image_changed(self, image):
        self.image_item.on_image_changed(image)
        self.setSceneRect(self.image_item.boundingRect())

#def qtransform_to_numpy(t):
#   return numpy.array(((t.m11(),t.m12(),t.m13()),(t.m21(),t.m22(),t.m23()),(t.m31(),t.m32(),t.m33())))
