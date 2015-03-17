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
from .gl_resources import GL, NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE, IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT, IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT
import math
import numpy
from PyQt5 import Qt
from .shader_scene import ShaderScene, ShaderItem
from .shader_view import ShaderView
import sys

class ImageScene(ShaderScene):
    def __init__(self, parent):
        super().__init__(parent)
        self.image_item = ImageItem()
        self.addItem(self.image_item)
        self._histogram_scene = None

    def on_image_changing(self, image):
        self.image_item.on_image_changing(image)
        self.setSceneRect(self.image_item.boundingRect())
        for view in self.views():
            view.on_image_changing(image)

    def on_histogram_gamma_or_min_max_changed(self):
        self.image_item.update()

    @property
    def histogram_scene(self):
        return self._histogram_scene

    @histogram_scene.setter
    def histogram_scene(self, histogram_scene):
        if self._histogram_scene is not None:
            self._histogram_scene.gamma_or_min_max_changed.disconnect(self.on_histogram_gamma_or_min_max_changed)
        histogram_scene.gamma_or_min_max_changed.connect(self.on_histogram_gamma_or_min_max_changed)
        self._histogram_scene = histogram_scene
        self.on_histogram_gamma_or_min_max_changed()

class ImageItem(ShaderItem):
    def boundingRect(self):
        return Qt.QRectF() if self.image is None else Qt.QRectF(Qt.QPointF(), Qt.QSizeF(self.image.size))

    def paint(self, qpainter, option, widget):
        if widget is None:
            print('WARNING: image_view.ImageItem.paint called with widget=None.  Ensure that view caching is disabled.')
        elif self.image is None:
            if widget.view in self.view_resources:
                vrs = self.view_resources[widget.view]
                if 'tex' in vrs:
                    vrs['tex'][0].destroy()
                    del vrs['tex']
        else:
            image = self.image
            desired_texture_format = IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT[image.type]
            view = widget.view
            with ExitStack() as stack:
                qpainter.beginNativePainting()
                stack.callback(qpainter.endNativePainting)
                if view in self.view_resources:
                    vrs = self.view_resources[view]
                else:
                    self.view_resources[view] = vrs = {}
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
                    tex.setData(IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT[image.type],
                                NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE[image.dtype],
                                image.data.ctypes.data,
                                pixel_transfer_opts)
                    vrs['tex'] = tex, self._image_id
                prog = vrs['progs'][self.image.type]
                prog.bind()
                stack.callback(prog.release)
                view.quad_buffer.bind()
                stack.callback(view.quad_buffer.release)
                view.quad_vao.bind()
                stack.callback(view.quad_vao.release)
                gl = GL()
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
                histogram_scene = view.scene().histogram_scene
                if self.image.is_grayscale:
                    if histogram_scene.rescale_enabled:
                        gamma = histogram_scene.gamma
                        min_max = numpy.array((histogram_scene.min, histogram_scene.max), dtype=float)
                        self._normalize_min_max(min_max)
                    else:
                        gamma = 1
                        min_max = numpy.array((0,1), dtype=float)
                    prog.setUniformValue('gamma', gamma)
                    prog.setUniformValue('intensity_rescale_min', min_max[0])
                    prog.setUniformValue('intensity_rescale_range', min_max[1] - min_max[0])
                else:
                    if histogram_scene.rescale_enabled:
                        gammas = (histogram_scene.gamma_red, histogram_scene.gamma_green, histogram_scene.gamma_blue)
                        min_maxs = numpy.array(((histogram_scene.min_red, histogram_scene.min_green, histogram_scene.min_blue),
                                                (histogram_scene.max_red, histogram_scene.max_green, histogram_scene.max_blue)))
                    else:
                        gammas = (1,1,1)
                        min_max = self.image.range
                        min_maxs = numpy.array((min_max,)*3).T
                    prog.setUniformValue('gammas', *gammas)
                    self._normalize_min_max(min_maxs)
                    prog.setUniformValue('intensity_rescale_mins', *min_maxs[0])
                    prog.setUniformValue('intensity_rescale_ranges', *(min_maxs[1]-min_maxs[0]))
                gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
                gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)

    def hoverMoveEvent(self, event):
        if self.image is not None:
            # NB: event.pos() is a QPointF, and one may call QPointF.toPoint(), as in the following line,
            # to get a QPoint from it.  However, toPoint() rounds x and y coordinates to the nearest int,
            # which would cause us to erroneously report mouse position as being over the pixel to the
            # right and/or below if the view with the mouse cursor is zoomed in such that an image pixel
            # occupies more than one screen pixel and the cursor is over the right and/or bottom half
            # of a pixel.
#           pos = event.pos().toPoint()
            pos = Qt.QPoint(event.pos().x(), event.pos().y())
            if Qt.QRect(Qt.QPoint(), self.image.size).contains(pos):
                mst = 'x:{} y:{} '.format(pos.x(), pos.y())
                image_type = self.image.type
                vt = '(' + ' '.join((c + ':{}' for c in image_type)) + ')'
                if len(image_type) == 1:
                    vt = vt.format(self.image.data[pos.x(), pos.y()])
                else:
                    vt = vt.format(*self.image.data[pos.x(), pos.y()])
                self.scene().update_mouseover_info(mst + vt, False, self)

    def hoverLeaveEvent(self, event):
        self.scene().clear_mouseover_info(self)

    def on_image_changing(self, image):
        if self.image is None and image is not None or \
           self.image is not None and (image is None or self.image.size != image.size):
            self.prepareGeometryChange()
        super().on_image_changing(image)

    def free_shader_view_resources(self, shader_view):
        # TODO: replace QOpenGLTexture usage with luminance-extension-format-supporting ShaderTexture and eliminate this
        # override
        if shader_view in self.view_resources:
            vrs = self.view_resources[shader_view]
            if 'tex' in vrs:
                vrs['tex'][0].destroy()
                del vrs['tex']
        super().free_shader_view_resources(shader_view)
