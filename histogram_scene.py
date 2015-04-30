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
from .shared_resources import GL, UNIQUE_QGRAPHICSITEM_TYPE
import math
import numpy
from PyQt5 import Qt
from .shader_scene import ShaderItem, ShaderScene, ShaderTexture
import sys

class HistogramScene(ShaderScene):
    def __init__(self, parent, HistogramItemClass):
        super().__init__(parent)
        self.setSceneRect(0, 0, 1, 1)
        self._image_item = None
        self.histogram_item = HistogramItemClass()
        self.addItem(self.histogram_item)

    @property
    def image_item(self):
        return self._image_item

    @image_item.setter
    def image_item(self, image_item):
        if image_item is not self._image_item:
            self._image_item = image_item
            # TODO: cause control widgets to update... after adding them back in...

class HistogramItem(ShaderItem):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()

    def __init__(self, graphics_item_parent=None):
        super().__init__(graphics_item_parent)
        self.image = None
        self._image_id = 0
        self._bounding_rect = Qt.QRectF(0, 0, 1, 1)
        self.tex = None

    def type(self):
        return HistogramItem.QGRAPHICSITEM_TYPE

    def boundingRect(self):
        return self._bounding_rect

    def paint(self, qpainter, option, widget):
        if widget is None:
            print('WARNING: histogram_view.HistogramItem.paint called with widget=None.  Ensure that view caching is disabled.')
        elif self.image is None:
            if self.tex is not None:
                self.tex.destroy()
                self.tex = None
        else:
            image = self.image
            view = widget.view
            scene = self.scene()
            if not image.is_grayscale:
                return
                # personal time todo: per-channel RGB histogram support
            with ExitStack() as stack:
                qpainter.beginNativePainting()
                stack.callback(qpainter.endNativePainting)
                gl = GL()
                desired_shader_type = 'g' if image.type in ('g', 'ga') else 'rgb'
                if desired_shader_type in self.progs:
                    prog = self.progs[desired_shader_type]
                else:
                    prog = self.build_shader_prog(desired_shader_type,
                                                  'histogram_widget_vertex_shader.glsl',
                                                  'histogram_widget_fragment_shader_{}.glsl'.format(desired_shader_type))
                desired_tex_width = image.histogram.shape[-1]
                tex = self.tex
                if tex is not None:
                    if tex.width != desired_tex_width:
                        tex.destroy()
                        tex = self.tex = None
                if tex is None:
                    tex = ShaderTexture(gl.GL_TEXTURE_1D)
                    tex.bind()
                    stack.callback(tex.release)
                    gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
                    gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
                    # tex stores histogram bin counts - values that are intended to be addressed by element without
                    # interpolation.  Thus, nearest neighbor for texture filtering.
                    gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
                    gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
                    tex.image_id = -1
                else:
                    tex.bind()
                    stack.callback(tex.release)
                if image.is_grayscale:
                    if image.type == 'g':
                        histogram = image.histogram
                        max_bin_val = histogram[image.max_histogram_bin]
                    else:
                        histogram = image.histogram[0]
                        max_bin_val = histogram[image.max_histogram_bin[0]]
                    if tex.image_id != self._image_id:
                        orig_unpack_alignment = gl.glGetIntegerv(gl.GL_UNPACK_ALIGNMENT)
                        if orig_unpack_alignment != 1:
                            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
                            # QPainter font rendering for OpenGL surfaces will become broken if we do not restore GL_UNPACK_ALIGNMENT
                            # to whatever QPainter had it set to (when it prepared the OpenGL context for our use as a result of
                            # qpainter.beginNativePainting()).
                            stack.callback(lambda oua=orig_unpack_alignment: gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, oua))
                        gl.glTexImage1D(gl.GL_TEXTURE_1D, 0,
                                        gl.GL_LUMINANCE32UI_EXT, desired_tex_width, 0,
                                        gl.GL_LUMINANCE_INTEGER_EXT, gl.GL_UNSIGNED_INT,
                                        histogram.data)
                        tex.image_id = self._image_id
                        tex.width = desired_tex_width
                        self.tex = tex
                    view.quad_buffer.bind()
                    stack.callback(view.quad_buffer.release)
                    view.quad_vao.bind()
                    stack.callback(view.quad_vao.release)
                    prog.bind()
                    stack.callback(prog.release)
                    vert_coord_loc = prog.attributeLocation('vert_coord')
                    prog.enableAttributeArray(vert_coord_loc)
                    prog.setAttributeBuffer(vert_coord_loc, gl.GL_FLOAT, 0, 2, 0)
                    prog.setUniformValue('tex', 0)
                    prog.setUniformValue('inv_view_size', 1/widget.size().width(), 1/widget.size().height())
                    inv_max_transformed_bin_val = max_bin_val**-scene.gamma_gamma
                    prog.setUniformValue('inv_max_transformed_bin_val', inv_max_transformed_bin_val)
                    prog.setUniformValue('gamma_gamma', scene.gamma_gamma)
                    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
                    gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)
                else:
                    pass
                    # personal time todo: per-channel RGB histogram support

    def hoverMoveEvent(self, event):
        image = self.image
        if image is not None:
            x = event.pos().x()
            if x >= 0 and x <= 1:
                if image.is_grayscale:
                    image_type = image.type
                    histogram = image.histogram
                    range_ = image.range
                    bin_count = histogram.shape[-1]
                    bin = int(x * bin_count)
                    bin_width = (range_[1] - range_[0]) / bin_count
                    if image.dtype == numpy.float32:
                        mst = '[{},{}) '.format(range_[0] + bin*bin_width, range_[0] + (bin+1)*bin_width)
                    else:
                        mst = '[{},{}] '.format(math.ceil(bin*bin_width), math.floor((bin+1)*bin_width))
                    vt = '(' + ' '.join((c + ':{}' for c in image_type)) + ')'
                    if len(image_type) == 1:
                        vt = vt.format(histogram[bin])
                    else:
                        vt = vt.format(*histogram[:,bin])
                    self.scene().update_contextual_info(mst + vt, False, self)
                else:
                    pass
                    # personal time todo: per-channel RGB histogram support

    def hoverLeaveEvent(self, event):
        self.scene().clear_contextual_info(self)
