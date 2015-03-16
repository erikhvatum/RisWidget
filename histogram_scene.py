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
from .gl_resources import GL
import math
import numpy
from PyQt5 import Qt
from .shader_scene import ShaderItem, ShaderScene, ShaderTexture
import sys

class HistogramScene(ShaderScene):
    gamma_or_min_max_changed = Qt.pyqtSignal()

#   _scalar_props = []
#   _min_max_props = {}

#   max = MinMaxProp(_scalar_props, _min_max_props, 'max')
#   min = MinMaxProp(_scalar_props, _min_max_props, 'min')

    def __init__(self, parent):
        super().__init__(parent)
        self.histogram_item = HistogramItem()
        self.addItem(self.histogram_item)
        self.gamma = 1.0
        self.gamma_gamma = 1.0
        self.rescale_enabled = True
        self.min = 0
        self.max = 65535

#       self._allow_inversion = True # Set to True during initialization for convenience...
#       for scalar_prop in HistogramView._scalar_props:
#           scalar_prop.instantiate(self, layout)
#       self._allow_inversion = False # ... and, enough stuff has been initialized that this can now be set to False without trouble

    def on_image_changing(self, image):
        self.histogram_item.on_image_changing(image)

class HistogramItem(ShaderItem):
    def __init__(self, graphics_item_parent=None):
        super().__init__(graphics_item_parent)
        self._image = None
        self._image_id = 0
        self._bounding_rect = Qt.QRectF()

    def boundingRect(self):
        return Qt.QRectF() if self._image is None else self._bounding_rect

    def _set_bounding_rect(self, rect):
        if self._image is not None:
            self.prepareGeometryChange()
        self._bounding_rect = rect

    def paint(self, qpainter, option, widget):
        if widget is None:
            print('WARNING: histogram_view.HistogramItem.paint called with widget=None.  Ensure that view caching is disabled.')
        elif self._image is None:
            if widget.view in self.view_resources:
                self._del_tex()
        else:
            image = self._image
            view = widget.view
            scene = self.scene()
            gl = GL()
            with ExitStack() as stack:
                qpainter.beginNativePainting()
                stack.callback(qpainter.endNativePainting)
                if view in self.view_resources:
                    vrs = self.view_resources[view]
                else:
                    self.view_resources[view] = vrs = {}
                    self.build_shader_prog('g',
                                            'histogram_widget_vertex_shader.glsl',
                                            'histogram_widget_fragment_shader_g.glsl',
                                            view)
                    vrs['progs']['ga'] = vrs['progs']['g']
                    self.build_shader_prog('rgb',
                                            'histogram_widget_vertex_shader.glsl',
                                            'histogram_widget_fragment_shader_rgb.glsl',
                                            view)
                    vrs['progs']['rgba'] = vrs['progs']['rgb']
                desired_tex_width = image.histogram.shape[-1]
                if 'tex' in vrs:
                    tex = vrs['tex']
                    if tex.width != desired_tex_width:
                        tex.destroy()
                        del vrs['tex']
                if 'tex' not in vrs:
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
                    vrs['tex'] = tex
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
                        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
                        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
                        gl.glTexImage1D(gl.GL_TEXTURE_1D, 0,
                                        gl.GL_LUMINANCE32UI_EXT, desired_tex_width, 0,
                                        gl.GL_LUMINANCE_INTEGER_EXT, gl.GL_UNSIGNED_INT,
                                        histogram.data)
                        tex.image_id = self._image_id
                        tex.width = desired_tex_width
                    view.quad_buffer.bind()
                    stack.callback(view.quad_buffer.release)
                    view.quad_vao.bind()
                    stack.callback(view.quad_vao.release)
                    prog = vrs['progs'][image.type]
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
                    prog.setUniformValue('rescale_enabled', scene.rescale_enabled)
                    if scene.rescale_enabled:
                        prog.setUniformValue('gamma', scene.gamma)
                        min_max = numpy.array((scene.min, scene.max), dtype=float)
                        self._normalize_min_max(min_max)
                        prog.setUniformValue('intensity_rescale_min', min_max[0])
                        prog.setUniformValue('intensity_rescale_range', min_max[1] - min_max[0])
                    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
                    gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)
                else:
                    pass
                    # personal time todo: per-channel RGB histogram support

    def on_image_changing(self, image):
        if (self._image is None) != (image is not None) or \
           self._image is not None and image is not None and self._image.histogram.shape[-1] != image.histogram.shape[-1]:
            self.prepareGeometryChange()
        super().on_image_changing(image)

class GammaPlotItem(Qt.QGraphicsItem):
    pass

class MinItem(Qt.QGraphicsItem):
    pass
