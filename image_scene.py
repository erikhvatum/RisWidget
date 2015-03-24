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
from .shader_scene import ShaderScene, ShaderItem, ShaderTexture, UNIQUE_QGRAPHICSITEM_TYPE
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
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()
    NUMPY_DTYPE_TO_GL_PIXEL_DATA_TYPE_STR = {
        numpy.uint8  : 'GL_UNSIGNED_BYTE',
        numpy.uint16 : 'GL_UNSIGNED_SHORT',
        numpy.uint32 : 'GL_UNSIGNED_INT',
        numpy.float32: 'GL_FLOAT'}
    NUMPY_DTYPE_TO_GL_INTERNAL_FORMAT_TYPE_STR = {
        numpy.uint8  : '8UI_EXT',
        numpy.uint16 : '16UI_EXT',
        numpy.uint32 : '32UI_EXT',
        numpy.float32: ''}
    IMAGE_TYPE_TO_GL_FORMAT_NAME_STR = {
        'g'   : 'LUMINANCE',
        'ga'  : 'LUMINANCE_ALPHA',
        'rgb' : 'RGB',
        'rgba': 'RGBA'}
    IMAGE_TYPE_TO_COMBINE_VT_COMPONENTS = {
        'g'   : 'vec4(vcomponents, vcomponents, vcomponents, 1.0f)',
        'ga'  : 'vec4(vcomponents, vcomponents, vcomponents, tcomponents.a)',
        'rgb' : 'vec4(vcomponents.rgb, 1.0f)',
        'rgba': 'vec4(vcomponents.rgb, tcomponents.a)'}

    def type(self):
        return GammaItem.QGRAPHICSITEM_TYPE

    def boundingRect(self):
        return Qt.QRectF() if self.image is None else Qt.QRectF(Qt.QPointF(), Qt.QSizeF(self.image.size))

    def paint(self, qpainter, option, widget):
        if widget is None:
            print('WARNING: image_view.ImageItem.paint called with widget=None.  Ensure that view caching is disabled.')
        elif self.image is None:
            if widget.view in self.view_resources:
                vrs = self.view_resources[widget.view]
                if 'tex' in vrs:
                    vrs['tex'].destroy()
                    del vrs['tex']
        else:
            with ExitStack() as stack:
                qpainter.beginNativePainting()
                stack.callback(qpainter.endNativePainting)
                gl = GL()
                image = self.image
                view = widget.view
                if view in self.view_resources:
                    vrs = self.view_resources[view]
                else:
                    self.view_resources[view] = vrs = {}
                tex_format_name_str = ImageItem.IMAGE_TYPE_TO_GL_FORMAT_NAME_STR[image.type]
                tex_internal_format_type_str = ImageItem.NUMPY_DTYPE_TO_GL_INTERNAL_FORMAT_TYPE_STR[image.dtype]
                tex_internal_format = getattr(gl, 'GL_{}{}'.format(tex_format_name_str, tex_internal_format_type_str))
                shader_desc = '{}_{}_{}'.format(image.type, tex_format_name_str, tex_internal_format_type_str)
                if 'progs' in vrs:
                    progs = vrs['progs']
                else:
                    progs = vrs['progs'] = {}
                if shader_desc in progs:
                    prog = progs[shader_desc]
                else:
                    if image.dtype == numpy.float32:
                        sampler_t = 'sampler2D'
                        raw_tcomponents_t = 'vec4'
                        raw_tcomponents_to_tcomponents = 'raw_tcomponents'
                    else:
                        sampler_t = 'usampler2D'
                        raw_tcomponents_t = 'uvec4'
                        raw_tcomponents_to_tcomponents = 'vec4(raw_tcomponents)'
                    if image.is_grayscale:
                        vcomponents_t = 'float'
                        extract_vcomponents = 'tcomponents.r'
                        vcomponents_ones_vector = '1.0f'
                    else:
                        vcomponents_t = 'vec3'
                        extract_vcomponents = 'tcomponents.rgb'
                        vcomponents_ones_vector = 'vec3(1.0f, 1.0f, 1.0f)'
                    self.build_shader_prog(shader_desc,
                                           'image_widget_vertex_shader.glsl',
                                           'image_widget_fragment_shader_template.glsl',
                                           view,
                                           sampler_t=sampler_t,
                                           raw_tcomponents_t=raw_tcomponents_t,
                                           raw_tcomponents_to_tcomponents=raw_tcomponents_to_tcomponents,
                                           vcomponents_t=vcomponents_t,
                                           extract_vcomponents=extract_vcomponents,
                                           vcomponents_ones_vector=vcomponents_ones_vector,
                                           combine_vt_components=ImageItem.IMAGE_TYPE_TO_COMBINE_VT_COMPONENTS[image.type])
                    prog = progs[shader_desc]
                prog.bind()
                stack.callback(prog.release)
                if 'tex' in vrs:
                    tex = vrs['tex']
                    if tex.size != image.size or tex.internal_format != tex_internal_format:
                        tex.destroy()
                        del vrs['tex']
                if 'tex' not in vrs:
                    tex = ShaderTexture(gl.GL_TEXTURE_2D)
                    tex.bind()
                    stack.callback(tex.release)
                    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
                    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
                    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
                    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
                    tex.image_id = -1
                    tex.size = None
                    vrs['tex'] = tex
                else:
                    tex.bind()
                    stack.callback(tex.release)
                if tex.image_id != self._image_id:
                    orig_unpack_alignment = gl.glGetIntegerv(gl.GL_UNPACK_ALIGNMENT)
                    if orig_unpack_alignment != 1:
                        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
                        # QPainter font rendering for OpenGL surfaces will become broken if we do not restore GL_UNPACK_ALIGNMENT
                        # to whatever QPainter had it set to (when it prepared the OpenGL context for our use as a result of
                        # qpainter.beginNativePainting()).
                        stack.callback(lambda oua=orig_unpack_alignment: gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, oua))
                    src_format_str = 'GL_' + tex_format_name_str
                    if image.type != numpy.float32:
                        src_format_str += '_INTEGER_EXT'
                    src_format = getattr(gl, src_format_str)
                    src_gl_data_type = getattr(gl, ImageItem.NUMPY_DTYPE_TO_GL_PIXEL_DATA_TYPE_STR[image.dtype])
                    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, tex_internal_format,
                                    image.size.width(), image.size.height(), 0,
                                    src_format,
                                    src_gl_data_type,
                                    memoryview(image.data.data))
                    tex.internal_format = tex_internal_format
                    tex.size = image.size
                    tex.image_id = self._image_id
                view.quad_buffer.bind()
                stack.callback(view.quad_buffer.release)
                view.quad_vao.bind()
                stack.callback(view.quad_vao.release)
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
                if image.is_grayscale:
                    if histogram_scene.rescale_enabled:
                        gamma = histogram_scene.gamma
                        min_max = numpy.array((histogram_scene.min, histogram_scene.max), dtype=float)
                        self._normalize_min_max(min_max)
                    else:
                        gamma = 1
                        min_max = numpy.array((0,1), dtype=float)
                    prog.setUniformValue('gammas', gamma)
                    prog.setUniformValue('vcomponent_rescale_mins', min_max[0])
                    prog.setUniformValue('vcomponent_rescale_ranges', min_max[1] - min_max[0])
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
                    prog.setUniformValue('vcomponent_rescale_mins', *min_maxs[0])
                    prog.setUniformValue('vcomponent_rescale_ranges', *(min_maxs[1]-min_maxs[0]))
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
