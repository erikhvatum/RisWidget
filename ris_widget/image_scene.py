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
import math
import numpy
from PyQt5 import Qt
from string import Template
import sys
#from ._qt_debug import qtransform_to_numpy
from .immutable_image_with_mutable_properties import ImmutableImageWithMutableProperties
from .shared_resources import GL, UNIQUE_QGRAPHICSITEM_TYPE
from .shader_scene import ShaderItem, ShaderScene
from .shader_view import ShaderView

class ImageScene(ShaderScene):
    def __init__(self, parent, ImageClass, ImageStackClass, ContextualInfoItemClass):
        super().__init__(parent, ContextualInfoItemClass)
        self.ImageClass = ImageClass
        self.image_stack = ImageStackClass()
        self.image_stack.bounding_box_changed.connect(self._on_image_stack_bounding_box_changed)
        self.addItem(self.image_stack)

    def _on_image_stack_bounding_box_changed(self):
        self.setSceneRect(self.image_item.boundingRect())
        for view in self.views():
            view._on_image_stack_bounding_box_changed()

class ImageStack(ShaderItem):
    """The image_objects member variable of an ImageStack instance contains a list of ImmutableImageWithMutableProperties
    instances (or instances of subclasses or compatible alternative implementations of ImmutableImageWithMutableProperties).
    In terms of composition ordering, these are in ascending Z-order, with the positive Z axis pointing out of the screen.

    The blend_function of the first (0th) element of image_objects is ignored, although its getcolor_expression and
    extra_transformation expression, if provided, are used.  In the fragment shader, the result of applying getcolor_expression
    and then extra_transformation expression are saved in the variables da (a float representing alpha channel value) and dca
    (a vec3, which is a vector of three floats, representing the premultiplied RGB channel values).

    Subsequent elements of image_objects are blended into da and dca using the blend_function specified by each image_object.
    When no elements remain to be blended, dca is divided element-wise by da, un-premultiplying it, and these three values and
    da are returned to OpenGL for src-over blending into the view.

    ImageStack's boundingRect has its top left at (0, 0) and has same dimensions as the first (0th) element of image_objects,
    or is 1x1 if image_objects is empty.  Therefore, if the scale of an ImageStack instance containing at least one image
    has not been modified, that ImageStack instance will be the same width and height in scene units as the first element
    of image_objects is in pixel units, making the mapping between scene units and pixel units 1:1 for the image at the bottom
    of the stack."""
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()
    NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE = {
        numpy.bool8  : Qt.QOpenGLTexture.UInt8,
        numpy.uint8  : Qt.QOpenGLTexture.UInt8,
        numpy.uint16 : Qt.QOpenGLTexture.UInt16,
        numpy.float32: Qt.QOpenGLTexture.Float32}
    IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT = {
        'g'   : Qt.QOpenGLTexture.R32F,
        'ga'  : Qt.QOpenGLTexture.RG32F,
        'rgb' : Qt.QOpenGLTexture.RGB32F,
        'rgba': Qt.QOpenGLTexture.RGBA32F}
    IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT = {
        'g'   : Qt.QOpenGLTexture.Red,
        'ga'  : Qt.QOpenGLTexture.RG,
        'rgb' : Qt.QOpenGLTexture.RGB,
        'rgba': Qt.QOpenGLTexture.RGBA}
    _OVERLAY_UNIFORMS_TEMPLATE = Template('\n'.join(
        'uniform sampler2D image${idx}_tex;',
        'uniform mat3 image${idx}_frag_to_tex;',
        'uniform float image${idx}_tex_global_alpha;',
        'uniform float image${idx}_rescale_min;',
        'uniform float image${idx}_rescale_range;',
        'uniform float image${idx}_gamma;'))
    _OVERLAY_BLENDING_TEMPLATE = Template('    ' + '\n    '.join(
        'tex_coord = transform_frag_to_tex(image${idx}_frag_to_tex);',
        'if(tex_coord.x >= 0 && tex_coord.x < 1 && tex_coord.y >= 0 && tex_coord.y < 1)',
        '{',
        '    s = texture2D(image${idx}_tex, tex_coord);',
        '    s = ${getcolor_expression};',
        '    sa = clamp(s.a, 0, 1) * image${idx}_tex_global_alpha;',
        '    sc = min_max_gamma_transform(s.rgb, image${idx}_rescale_min, image${idx}_rescale_range, image${idx}_gamma);',
        '    ${extra_transformation_expression};',
        '    sca = sc * sa;',
        '    ${blend_function}',
        '    da = clamp(da, 0, 1);',
        '    dca = clamp(dca, 0, 1);',
        '}'))

    bounding_box_changed = Qt.pyqtSignal()
    # First parameter of image_* signals is 0-based index into image_objects
    image_inserted = Qt.pyqtSignal(int)
    # Second parameter is image object that was replaced or removed
    image_replaced = Qt.pyqtSignal(int, object)
    image_removed = Qt.pyqtSignal(int, object)

    def __init__(self, parent_item=None):
        self.image_objects = [] # In ascending order, with bottom image (backmost) as element 0
        self._image_object_ids = []
        self._next_image_object_id = -1

    def type(self):
        return ImageItem.QGRAPHICSITEM_TYPE

    def boundingRect(self):
        return Qt.QRectF(Qt.QPointF(), Qt.QRectF(self.image_objects[0].size))

    def append_image(self, image_data, *va, **ka):
        self.insert_image_object(len(self.image_objects), self.ImageClass(image_data, *va, **ka))

    def insert_image(self, idx, image_data, *va, **ka):
        self.insert_image_object(idx, self.ImageClass(image_data, *va, **ka))

    def replace_image(self, idx, image_data, *va, **ka):
        self.replace_image_object(idx, self.ImageClass(image_data, *va, **ka), True)

    def remove_image(self, idx):
        self.remove_image_object(idx)

    def insert_image_object(self, idx, image_object):
        assert idx <= len(self.image_objects)
        if idx == 0:
            self.prepareGeometryChange()
        image_object.property_changed.connect(self.update)
        self.image_objects.insert(idx, image_object)
        self._image_object_ids.insert(idx, self._generate_image_object_id)
        self.image_inserted.emit(idx)
        self.update()

    def replace_image_object(self, idx, image_object, preserve_properties=False):
        if idx == 0:
            self.prepareGeometryChange()
        old_image_object = self.image_objects[idx]
        old_image_object.property_changed.disconnect(self.update)
        self.image_objects[idx] = image_object
        self._image_object_ids[idx] = self._generate_image_object_id()
        image_object.property_changed.connect(self.update)
        try:
            if preserve_properties:
                # Todo: abstract property copying or make image object data mutable so that it can be avoided
                image_object.trilinear_filtering_enabled = old_image_object.trilinear_filtering_enabled
                image_object.gamma = old_image_object.gamma
                if image_object.type == old_image_object.type:
                    image_object.min, image_object.max = old_image_object.min, old_image_object.max
                if old_image_object.auto_getcolor_expression_enabled:
                    image_object.auto_getcolor_expression_enabled = True
                else:
                    image_object.auto_getcolor_expression_enabled = False
                    image_object.getcolor_expression = old_image_object.getcolor_expression
                image_object.extra_transformation_expression = old_image_object.extra_transformation_expression
                image_object.blend_function = old_image_object.blend_function
        finally:
            self.image_replaced.emit(idx, old_image_object)
            self.update()

    def remove_image_object(self, idx):
        if idx == 0:
            self.prepareGeometryChange()
        image_object = self.image_objects[idx]
        image_object.property_changed.disconnect(self.update)
        del self.image_objects[idx]
        del self._image_object_ids[idx]
        self.image_removed.emit(idx, image_object)
        self.update()

    def hoverMoveEvent(self, event):
        pass
#        if self.image_objects:
#            # NB: event.pos() is a QPointF, and one may call QPointF.toPoint(), as in the following line,
#            # to get a QPoint from it.  However, toPoint() rounds x and y coordinates to the nearest int,
#            # which would cause us to erroneously report mouse position as being over the pixel to the
#            # right and/or below if the view with the mouse cursor is zoomed in such that an image pixel
#            # occupies more than one screen pixel and the cursor is over the right and/or bottom half
#            # of a pixel.
##           pos = event.pos().toPoint()
#            pos = Qt.QPoint(event.pos().x(), event.pos().y())
#            cis = []
#            ci = self.generate_contextual_info_for_pos(pos)
#            if ci is not None:
#                cis.append(ci)
#            self._update_overlay_items_z_sort()
#            for overlay_stack_idx, overlay_item in enumerate(self._overlay_items):
#                if overlay_item.isVisible():
#                    # For a number of potential reasons including overlay rotation, differing resolution
#                    # or scale, and fractional offset relative to parent, it is necessary to project floating
#                    # point coordinates and not integer coordinates into overlay item space in order to
#                    # accurately determine which overlay image pixel contains the mouse pointer
#                    o_pos = self.mapToItem(overlay_item, event.pos())
#                    if overlay_item.boundingRect().contains(o_pos):
#                        ci = overlay_item.generate_contextual_info_for_pos(o_pos, overlay_stack_idx)
#                        if ci is not None:
#                            cis.append(ci)
#            self.scene().update_contextual_info('\n'.join(cis), self)
#        else:
#            self.scene().clear_contextual_info(self)

    def paint(self, qpainter, option, widget):
        assert widget is not None, 'image_view.ImageStack.paint called with widget=None.  Ensure that view caching is disabled.'
        qpainter.beginNativePainting()
        with ExitStack() as estack:
            estack.callback(qpainter.endNativePainting)
            self._update_tex(estack)
            if self._tex is not None and self._getcolor_expression is not None:
                gl = GL()
                image = self._image
                self._update_overlay_items_z_sort()
                prog_desc = [self._getcolor_expression, self._extra_transformation_expression]
                visible_overlays = []
                for overlay_item in self._overlay_items:
                    if overlay_item.isVisible() and overlay_item._image is not None:
                        prog_desc += [overlay_item._getcolor_expression, overlay_item._blend_function, overlay_item._extra_transformation_expression]
                        visible_overlays.append(overlay_item)
                prog_desc = tuple(prog_desc)
                if prog_desc in self.progs:
                    prog = self.progs[prog_desc]
                else:
                    overlay_uniforms = ''
                    do_overlay_blending = ''
                    for overlay_idx, overlay_item in enumerate(visible_overlays):
                        overlay_uniforms += self._OVERLAY_UNIFORMS_TEMPLATE.substitute(idx=overlay_idx)
                        do_overlay_blending += self._OVERLAY_BLENDING_TEMPLATE.substitute(
                            idx=overlay_idx,
                            getcolor_expression=overlay_item._getcolor_expression,
                            extra_transformation_expression='' if overlay_item._extra_transformation_expression is None else overlay_item._extra_transformation_expression,
                            blend_function=overlay_item._blend_function_impl)
                    prog = self.build_shader_prog(prog_desc,
                                                  'image_widget_vertex_shader.glsl',
                                                  'image_widget_fragment_shader_template.glsl',
                                                  overlay_uniforms=overlay_uniforms,
                                                  getcolor_expression=self._getcolor_expression,
                                                  extra_transformation_expression='' if self._extra_transformation_expression is None else self._extra_transformation_expression,
                                                  do_overlay_blending=do_overlay_blending)
                prog.bind()
                estack.callback(prog.release)
                view = widget.view
                view.quad_buffer.bind()
                estack.callback(view.quad_buffer.release)
                view.quad_vao.bind()
                estack.callback(view.quad_vao.release)
                vert_coord_loc = prog.attributeLocation('vert_coord')
                prog.enableAttributeArray(vert_coord_loc)
                prog.setAttributeBuffer(vert_coord_loc, gl.GL_FLOAT, 0, 2, 0)
                prog.setUniformValue('tex', 0)
                frag_to_tex = Qt.QTransform()
                frame = Qt.QPolygonF(view.mapFromScene(Qt.QPolygonF(self.sceneTransform().mapToPolygon(self.boundingRect().toRect()))))
                if not qpainter.transform().quadToSquare(frame, frag_to_tex):
                    raise RuntimeError('Failed to compute gl_FragCoord to texture coordinate transformation matrix.')
                prog.setUniformValue('frag_to_tex', frag_to_tex)
                prog.setUniformValue('tex_global_alpha', self.opacity())
                prog.setUniformValue('viewport_height', float(widget.size().height()))

#               print('qpainter.transform():', qtransform_to_numpy(qpainter.transform()))
#               print('self.deviceTransform(view.viewportTransform()):', qtransform_to_numpy(self.deviceTransform(view.viewportTransform())))

                min_max = numpy.array((self._normalized_min, self._normalized_max), dtype=float)
                min_max = self._renormalize_for_gl(min_max)
                prog.setUniformValue('gamma', self.gamma)
                prog.setUniformValue('rescale_min', min_max[0])
                prog.setUniformValue('rescale_range', min_max[1] - min_max[0])
                for overlay_idx, overlay_item in enumerate(visible_overlays):
                    texture_unit = overlay_idx + 1 # +1 because first texture unit is occupied by image texture
                    overlay_item._update_tex(estack, texture_unit) 
                    frag_to_tex = Qt.QTransform()
                    frame = Qt.QPolygonF(view.mapFromScene(Qt.QPolygonF(overlay_item.sceneTransform().mapToPolygon(overlay_item.boundingRect().toRect()))))
                    qpainter_transform = overlay_item.deviceTransform(view.viewportTransform())
                    if not qpainter_transform.quadToSquare(frame, frag_to_tex):
                        raise RuntimeError('Failed to compute gl_FragCoord to texture coordinate transformation matrix for overlay {}.'.format(overlay_idx))
                    prog.setUniformValue('overlay{}_frag_to_tex'.format(overlay_idx), frag_to_tex)
                    prog.setUniformValue('overlay{}_tex'.format(overlay_idx), texture_unit)
                    prog.setUniformValue('overlay{}_tex_global_alpha'.format(overlay_idx), overlay_item.opacity())
                    prog.setUniformValue('overlay{}_gamma'.format(overlay_idx), overlay_item.gamma)
                    min_max[0], min_max[1] = overlay_item._normalized_min, overlay_item._normalized_max
                    min_max = overlay_item._renormalize_for_gl(min_max)
                    prog.setUniformValue('overlay{}_rescale_min'.format(overlay_idx), min_max[0])
                    prog.setUniformValue('overlay{}_rescale_range'.format(overlay_idx), min_max[1] - min_max[0])
                gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
                gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)
        self._paint_frame(qpainter)

    def _generate_image_object_id(self):
        r = self._next_image_object_id
        self._next_image_object_id += 1
        return r

    def _update_tex(self, estack, idx):
        """Meant to be executed between a pair of QPainter.beginNativePainting() QPainter.endNativePainting() calls or,
        at the very least, when an OpenGL context is current, _update_tex does whatever is required for self._tex[idx] to
        represent self._image_objects[idx], including texture object creation and texture data uploading, and it leaves
        self._tex[n] bound to texture unit n.  Additionally, if n >= len(self._image_objects), self._tex[n]
        is destroyed."""
        if self._image is None:
            if self._tex is not None:
                self._tex.destroy()
                self._tex = None
        else:
            tex = self._tex
            image = self._image
            desired_texture_format = self.IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT[image.type]
            even_width = image.size.width() % 2 == 0
            desired_texture_size = Qt.QSize(image.size) if even_width else Qt.QSize(image.size.width()+1, image.size.height())
            desired_minification_filter = Qt.QOpenGLTexture.LinearMipMapLinear if self._trilinear_filtering_enabled else Qt.QOpenGLTexture.Linear
            if tex is not None:
                if image.size != desired_texture_size or tex.format() != desired_texture_format or tex.minificationFilter() != desired_minification_filter:
                    tex.destroy()
                    tex = self._tex = None
            if tex is None:
                tex = Qt.QOpenGLTexture(Qt.QOpenGLTexture.Target2D)
                tex.setFormat(desired_texture_format)
                tex.setWrapMode(Qt.QOpenGLTexture.ClampToEdge)
                if self._trilinear_filtering_enabled:
                    tex.setMipLevels(6)
                    tex.setAutoMipMapGenerationEnabled(True)
                else:
                    tex.setMipLevels(1)
                    tex.setAutoMipMapGenerationEnabled(False)
                tex.setSize(desired_texture_size.width(), desired_texture_size.height(), 1)
                tex.allocateStorage()
                tex.setMinMagFilters(desired_minification_filter, Qt.QOpenGLTexture.Nearest)
                tex.image_id = -1
            tex.bind(texture_unit)
            estack.callback(lambda: tex.release(texture_unit))
            if tex.image_id != self._image_id:
#               import time
#               t0=time.time()
                if even_width:
                    pixel_transfer_opts = Qt.QOpenGLPixelTransferOptions()
                    pixel_transfer_opts.setAlignment(1)
                    tex.setData(self.IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT[image.type],
                                self.NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE[image.dtype],
                                image.data.ctypes.data,
                                pixel_transfer_opts)
                else:
                    gl = GL()
                    NUMPY_DTYPE_TO_GL_PIXEL_TYPE = {
                        numpy.bool8  : gl.GL_UNSIGNED_BYTE,
                        numpy.uint8  : gl.GL_UNSIGNED_BYTE,
                        numpy.uint16 : gl.GL_UNSIGNED_SHORT,
                        numpy.float32: gl.GL_FLOAT}
                    IMAGE_TYPE_TO_GL_TEX_FORMAT = {
                        'g'   : gl.GL_R32F,
                        'ga'  : gl.GL_RG32F,
                        'rgb' : gl.GL_RGB32F,
                        'rgba': gl.GL_RGBA32F}
                    IMAGE_TYPE_TO_GL_SRC_PIX_FORMAT = {
                        'g'   : gl.GL_RED,
                        'ga'  : gl.GL_RG,
                        'rgb' : gl.GL_RGB,
                        'rgba': gl.GL_RGBA}
                    orig_unpack_alignment = gl.glGetIntegerv(gl.GL_UNPACK_ALIGNMENT)
                    if orig_unpack_alignment != 1:
                        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
                        # QPainter font rendering for OpenGL surfaces will become broken if we do not restore GL_UNPACK_ALIGNMENT
                        # to whatever QPainter had it set to (when it prepared the OpenGL context for our use as a result of
                        # qpainter.beginNativePainting()).
                        estack.callback(lambda oua=orig_unpack_alignment: gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, oua))
                    gl.glTexSubImage2D(
                        gl.GL_TEXTURE_2D, 0, 0, 0, image.size.width(), image.size.height(),
                        IMAGE_TYPE_TO_GL_SRC_PIX_FORMAT[image.type],
                        NUMPY_DTYPE_TO_GL_PIXEL_TYPE[image.dtype],
                        memoryview(image.data_T.flatten()))
                    if self._trilinear_filtering_enabled:
                        tex.generateMipMaps(0)
#               t1=time.time()
#               print('tex.setData {}ms / {}fps'.format(1000*(t1-t0), 1/(t1-t0)))
                tex.image_id = self._image_id
                # self._tex is updated here and not before so that any failure preparing tex results in a retry the next time self._tex
                # is needed
                self._tex = tex
