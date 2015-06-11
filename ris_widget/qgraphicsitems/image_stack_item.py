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
from .image import Image
from .shared_resources import UNIQUE_QGRAPHICSITEM_TYPE
from .shader_scene import ShaderItem, BaseScene
from .shader_view import BaseView

class ImageStackItem(ShaderItem):
    """The image_objects member variable of an ImageStackItem instance contains a list of Image
    instances (or instances of subclasses or compatible alternative implementations of Image).
    In terms of composition ordering, these are in ascending Z-order, with the positive Z axis pointing out of the screen.

    The blend_function of the first (0th) element of image_objects is ignored, although its getcolor_expression and
    extra_transformation expression, if provided, are used.  In the fragment shader, the result of applying getcolor_expression
    and then extra_transformation expression are saved in the variables da (a float representing alpha channel value) and dca
    (a vec3, which is a vector of three floats, representing the premultiplied RGB channel values).

    Subsequent elements of image_objects are blended into da and dca using the blend_function specified by each image_object.
    When no elements remain to be blended, dca is divided element-wise by da, un-premultiplying it, and these three values and
    da are returned to OpenGL for src-over blending into the view.

    ImageStackItem's boundingRect has its top left at (0, 0) and has same dimensions as the first (0th) element of image_objects,
    or is 1x1 if image_objects is empty.  Therefore, if the scale of an ImageStackItem instance containing at least one image
    has not been modified, that ImageStackItem instance will be the same width and height in scene units as the first element
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
    _OVERLAY_UNIFORMS_TEMPLATE = Template('\n'.join((
        'uniform sampler2D image${idx}_tex;',
        'uniform mat3 image${idx}_frag_to_tex;',
        'uniform float image${idx}_tex_global_alpha;',
        'uniform float image${idx}_rescale_min;',
        'uniform float image${idx}_rescale_range;',
        'uniform float image${idx}_gamma;')))
    _OVERLAY_BLENDING_TEMPLATE = Template('    ' + '\n    '.join((
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
        '}')))

    bounding_rect_changed = Qt.pyqtSignal()
    # First parameter of image_* signals is 0-based index into image_objects
    image_object_inserted = Qt.pyqtSignal(int)
    # Second parameter is image object that was removed/replaced
    image_object_replaced = Qt.pyqtSignal(int, object)
    image_object_removed = Qt.pyqtSignal(int, object)

    def __init__(self, parent_item=None, ImageClass=Image):
        super().__init__(parent_item)
        self.ImageClass = ImageClass
        self.image_objects = [] # In ascending order, with bottom image (backmost) as element 0
        self._texs = [] # 1:1 correspondance with self.image_objects
        self._dead_texs = [] # Textures queued for deletion when an OpenGL context is available
        self._image_data_serials = {}
        self._next_data_serial = 0
        self._image_data_changed_signal_mapper = Qt.QSignalMapper(self)
        # NB: mapped[Qt.QObject] selects the overload of the mapped signal having a parameter of type QObject.
        # There is also an overload for int, but using that overload would require maintaining an int -> image_object
        # dict or remaking all affected mappings when the image stack is modified.
        # For details regarding signal overloading, see http://pyqt.sourceforge.net/Docs/PyQt5/signals_slots.html
        self._image_data_changed_signal_mapper.mapped[Qt.QObject].connect(self._on_image_data_changed)

    def type(self):
        return ImageStackItem.QGRAPHICSITEM_TYPE

    def boundingRect(self):
        if self.image_objects:
            return Qt.QRectF(Qt.QPointF(), Qt.QSizeF(self.image_objects[0].size))
        else:
            return Qt.QRectF(Qt.QPointF(), Qt.QSizeF(1, 1))

    def append_image_data(self, image_data, *va, **ka):
        self.insert_image_object(len(self.image_objects), self.ImageClass(image_data, *va, **ka))

    def insert_image_data(self, idx, image_data, *va, **ka):
        self.insert_image_object(idx, self.ImageClass(image_data, *va, **ka))

    def replace_image_data(self, idx, image_data, *va, **ka):
        """In the special case where idx == len(self.image_objects), ie when replacing one-beyond-the-last,
        the effect of calling replace_image_data(idx...) is the same as append_image_data(...)."""
        if idx == 0:
            self.prepareGeometryChange()
        if idx == len(self.image_objects):
            if 'keep_name' in ka:
                kka = ka.copy()
                del kka['keep_name']
                ka = kka
            self.append_image_data(image_data, *va, **ka)
        else:
            image_object = self.image_objects[idx]
            image_object.set_data(image_data, *va, **ka)
        if idx == 0:
            self.bounding_rect_changed.emit()

    def remove_image(self, idx):
        self.remove_image_object(idx)

    def append_image_object(self, image_object):
        self.insert_image_object(len(self.image_objects), image_object)

    def insert_image_object(self, idx, image_object):
        assert idx <= len(self.image_objects)
        assert image_object not in self.image_objects
        if idx == 0:
            self.prepareGeometryChange()
        self.image_objects.insert(idx, image_object)
        # Any change, including image data change, may change result of rendering image and therefore requires refresh
        image_object.changed.connect(self._do_update)
        # Only change to image data invalidates a texture.  Texture uploading is deferred until rendering, and rendering is
        # deferred until the next iteration of the event loop.  When image_object emits image_changed, it will also emit
        # changed.  In effect, self.update marks the scene as requiring refresh while self._on_image_changed marks the
        # associated texture as requiring refresh.  Both marking operations are fast and may be called redundantly multiple
        # times per frame without significantly impacting performace.
        self._image_data_changed_signal_mapper.setMapping(image_object, image_object)
        image_object.data_changed.connect(self._image_data_changed_signal_mapper.map)
        self._image_data_serials[image_object] = self._generate_data_serial()
        self._texs.insert(idx, None)
        if idx == 0:
            self.bounding_rect_changed.emit()
        self.image_object_inserted.emit(idx)
        self.update()

    def _do_update(self):
        self.update()

    def replace_image_object(self, idx, image_object):
        """In the special case where idx == len(self.image_objects), ie when replacing one-beyond-the-last,
        the effect of calling replace_image_object(idx...) is the same as append_image_object(...)."""
        if idx == 0:
            self.prepareGeometryChange()
        if idx == len(self.image_objects):
            self.append_image_object(image_object)
        else:
            assert image_object not in self.image_objects
            old_image_object = self.image_objects[idx]
            image_object.changed.connect(self._do_update)
            self._image_data_changed_signal_mapper.setMapping(image_object, image_object)
            image_object.data_changed.connect(self._image_data_changed_signal_mapper.map)
            self._image_data_serials[image_object] = self._generate_data_serial()
            self.image_objects[idx] = image_object
            old_image_object.changed.disconnect(self._do_update)
            old_image_object.data_changed.disconnect(self._image_data_changed_signal_mapper.map)
            self._image_data_changed_signal_mapper.removeMappings(old_image_object)
            self.image_object_replaced.emit(idx, old_image_object)
        self.update()

    def remove_image_object(self, idx):
        if idx == 0:
            self.prepareGeometryChange()
        image_object = self.image_objects[idx]
        image_object.disconnect(self._do_update)
        image_object.data_changed.disconnect(self._image_data_changed_signal_mapper.map)
        self._image_data_changed_signal_mapper.removeMappings(image_object)
        del self.image_objects[idx]
        del self._image_data_serials[image_object]
        dead_tex = self._texs[idx]
        if dead_tex is not None:
            self._dead_texs.append(dead_tex)
        del self._texs[idx]
        if idx == 0:
            self.bounding_rect_changed.emit()
        self.image_object_removed.emit(idx, image_object)
        self.update()

    def _on_image_data_changed(self, image_object):
        self._image_data_serials[image_object] = self._generate_data_serial()

    def hoverMoveEvent(self, event):
        if self.image_objects:
            # NB: event.pos() is a QPointF, and one may call QPointF.toPoint(), as in the following line,
            # to get a QPoint from it.  However, toPoint() rounds x and y coordinates to the nearest int,
            # which would cause us to erroneously report mouse position as being over the pixel to the
            # right and/or below if the view with the mouse cursor is zoomed in such that an image pixel
            # occupies more than one screen pixel and the cursor is over the right and/or bottom half
            # of a pixel.
#           pos = event.pos().toPoint()
            fpos = event.pos()
            ipos = Qt.QPoint(event.pos().x(), event.pos().y())
            cis = []
            if self.opacity() > 0:
                ci = self.image_objects[0].generate_contextual_info_for_pos(pos, ipos.x(), ipos.y(), 0 if len(self.image_objects) > 1 else None)
            if ci is not None:
                cis.append(ci)
            for idx, image_object in enumerate(self.image_objects[1:], 1):
                if image_object.global_alpha > 0:
                    # For a number of potential reasons including overlay rotation, differing resolution
                    # or scale, and fractional offset relative to parent, it is necessary to project floating
                    # point coordinates and not integer coordinates into overlay item space in order to
                    # accurately determine which overlay image pixel contains the mouse pointer
#                   o_pos = self.mapToItem(overlay_item, event.pos())
                    ci = image_object.generate_contextual_info_for_pos(ipos.x(), ipos.y(), idx)
                    if ci is not None:
                        cis.append(ci)
            self.scene().update_contextual_info('\n'.join(cis), self)
        else:
            self.scene().clear_contextual_info(self)

    def paint(self, qpainter, option, widget):
        assert widget is not None, 'general_view.ImageStackItem.paint called with widget=None.  Ensure that view caching is disabled.'
        if not self.image_objects:
            return
        qpainter.beginNativePainting()
        with ExitStack() as estack:
            estack.callback(qpainter.endNativePainting)
            self._destroy_dead_texs()
            GL = widget.GL
            for idx in range(len(self.image_objects)):
                self._update_tex(idx, GL, estack)
            image0 = self.image_objects[0]
            prog_desc = [image0.getcolor_expression, image0.extra_transformation_expression]
            for image in self.image_objects[1:]:
                prog_desc += [image.getcolor_expression, image.blend_function, image.extra_transformation_expression]
            prog_desc = tuple(prog_desc)
            if prog_desc in self.progs:
                prog = self.progs[prog_desc]
            else:
                overlay_uniforms = ''
                do_overlay_blending = ''
                for overlay_idx, overlay_image in enumerate(self.image_objects[1:]):
                    overlay_uniforms += self._OVERLAY_UNIFORMS_TEMPLATE.substitute(idx=overlay_idx)
                    ete = overlay_image.extra_transformation_expression
                    do_overlay_blending += self._OVERLAY_BLENDING_TEMPLATE.substitute(
                        idx=overlay_idx,
                        getcolor_expression=overlay_image.getcolor_expression,
                        extra_transformation_expression='' if ete is None else ete,
                        blend_function=overlay_image.blend_function_impl)
                prog = self.build_shader_prog(prog_desc,
                                              'image_widget_vertex_shader.glsl',
                                              'image_widget_fragment_shader_template.glsl',
                                              overlay_uniforms=overlay_uniforms,
                                              getcolor_expression=image0.getcolor_expression,
                                              extra_transformation_expression='' if image0.extra_transformation_expression is None else image0.extra_transformation_expression,
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
            prog.setAttributeBuffer(vert_coord_loc, GL.GL_FLOAT, 0, 2, 0)
            prog.setUniformValue('tex', 0)
            frag_to_tex = Qt.QTransform()
            frame = Qt.QPolygonF(view.mapFromScene(Qt.QPolygonF(self.sceneTransform().mapToPolygon(self.boundingRect().toRect()))))
            if not qpainter.transform().quadToSquare(frame, frag_to_tex):
                raise RuntimeError('Failed to compute gl_FragCoord to texture coordinate transformation matrix.')
            prog.setUniformValue('frag_to_tex', frag_to_tex)
            prog.setUniformValue('item_opacity', self.opacity())
            prog.setUniformValue('tex_global_alpha', image0.global_alpha)
            prog.setUniformValue('viewport_height', float(widget.size().height()))
#           print('qpainter.transform():', qtransform_to_numpy(qpainter.transform()))
#           print('self.deviceTransform(view.viewportTransform()):', qtransform_to_numpy(self.deviceTransform(view.viewportTransform())))
            min_max = numpy.array((image0.min, image0.max), dtype=float)
            min_max = self._normalize_for_gl(min_max, image0)
            prog.setUniformValue('gamma', image0.gamma)
            prog.setUniformValue('rescale_min', min_max[0])
            prog.setUniformValue('rescale_range', min_max[1] - min_max[0])
            for overlay_idx, overlay_image in enumerate(self.image_objects[1:]):
                texture_unit = overlay_idx + 1 # +1 because first texture unit is occupied by image texture
#                   frag_to_tex = Qt.QTransform()
#                   frame = Qt.QPolygonF(view.mapFromScene(Qt.QPolygonF(overlay_image.sceneTransform().mapToPolygon(overlay_image.boundingRect().toRect()))))
#                   qpainter_transform = overlay_image.deviceTransform(view.viewportTransform())
#                   if not qpainter_transform.quadToSquare(frame, frag_to_tex):
#                       raise RuntimeError('Failed to compute gl_FragCoord to texture coordinate transformation matrix for overlay {}.'.format(overlay_idx))
                prog.setUniformValue('overlay{}_frag_to_tex'.format(overlay_idx), frag_to_tex)
                prog.setUniformValue('overlay{}_tex'.format(overlay_idx), texture_unit)
                prog.setUniformValue('overlay{}_tex_global_alpha'.format(overlay_idx), overlay_image.global_alpha)
                prog.setUniformValue('overlay{}_gamma'.format(overlay_idx), overlay_image.gamma)
                min_max[0], min_max[1] = overlay_image.min, overlay_image.max
                min_max = self._normalize_for_gl(min_max, overlay_image)
                prog.setUniformValue('overlay{}_rescale_min'.format(overlay_idx), min_max[0])
                prog.setUniformValue('overlay{}_rescale_range'.format(overlay_idx), min_max[1] - min_max[0])
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)

    @staticmethod
    def _normalize_for_gl(v, image_object):
        """Some things to note:
        * OpenGL normalizes uint16 data uploaded to float32 texture for the full uint16 range.  We store
        our unpacked 12-bit images in uint16 arrays.  Therefore, OpenGL will normalize by dividing by
        65535, even though no 12-bit image will have a component value larger than 4095.
        * float32 data uploaded to float32 texture is not normalized"""
        if image_object.dtype == numpy.uint16:
            v /= 65535
        elif image_object.dtype == numpy.uint8:
            v /= 255
        elif image_object.dtype == numpy.float32:
            pass
        else:
            raise NotImplementedError('OpenGL-compatible normalization for {} missing.'.format(image_object.dtype))
        return v

    def _generate_data_serial(self):
        r = self._next_data_serial
        self._next_data_serial += 1
        return r

    def _update_tex(self, idx, GL, estack):
        """Meant to be executed between a pair of QPainter.beginNativePainting() QPainter.endNativePainting() calls or,
        at the very least, when an OpenGL context is current, _update_tex does whatever is required for self._tex[idx] to
        represent self._image_objects[idx], including texture object creation and texture data uploading, and it leaves
        self._tex[n] bound to texture unit n."""
        tex = self._texs[idx]
        image = self.image_objects[idx]
        serial = self._image_data_serials[image]
#       even_width = image.size.width() % 2 == 0
        desired_texture_format = self.IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT[image.type]
        desired_texture_size = Qt.QSize(image.size) #if even_width else Qt.QSize(image.size.width()+1, image.size.height())
        desired_minification_filter = Qt.QOpenGLTexture.LinearMipMapLinear if image.trilinear_filtering_enabled else Qt.QOpenGLTexture.Linear
        if tex is not None:
            if Qt.QSize(tex.width(), tex.height()) != desired_texture_size or tex.format() != desired_texture_format or tex.minificationFilter() != desired_minification_filter:
                tex.destroy()
                tex = self._texs[idx] = None
        if tex is None:
            tex = Qt.QOpenGLTexture(Qt.QOpenGLTexture.Target2D)
            tex.setFormat(desired_texture_format)
            tex.setWrapMode(Qt.QOpenGLTexture.ClampToEdge)
            if image.trilinear_filtering_enabled:
                tex.setMipLevels(6)
                tex.setAutoMipMapGenerationEnabled(True)
            else:
                tex.setMipLevels(1)
                tex.setAutoMipMapGenerationEnabled(False)
            tex.setSize(desired_texture_size.width(), desired_texture_size.height(), 1)
            tex.allocateStorage()
            tex.setMinMagFilters(desired_minification_filter, Qt.QOpenGLTexture.Nearest)
            tex.serial = -1
        tex.bind(idx)
        estack.callback(lambda: tex.release(idx))
        if tex.serial != serial:
#           if even_width:
            pixel_transfer_opts = Qt.QOpenGLPixelTransferOptions()
            pixel_transfer_opts.setAlignment(1)
            tex.setData(self.IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT[image.type],
                        self.NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE[image.dtype],
                        image.data.ctypes.data,
                        pixel_transfer_opts)
#           else:
#               NUMPY_DTYPE_TO_GL_PIXEL_TYPE = {
#                   numpy.bool8  : GL.GL_UNSIGNED_BYTE,
#                   numpy.uint8  : GL.GL_UNSIGNED_BYTE,
#                   numpy.uint16 : GL.GL_UNSIGNED_SHORT,
#                   numpy.float32: GL.GL_FLOAT}
#               IMAGE_TYPE_TO_GL_TEX_FORMAT = {
#                   'g'   : GL.GL_R32F,
#                   'ga'  : GL.GL_RG32F,
#                   'rgb' : GL.GL_RGB32F,
#                   'rgba': GL.GL_RGBA32F}
#               IMAGE_TYPE_TO_GL_SRC_PIX_FORMAT = {
#                   'g'   : GL.GL_RED,
#                   'ga'  : GL.GL_RG,
#                   'rgb' : GL.GL_RGB,
#                   'rgba': GL.GL_RGBA}
#               orig_unpack_alignment = GL.glGetIntegerv(GL.GL_UNPACK_ALIGNMENT)
#               if orig_unpack_alignment != 1:
#                   GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
#                   # QPainter font rendering for OpenGL surfaces will become broken if we do not restore GL_UNPACK_ALIGNMENT
#                   # to whatever QPainter had it set to (when it prepared the OpenGL context for our use as a result of
#                   # qpainter.beginNativePainting()).
#                   estack.callback(lambda oua=orig_unpack_alignment: GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, oua))
#               GL.glTexSubImage2D(
#                   GL.GL_TEXTURE_2D, 0, 0, 0, image.size.width(), image.size.height(),
#                   IMAGE_TYPE_TO_GL_SRC_PIX_FORMAT[image.type],
#                   NUMPY_DTYPE_TO_GL_PIXEL_TYPE[image.dtype],
#                   memoryview(image.data_T.flatten()))
#               if self._trilinear_filtering_enabled:
#                   tex.generateMipMaps(0)
            tex.serial = serial
            # self._tex[idx] is updated here and not before so that any failure preparing tex results in a retry the next time self._tex[idx]
            # is needed
            self._texs[idx] = tex

    def _destroy_dead_texs(self):
        """Meant to be executed between a pair of QPainter.beginNativePainting() QPainter.endNativePainting() calls or,
        at the very least, when an OpenGL context is current."""
        while self._dead_texs:
            dead_tex = self._dead_texs.pop()
            dead_tex.destroy()
