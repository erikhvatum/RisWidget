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
import textwrap
#from ._qt_debug import qtransform_to_numpy
from ..image.image import Image
from ..signaling_list.signaling_list import SignalingList
from ..shared_resources import UNIQUE_QGRAPHICSITEM_TYPE
from .shader_item import ShaderItem

class ImageStackItem(ShaderItem):
    """The image_stack attribute of ImageStackItem is an SignalingList, a container with a list interface, containing a sequence
    of Image instances (or instances of subclasses of Image or some duck-type compatible thing).  In terms of composition ordering,
    these are in ascending Z-order, with the positive Z axis pointing out of the screen.  image_stack should be manipulated via the
    standard list interface, which it implements completely.  So, for example, to place an image at the top of the stack:

    ImageStackItem_instance.image_stack.append(Image(numpy.zeros((400,400,3), dtype=numpy.uint8)))

    SignalingList emits signals when elements are removed, inserted, or replaced.  ImageStackItem responds to these signals
    in order to trigger repainting and otherwise keep its state consistent with that of its image_stack attribute.  Users
    and extenders of ImageStackItem may do so in the same way: by connecting Python functions directly to
    ImageStackItem_instance.image_stack.inserted, ImageStackItem_instance.image_stack.removed, and
    ImageStackItem_instance.image_stack.replaced.  For a concrete example, take a look at ImageStackWidget.

    The blend_function of the first (0th) element of image_stack is ignored, although its getcolor_expression and
    extra_transformation expression, if provided, are used.  In the fragment shader, the result of applying getcolor_expression
    and then extra_transformation expression are saved in the variables da (a float representing alpha channel value) and dca
    (a vec3, which is a vector of three floats, representing the premultiplied RGB channel values).

    Subsequent elements of image_stack are blended into da and dca using the blend_function specified by each Image.
    When no elements remain to be blended, dca is divided element-wise by da, un-premultiplying it, and these three values and
    da are returned to OpenGL for src-over blending into the view.

    ImageStackItem's boundingRect has its top left at (0, 0) and has same dimensions as the first (0th) element of image_stack,
    or is 1x1 if image_stack is empty.  Therefore, if the scale of an ImageStackItem instance containing at least one image
    has not been modified, that ImageStackItem instance will be the same width and height in scene units as the first element
    of image_stack is in pixel units, making the mapping between scene units and pixel units 1:1 for the image at the bottom
    of the stack (ie, image_stack[0])."""
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
    UNIFORM_SECTION_TEMPLATE = Template(textwrap.dedent("""\
        uniform sampler2D tex_${idx};
        uniform float global_alpha_${idx};
        uniform float rescale_min_${idx};
        uniform float rescale_range_${idx};
        uniform float gamma_${idx};"""))
    MAIN_SECTION_TEMPLATE = Template(textwrap.dedent("""\
            // image_stack[${idx}]
            s = texture2D(tex_${idx}, tex_coord);
            s = ${getcolor_expression};
            sa = clamp(s.a, 0, 1) * global_alpha_${idx};
            sc = min_max_gamma_transform(s.rgb, rescale_min_${idx}, rescale_range_${idx}, gamma_${idx});
            ${extra_transformation_expression};
            sca = sc * sa;
        ${blend_function}
            da = clamp(da, 0, 1);
            dca = clamp(dca, 0, 1);
        """))
    DEFAULT_BOUNDING_RECT = Qt.QRectF(Qt.QPointF(), Qt.QSizeF(1, 1))
    TEXTURE_BORDER_COLOR = Qt.QColor(0, 0, 0, 0)

    bounding_rect_changed = Qt.pyqtSignal()

    def __init__(self, parent_item=None):
        super().__init__(parent_item)
        self._bounding_rect = Qt.QRectF(self.DEFAULT_BOUNDING_RECT)
        self.image_stack = SignalingList(parent=self) # In ascending order, with bottom image (backmost) as element 0
        self.image_stack.inserted.connect(self._on_images_inserted)
        self.image_stack.removed.connect(self._on_images_removed)
        self.image_stack.replaced.connect(self._on_images_replaced)
        self._texs = {}
        self._dead_texs = [] # Textures queued for deletion when an OpenGL context is available
        self._image_data_serials = {}
        self._next_data_serial = 0
        self._image_instance_counts = {}

    def type(self):
        return ImageStackItem.QGRAPHICSITEM_TYPE

    def boundingRect(self):
        return self._bounding_rect

    def _attach_images(self, images):
        for image in images:
            instance_count = self._image_instance_counts.get(image, 0) + 1
            assert instance_count > 0
            self._image_instance_counts[image] = instance_count
            if instance_count == 1:
                # Any change, including image data change, may change result of rendering image and therefore requires refresh
                image.changed.connect(self._on_image_changed)
                # Only change to image data invalidates a texture.  Texture uploading is deferred until rendering, and rendering is
                # deferred until the next iteration of the event loop.  When image emits image_changed, it will also emit
                # changed.  In effect, self.update marks the scene as requiring refresh while self._on_image_changed marks the
                # associated texture as requiring refresh.  Both marking operations are fast and may be called redundantly multiple
                # times per frame without significantly impacting performace.
                image.data_changed.connect(self._on_image_data_changed)
                self._image_data_serials[image] = self._generate_data_serial()
                self._texs[image] = None

    def _detach_images(self, images):
        for image in images:
            instance_count = self._image_instance_counts[image] - 1
            assert instance_count >= 0
            if instance_count == 0:
                image.changed.disconnect(self._on_image_changed)
                image.data_changed.disconnect(self._on_image_data_changed)
                del self._image_instance_counts[image]
                del self._image_data_serials[image]
                dead_tex = self._texs[image]
                if dead_tex is not None:
                    self._dead_texs.append(dead_tex)
                del self._texs[image]
            else:
                self._image_instance_counts[image] = instance_count

    def _on_images_inserted(self, idx, images):
        br_change = idx == 0 and (len(self.image_stack) == 1 or self.image_stack[1].size != images[0].size)
        if br_change:
            self.prepareGeometryChange()
            self._bounding_rect = Qt.QRectF(Qt.QPointF(), Qt.QSizeF(images[0].size))
        self._attach_images(images)
        if br_change:
            self.bounding_rect_changed.emit()
        self.update()

    def _on_images_removed(self, idxs, images):
        br_change = idxs[0] == 0 and (not self.image_stack or self.image_stack[0].size != images[0].size)
        if br_change:
            self.prepareGeometryChange()
            if self.image_stack:
                self._bounding_rect = Qt.QRectF(Qt.QPointF(), Qt.QSizeF(self.image_stack[0].size))
            else:
                self._bounding_rect = self.DEFAULT_BOUNDING_RECT
        self._detach_images(images)
        if br_change:
            self.bounding_rect_changed.emit()
        self.update()

    def _on_images_replaced(self, idxs, replaced_images, images):
        br_change = idxs[0] == 0 and replaced_images[0].size != images[0].size
        if br_change:
            self.prepareGeometryChange()
            self._bounding_rect = Qt.QRectF(Qt.QPointF(), Qt.QSizeF(images[0].size))
        self._detach_images(replaced_images)
        self._attach_images(images)
        if br_change:
            self.bounding_rect_changed.emit()
        self.update()

    def _on_image_changed(self, image):
        self.update()

    def _on_image_data_changed(self, image):
        self._image_data_serials[image] = self._generate_data_serial()

    def hoverMoveEvent(self, event):
        non_muted_idxs = [idx for idx, image in enumerate(self.image_stack) if not image.mute_enabled]
        if len(non_muted_idxs) == 0:
            self.scene().clear_contextual_info(self)
            return
        # NB: event.pos() is a QPointF, and one may call QPointF.toPoint(), as in the following line,
        # to get a QPoint from it.  However, toPoint() rounds x and y coordinates to the nearest int,
        # which would cause us to erroneously report mouse position as being over the pixel to the
        # right and/or below if the view with the mouse cursor is zoomed in such that an image pixel
        # occupies more than one screen pixel and the cursor is over the right and/or bottom half
        # of a pixel.
#       pos = event.pos().toPoint()
        fpos = event.pos()
        ipos = Qt.QPoint(event.pos().x(), event.pos().y())
        cis = []
        it = iter((idx, self.image_stack[idx]) for idx in non_muted_idxs)
        idx, image = next(it)
        ci = image.generate_contextual_info_for_pos(ipos.x(), ipos.y(), idx if len(self.image_stack) > 1 else None)
        if ci is not None:
            cis.append(ci)
        image0size = image.size
        for idx, image in it:
                # Because the aspect ratio of subsequent images may differ from the first, fractional
                # offsets must be discarded only after projecting from lowest-image pixel coordinates
                # to current image pixel coordinates.  It is easy to see why in the case of an overlay
                # exactly half the width and height of the base: one base unit is two overlay units,
                # so dropping base unit fractions would cause overlay units to be rounded to the preceeding
                # even number in any case where an overlay coordinate component should be odd.
                ci = image.generate_contextual_info_for_pos(int(fpos.x()*image.size.width()/image0size.width()),
                                                            int(fpos.y()*image.size.height()/image0size.height()),
                                                            idx)
                if ci is not None:
                    cis.append(ci)
        self.scene().update_contextual_info('\n'.join(cis), self)

    def hoverLeaveEvent(self, event):
        self.scene().clear_contextual_info(self)

    def paint(self, qpainter, option, widget):
        assert widget is not None, 'ImageStackItem.paint called with widget=None.  Ensure that view caching is disabled.'
        qpainter.beginNativePainting()
        with ExitStack() as estack:
            estack.callback(qpainter.endNativePainting)
            self._destroy_dead_texs()
            GL = widget.GL
            non_muted_idxs = self._get_nonmuted_idxs_and_update_texs(GL, estack)
            if not non_muted_idxs:
                return
            prog_desc = tuple((image.getcolor_expression,
                               'src' if tex_unit==0 else image.blend_function,
                               image.extra_transformation_expression)
                              for tex_unit, image in ((tex_unit, self.image_stack[idx]) for tex_unit, idx in enumerate(non_muted_idxs)))
            if prog_desc in self.progs:
                prog = self.progs[prog_desc]
            else:
                uniforms, main = zip(*((self.UNIFORM_SECTION_TEMPLATE.substitute(idx=idx),
                                        self.MAIN_SECTION_TEMPLATE.substitute(idx=idx,
                                                                              getcolor_expression=image.getcolor_expression,
                                                                              blend_function=type(image).BLEND_FUNCTIONS['src'] if tex_unit==0 else image.blend_function_impl,
                                                                              extra_transformation_expression='' if image.extra_transformation_expression is None
                                                                                                                 else image.extra_transformation_expression))
                                       for idx, tex_unit, image in ((idx, tex_unit, self.image_stack[idx]) for tex_unit, idx in enumerate(non_muted_idxs))))
                prog = self.build_shader_prog(
                    prog_desc,
                    'planar_quad_vertex_shader.glsl',
                    'image_stack_item_fragment_shader_template.glsl',
                    uniforms='\n'.join(uniforms),
                    main='\n'.join(main))
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
            prog.setUniformValue('viewport_height', float(widget.size().height()))
            prog.setUniformValue('image_stack_item_opacity', self.opacity())
            # The next few lines of code compute frag_to_tex, representing an affine transform in 2D space from pixel coordinates
            # to normalized (unit square) texture coordinates.  That is, matrix multiplication of frag_to_tex and homogenous
            # pixel coordinate vector <x, max_y-y, w> (using max_y-y to invert GL's Y axis which is upside-down, typically
            # with 1 for w) yields <x_t, y_t, w_t>.  In non-homogenous coordinates, that's <x_t/w_t, y_t/w_t>, which is
            # ready to be fed to the GLSL texture2D call.
            # 
            # So, GLSL's Texture2D accepts 0-1 element-wise-normalized coordinates (IE, unit square, not unit circle), and
            # frag_to_tex maps from view pixel coordinates to texture coordinates.  If either element of the resulting coordinate
            # vector is outside the interval [0,1], the associated pixel in the view is outside of ImageStackItem.
            #
            # Frame represents, in screen pixel coordinates with origin at the top left of the view, the virtual extent of
            # the rectangular region containing ImageStackItem.  This rectangle may extend beyond any combination of the view's
            # four edges.
            #
            # Frame is computed from ImageStackItem's boundingRect, which is computed from the dimensions of the lowest
            # image of the image_stack, image_stack[0].  Therefore, it is this lowest image that determines the aspect
            # ratio of the unit square's projection onto the view.  Any subsequent images in the stack use this same projection,
            # with the result that they are stretched to fill the ImageStackItem.
            frag_to_tex = Qt.QTransform()
            frame = Qt.QPolygonF(view.mapFromScene(Qt.QPolygonF(self.sceneTransform().mapToPolygon(self.boundingRect().toRect()))))
            if not qpainter.transform().quadToSquare(frame, frag_to_tex):
                raise RuntimeError('Failed to compute gl_FragCoord to texture coordinate transformation matrix.')
            prog.setUniformValue('frag_to_tex', frag_to_tex)
            min_max = numpy.empty((2,), dtype=float)
            for idx, tex_unit, image in ((idx, tex_unit, self.image_stack[idx]) for tex_unit, idx in enumerate(non_muted_idxs)):
                min_max[0], min_max[1] = image.min, image.max
                min_max = self._normalize_for_gl(min_max, image)
                idxstr = str(idx)
                prog.setUniformValue('tex_'+idxstr, tex_unit)
                prog.setUniformValue('global_alpha_'+idxstr, image.global_alpha)
                prog.setUniformValue('rescale_min_'+idxstr, min_max[0])
                prog.setUniformValue('rescale_range_'+idxstr, min_max[1] - min_max[0])
                prog.setUniformValue('gamma_'+idxstr, image.gamma)
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)

    @staticmethod
    def _normalize_for_gl(v, image):
        """Some things to note:
        * OpenGL normalizes uint16 data uploaded to float32 texture for the full uint16 range.  We store
        our unpacked 12-bit images in uint16 arrays.  Therefore, OpenGL will normalize by dividing by
        65535, even though no 12-bit image will have a component value larger than 4095.
        * float32 data uploaded to float32 texture is not normalized"""
        if image.dtype == numpy.uint16:
            v /= 65535
        elif image.dtype == numpy.uint8:
            v /= 255
        elif image.dtype == numpy.float32:
            pass
        else:
            raise NotImplementedError('OpenGL-compatible normalization for {} missing.'.format(image.dtype))
        return v

    def _generate_data_serial(self):
        r = self._next_data_serial
        self._next_data_serial += 1
        return r

    def _get_nonmuted_idxs_and_update_texs(self, GL, estack):
        """Meant to be executed between a pair of QPainter.beginNativePainting() QPainter.endNativePainting() calls or,
        at the very least, when an OpenGL context is current, _get_nonmuted_idxs_and_update_texs does whatever is required,
        for every non-muted image in self.image_stack, in order that self._texs[image] represents image, including texture
        object creation and texture data uploading, and it leaves self._texs[image] bound to texture unit n, where n is
        the associated non_muted_idx."""
        non_muted_idxs = [idx for idx, image in enumerate(self.image_stack) if not image.mute_enabled]
        for tex_unit, idx in enumerate(non_muted_idxs):
            image = self.image_stack[idx]
            tex = self._texs[image]
            serial = self._image_data_serials[image]
#           even_width = image.size.width() % 2 == 0
            desired_texture_format = self.IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT[image.type]
            desired_texture_size = Qt.QSize(image.size) #if even_width else Qt.QSize(image.size.width()+1, image.size.height())
            desired_minification_filter = Qt.QOpenGLTexture.LinearMipMapLinear if image.trilinear_filtering_enabled else Qt.QOpenGLTexture.Linear
            if tex is not None:
                if Qt.QSize(tex.width(), tex.height()) != desired_texture_size or tex.format() != desired_texture_format or tex.minificationFilter() != desired_minification_filter:
                    tex.destroy()
                    tex = self._texs[image] = None
            if tex is None:
                tex = Qt.QOpenGLTexture(Qt.QOpenGLTexture.Target2D)
                tex.setFormat(desired_texture_format)
                tex.setWrapMode(Qt.QOpenGLTexture.ClampToBorder)
                if sys.platform != 'darwin':
                    # TODO: determine why the following call segfaults on OS X and remove the enclosing if statement
                    tex.setBorderColor(self.TEXTURE_BORDER_COLOR)
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
            tex.bind(tex_unit)
            estack.callback(lambda: tex.release(tex_unit))
            if tex.serial != serial:
#               if even_width:
                pixel_transfer_opts = Qt.QOpenGLPixelTransferOptions()
                pixel_transfer_opts.setAlignment(1)
                tex.setData(self.IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT[image.type],
                            self.NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE[image.dtype],
                            image.data.ctypes.data,
                            pixel_transfer_opts)
#               else:
#                   NUMPY_DTYPE_TO_GL_PIXEL_TYPE = {
#                       numpy.bool8  : GL.GL_UNSIGNED_BYTE,
#                       numpy.uint8  : GL.GL_UNSIGNED_BYTE,
#                       numpy.uint16 : GL.GL_UNSIGNED_SHORT,
#                       numpy.float32: GL.GL_FLOAT}
#                   IMAGE_TYPE_TO_GL_TEX_FORMAT = {
#                       'g'   : GL.GL_R32F,
#                       'ga'  : GL.GL_RG32F,
#                       'rgb' : GL.GL_RGB32F,
#                       'rgba': GL.GL_RGBA32F}
#                   IMAGE_TYPE_TO_GL_SRC_PIX_FORMAT = {
#                       'g'   : GL.GL_RED,
#                       'ga'  : GL.GL_RG,
#                       'rgb' : GL.GL_RGB,
#                       'rgba': GL.GL_RGBA}
#                   orig_unpack_alignment = GL.glGetIntegerv(GL.GL_UNPACK_ALIGNMENT)
#                   if orig_unpack_alignment != 1:
#                       GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
#                       # QPainter font rendering for OpenGL surfaces will become broken if we do not restore GL_UNPACK_ALIGNMENT
#                       # to whatever QPainter had it set to (when it prepared the OpenGL context for our use as a result of
#                       # qpainter.beginNativePainting()).
#                       estack.callback(lambda oua=orig_unpack_alignment: GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, oua))
#                   GL.glTexSubImage2D(
#                       GL.GL_TEXTURE_2D, 0, 0, 0, image.size.width(), image.size.height(),
#                       IMAGE_TYPE_TO_GL_SRC_PIX_FORMAT[image.type],
#                       NUMPY_DTYPE_TO_GL_PIXEL_TYPE[image.dtype],
#                       memoryview(image.data_T.flatten()))
#                   if self._trilinear_filtering_enabled:
#                       tex.generateMipMaps(0)
                tex.serial = serial
                # self._texs[image] is updated here and not before so that any failure preparing tex results in a retry the next time self._texs[image]
                # is needed
                self._texs[image] = tex
        return non_muted_idxs

    def _destroy_dead_texs(self):
        """Meant to be executed between a pair of QPainter.beginNativePainting() QPainter.endNativePainting() calls or,
        at the very least, when an OpenGL context is current."""
        while self._dead_texs:
            dead_tex = self._dead_texs.pop()
            dead_tex.destroy()
