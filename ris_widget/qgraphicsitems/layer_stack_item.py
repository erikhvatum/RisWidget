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
from PyQt5 import Qt
from string import Template
import textwrap
from ..contextual_info import ContextualInfo
from ..shared_resources import QGL, UNIQUE_QGRAPHICSITEM_TYPE
from .shader_item import ShaderItem

class LayerStackItem(ShaderItem):
    """The layer_stack attribute of LayerStackItem is an SignalingList, a container with a list interface, containing a sequence
    of Layer instances (or instances of subclasses of Layer or some duck-type compatible thing).  In terms of composition ordering,
    these are in ascending Z-order, with the positive Z axis pointing out of the screen.  layer_stack should be manipulated via the
    standard list interface, which it implements completely.  So, for example, to place an layer at the top of the stack:

    LayerStackItem_instance.layer_stack.append(Layer(Image(numpy.zeros((400,400,3), dtype=numpy.uint8))))

    SignalingList emits signals when elements are removed, inserted, or replaced.  LayerStackItem responds to these signals
    in order to trigger repainting and otherwise keep its state consistent with that of its layer_stack attribute.  Users
    and extenders of LayerStackItem may do so in the same way: by connecting Python functions directly to
    LayerStackItem_instance.layer_stack.inserted, LayerStackItem_instance.layer_stack.removed, and
    LayerStackItem_instance.layer_stack.replaced.  For a concrete example, take a look at ImageStackWidget.

    The blend_function of the first (0th) element of layer_stack is ignored, although its getcolor_expression and
    extra_transformation expression, if provided, are used.  In the fragment shader, the result of applying getcolor_expression
    and then extra_transformation expression are saved in the variables da (a float representing alpha channel value) and dca
    (a vec3, which is a vector of three floats, representing the premultiplied RGB channel values).

    Subsequent elements of layer_stack are blended into da and dca using the blend_function specified by each Image.
    When no elements remain to be blended, dca is divided element-wise by da, un-premultiplying it, and these three values and
    da are returned to OpenGL for src-over blending into the view.

    LayerStackItem's boundingRect has its top left at (0, 0) and has same dimensions as the first (0th) element of layer_stack,
    or is 1x1 if layer_stack is empty.  Therefore, if the scale of an LayerStackItem instance containing at least one layer
    has not been modified, that LayerStackItem instance will be the same width and height in scene units as the first element
    of layer_stack is in pixel units, making the mapping between scene units and pixel units 1:1 for the layer at the bottom
    of the stack (ie, layer_stack[0])."""
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()
    UNIFORM_SECTION_TEMPLATE = Template(textwrap.dedent("""\
        uniform sampler2D tex_${tidx};
        uniform float rescale_min_${tidx};
        uniform float rescale_range_${tidx};
        uniform float gamma_${tidx};
        uniform vec4 tint_${tidx};"""))
    COLOR_TRANSFORM_PROCEDURE_TEMPLATE = Template(textwrap.dedent("""\
        vec4 color_transform_${tidx}(vec4 in_, vec4 tint, float rescale_min, float rescale_range, float gamma_scalar)
        {
            vec4 out_;
            out_.a = in_.a;
            vec3 gamma = vec3(gamma_scalar, gamma_scalar, gamma_scalar);
            ${transform_section}
            return clamp(out_, 0, 1);
        }"""))
    MAIN_SECTION_TEMPLATE = Template(textwrap.dedent("""\
            // layer_stack[${idx}]
            s = texture2D(tex_${tidx}, tex_coord);
            s = color_transform_${tidx}(${getcolor_expression}, tint_${tidx}, rescale_min_${tidx}, rescale_range_${tidx}, gamma_${tidx});
            sca = s.rgb * s.a;
        ${blend_function}
            da = clamp(da, 0, 1);
            dca = clamp(dca, 0, 1);
        """))
    DEFAULT_BOUNDING_RECT = Qt.QRectF(Qt.QPointF(), Qt.QSizeF(1, 1))
    TEXTURE_BORDER_COLOR = Qt.QColor(0, 0, 0, 0)

    bounding_rect_changed = Qt.pyqtSignal()
    painted = Qt.pyqtSignal()

    def __init__(self, layer_stack, parent_item=None):
        super().__init__(parent_item)
        self._bounding_rect = Qt.QRectF(self.DEFAULT_BOUNDING_RECT)
        self.contextual_info = ContextualInfo(self)
        self.layer_stack = layer_stack
        layer_stack.layers_replaced.connect(self._on_layerlist_replaced)
        layer_stack.layer_focus_changed.connect(self._on_layer_focus_changed)
        layer_stack.examine_layer_mode_action.toggled.connect(self.update)
        self._layer_instance_counts = {}

    def __del__(self):
        scene = self.scene()
        if scene is None:
            return
        views = scene.views()
        if not views:
            return
        view = views[0]
        gl_widget = view.gl_widget
        context = gl_widget.context()
        if not context:
            return
        gl_widget.makeCurrent()
        try:
            self._dead_texs.extend(self._texs.values())
            self._destroy_dead_texs()
        finally:
            gl_widget.doneCurrent()

    def type(self):
        return LayerStackItem.QGRAPHICSITEM_TYPE

    def boundingRect(self):
        return self._bounding_rect

    def _on_layerlist_replaced(self, layer_stack, old_layers, layers):
        old_sz = None
        if old_layers is not None:
            if old_layers and old_layers[0].image is not None:
                old_sz = old_layers[0].image.size
            self._detach_layers(old_layers)
            old_layers.inserted.disconnect(self._on_layers_inserted)
            old_layers.removed.disconnect(self._on_layers_removed)
            old_layers.replaced.disconnect(self._on_layers_replaced)
        new_sz = None
        if layers is not None:
            if layers and layers[0].image is not None:
                new_sz = layers[0].image.size
            layers.inserted.connect(self._on_layers_inserted)
            layers.removed.connect(self._on_layers_removed)
            layers.replaced.connect(self._on_layers_replaced)
            self._attach_layers(layers)
        if new_sz != old_sz:
            self.prepareGeometryChange()
            self._bounding_rect = self.DEFAULT_BOUNDING_RECT if new_sz is None else Qt.QRectF(Qt.QPointF(), Qt.QSizeF(new_sz))
            self.bounding_rect_changed.emit()
        self.update()

    def _attach_layers(self, layers):
        layer_stack = self.layer_stack
        examine_layer_mode_enabled = layer_stack.examine_layer_mode_enabled
        if examine_layer_mode_enabled:
            focused_layer = layer_stack.focused_layer
        layer_instance_counts = self._layer_instance_counts
        for layer in layers:
            instance_count = layer_instance_counts.get(layer, 0) + 1
            assert instance_count > 0
            layer_instance_counts[layer] = instance_count
            if instance_count == 1:
                # Initiate background texture upload if layer has a visible image; a no-op if image's texture is already uploaded or
                # queued for uploading.
                image = layer.image
                if image is not None and (examine_layer_mode_enabled and layer is focused_layer or not examine_layer_mode_enabled and layer.visible):
                    image.async_texture.upload()
                # Any change, including layer data change, may change result of rendering layer and therefore requires refresh
                layer.changed.connect(self._on_layer_changed)
                # When layer emits layer_changed, it will also emit changed.  In effect, self.update marks the scene as requiring
                # refresh while self._on_layer_changed initiates a texture upload in the background.  Both operations are fast
                # and may be called redundantly multiple times per frame without significantly impacting performance.
                layer.image_changed.connect(self._on_layer_image_changed)

    def _detach_layers(self, layers):
        for layer in layers:
            instance_count = self._layer_instance_counts[layer] - 1
            assert instance_count >= 0
            if instance_count == 0:
                layer.changed.disconnect(self._on_layer_changed)
                layer.image_changed.disconnect(self._on_layer_image_changed)
            else:
                self._layer_instance_counts[layer] = instance_count

    def _on_layers_inserted(self, idx, layers):
        br_change = False
        if idx == 0:
            layer_stack = self.layer_stack
            nbi = layer_stack.layers[0].image
            nbi_nN = nbi is not None
            if len(layer_stack.layers) == len(layers):
                if nbi_nN:
                    br_change = True
            else:
                obi = layer_stack.layers[1].image
                obi_nN = obi is not None
                if nbi_nN != obi_nN or (nbi_nN and nbi.size != obi.size):
                    br_change = True
        if br_change:
            self.prepareGeometryChange()
            self._bounding_rect = Qt.QRectF(Qt.QPointF(), Qt.QSizeF(nbi.size)) if nbi_nN else self.DEFAULT_BOUNDING_RECT
        self._attach_layers(layers)
        if br_change:
            self.bounding_rect_changed.emit()
        self.update()
        self._update_contextual_info()

    def _on_layers_removed(self, idxs, layers):
        assert all(idx1 > idx0 for idx0, idx1 in zip(idxs, idxs[1:])), "Implementation of _on_layers_removed relies on idxs being in ascending order"
        br_change = False
        if idxs[0] == 0:
            layer_stack = self.layer_stack
            obi = layers[0].image
            obi_nN = obi is not None
            if not layer_stack.layers:
                if obi_nN:
                    br_changed = True
            else:
                nbi = layer_stack.layers[0].image
                nbi_nN = nbi is not None
                if nbi_nN != obi_nN or (nbi_nN and nbi.size != obi.size):
                    br_change = True
        if br_change:
            self.prepareGeometryChange()
            self._bounding_rect = self.DEFAULT_BOUNDING_RECT if not layer_stack.layers or not nbi_nN else Qt.QRectF(Qt.QPointF(), Qt.QSizeF(nbi.size))
        self._detach_layers(layers)
        if br_change:
            self.bounding_rect_changed.emit()
        self.update()
        self._update_contextual_info()

    def _on_layers_replaced(self, idxs, replaced_layers, layers):
        assert all(idx1 > idx0 for idx0, idx1 in zip(idxs, idxs[1:])), "Implementation of _on_layers_replaced relies on idxs being in ascending order"
        br_change = False
        if idxs[0] == 0:
            obi = replaced_layers[0].image
            obi_nN = obi is not None
            nbi = layers[0].image
            nbi_nN = nbi is not None
            if nbi_nN != obi_nN or (nbi_nN and nbi.size != obi.size):
                br_change = True
        if br_change:
            self.prepareGeometryChange()
            self._bounding_rect = Qt.QRectF(Qt.QPointF(), Qt.QSizeF(nbi.size)) if nbi_nN else self.DEFAULT_BOUNDING_RECT
        self._detach_layers(replaced_layers)
        self._attach_layers(layers)
        if br_change:
            self.bounding_rect_changed.emit()
        self.update()
        self._update_contextual_info()

    def _on_layer_changed(self, layer):
        self.update()

    def _on_layer_image_changed(self, layer):
        idx = self.layer_stack.layers.index(layer)
        if idx == 0:
            image = layer.image
            current_br = self.boundingRect()
            if image is None:
                new_br = self.DEFAULT_BOUNDING_RECT
            else:
                new_br = Qt.QRectF(Qt.QPointF(), Qt.QSizeF(image.size))
                examine_layer_mode_enabled = self.layer_stack.examine_layer_mode_enabled
                if examine_layer_mode_enabled and layer is self.layer_stack.focused_layer or not examine_layer_mode_enabled and layer.visible:
                    image.async_texture.upload()
            if new_br != current_br:
                self.prepareGeometryChange()
                self._bounding_rect = new_br
                self.bounding_rect_changed.emit()
        self._update_contextual_info()

    def _on_layer_focus_changed(self, old_layer, layer):
        # The appearence of a layer_stack_item may depend on which layer table row is current while
        # "examine layer mode" is enabled.
        if self.layer_stack.examine_layer_mode_enabled:
            self.update()

    def hoverMoveEvent(self, event):
        # NB: contextual info overlay will only be correct for the first view containing this item.
        self.contextual_info.pos = self.scene().views()[0].mapFromScene(event.scenePos())
        self.scene().contextual_info_item.set_contextual_info(self.contextual_info)
        self._update_contextual_info()

    def hoverLeaveEvent(self, event):
        self.contextual_info.pos = None
        self.scene().contextual_info_item.clear_contextual_info(self)

    def _update_contextual_info(self):
        if self.layer_stack.examine_layer_mode_enabled:
            idx = self.layer_stack.focused_layer_idx
            visible_idxs = [] if idx is None else [idx]
        elif self.layer_stack.layers:
            visible_idxs = [idx for idx, layer in enumerate(self.layer_stack.layers) if layer.visible]
        else:
            visible_idxs = []
        if not visible_idxs or self.contextual_info.pos is None or self.scene() is None or not self.scene().views():
            self.contextual_info.value = ''
            return
        fpos = self.scene().views()[0].mapToScene(self.contextual_info.pos)
        # NB: fpos is a QPointF, and one may call QPointF.toPoint(), as in the following line,
        # to get a QPoint from it.  However, toPoint() rounds x and y coordinates to the nearest int,
        # which would cause us to erroneously report mouse position as being over the pixel to the
        # right and/or below if the view with the mouse cursor is zoomed in such that an layer pixel
        # occupies more than one screen pixel and the cursor is over the right and/or bottom half
        # of a pixel.
        # ipos = fpos.toPoint()
        ipos = Qt.QPoint(fpos.x(), fpos.y())
        cis = []
        it = iter((idx, self.layer_stack.layers[idx]) for idx in visible_idxs)
        idx, layer = next(it)
        ci = layer.generate_contextual_info_for_pos(
            ipos.x(),
            ipos.y(),
            idx if len(self.layer_stack.layers) > 1 else None,
            self.layer_stack.layer_name_in_contextual_info_enabled,
            self.layer_stack.image_name_in_contextual_info_enabled)
        if ci is not None:
            cis.append(ci)
        image = layer.image
        image0size = self.DEFAULT_BOUNDING_RECT.size() if image is None else image.size
        for idx, layer in it:
            # Because the aspect ratio of subsequent layers may differ from the first, fractional
            # offsets must be discarded only after projecting from lowest-layer pixel coordinates
            # to current layer pixel coordinates.  It is easy to see why in the case of an overlay
            # exactly half the width and height of the base: one base unit is two overlay units,
            # so dropping base unit fractions would cause overlay units to be rounded to the preceding
            # even number in any case where an overlay coordinate component should be odd.
            image = layer.image
            if image is None:
                ci = layer.generate_contextual_info_for_pos(
                    None, None, idx,
                    self.layer_stack.layer_name_in_contextual_info_enabled,
                    self.layer_stack.image_name_in_contextual_info_enabled)
            else:
                imagesize = image.size
                ci = layer.generate_contextual_info_for_pos(
                    int(fpos.x() * imagesize.width() / image0size.width()),
                    int(fpos.y() * imagesize.height() / image0size.height()),
                    idx,
                    self.layer_stack.layer_name_in_contextual_info_enabled,
                    self.layer_stack.image_name_in_contextual_info_enabled)
            if ci is not None:
                cis.append(ci)
        self.contextual_info.value = '\n'.join(reversed(cis))

    def paint(self, qpainter, option, widget):
        qpainter.beginNativePainting()
        with ExitStack() as estack:
            estack.callback(qpainter.endNativePainting)
            visible_idxs = self._get_visible_idxs_and_update_texs(estack)
            if not visible_idxs:
                return
            prog_desc = tuple((layer.getcolor_expression,
                               'src' if tidx==0 else layer.blend_function,
                               layer.transform_section)
                              for tidx, layer in ((tidx, self.layer_stack.layers[idx]) for tidx, idx in enumerate(visible_idxs)))
            if prog_desc in self.progs:
                prog = self.progs[prog_desc]
            else:
                uniforms, color_transform_procedures, main = \
                    zip(*(
                            (
                                self.UNIFORM_SECTION_TEMPLATE.substitute(tidx=tidx),
                                self.COLOR_TRANSFORM_PROCEDURE_TEMPLATE.substitute(
                                    tidx=tidx,
                                    transform_section=layer.transform_section),
                                self.MAIN_SECTION_TEMPLATE.substitute(
                                    idx=idx,
                                    tidx=tidx,
                                    getcolor_expression=layer.getcolor_expression,
                                    blend_function=layer.BLEND_FUNCTIONS['src' if tidx==0 else layer.blend_function])
                            ) for idx, tidx, layer in
                                (
                                    (idx, tidx, self.layer_stack.layers[idx]) for tidx, idx in enumerate(visible_idxs)
                                )
                       ) )
                prog = self.build_shader_prog(
                    prog_desc,
                    'planar_quad_vertex_shader.glsl',
                    'layer_stack_item_fragment_shader_template.glsl',
                    uniforms='\n'.join(uniforms),
                    color_transform_procedures='\n'.join(color_transform_procedures),
                    main='\n'.join(main))
            prog.bind()
            estack.callback(prog.release)
            if widget is None:
                # We are being called as a result of a BaseView.snapshot(..) invocation
                widget = self.scene().views()[0].gl_widget
            view = widget.view
            view.quad_buffer.bind()
            estack.callback(view.quad_buffer.release)
            view.quad_vao.bind()
            estack.callback(view.quad_vao.release)
            vert_coord_loc = prog.attributeLocation('vert_coord')
            prog.enableAttributeArray(vert_coord_loc)
            GL = QGL()
            prog.setAttributeBuffer(vert_coord_loc, GL.GL_FLOAT, 0, 2, 0)
            prog.setUniformValue('viewport_height', GL.glGetFloatv(GL.GL_VIEWPORT)[3])
            prog.setUniformValue('layer_stack_item_opacity', self.opacity())
            # The next few lines of code compute frag_to_tex, representing an affine transform in 2D space from pixel coordinates
            # to normalized (unit square) texture coordinates.  That is, matrix multiplication of frag_to_tex and homogenous
            # pixel coordinate vector <x, max_y-y, w> (using max_y-y to invert GL's Y axis which is upside-down, typically
            # with 1 for w) yields <x_t, y_t, w_t>.  In non-homogenous coordinates, that's <x_t/w_t, y_t/w_t>, which is
            # ready to be fed to the GLSL texture2D call.
            #
            # So, GLSL's Texture2D accepts 0-1 element-wise-normalized coordinates (IE, unit square, not unit circle), and
            # frag_to_tex maps from view pixel coordinates to texture coordinates.  If either element of the resulting coordinate
            # vector is outside the interval [0,1], the associated pixel in the view is outside of LayerStackItem.
            #
            # Frame represents, in screen pixel coordinates with origin at the top left of the view, the virtual extent of
            # the rectangular region containing LayerStackItem.  This rectangle may extend beyond any combination of the view's
            # four edges.
            #
            # Frame is computed from LayerStackItem's boundingRect, which is computed from the dimensions of the lowest
            # layer of the layer_stack, layer_stack[0].  Therefore, it is this lowest layer that determines the aspect
            # ratio of the unit square's projection onto the view.  Any subsequent layers in the stack use this same projection,
            # with the result that they are stretched to fill the LayerStackItem.
            frag_to_tex = Qt.QTransform()
            frame = Qt.QPolygonF(view.mapFromScene(Qt.QPolygonF(self.sceneTransform().mapToPolygon(self.boundingRect().toRect()))))
            dpi_ratio = widget.devicePixelRatio()
            if dpi_ratio != 1:
                dpi_transform = Qt.QTransform()
                dpi_transform.scale(dpi_ratio, dpi_ratio)
                frame = dpi_transform.map(frame)
            if not qpainter.transform().quadToSquare(frame, frag_to_tex):
                raise RuntimeError('Failed to compute gl_FragCoord to texture coordinate transformation matrix.')
            prog.setUniformValue('frag_to_tex', frag_to_tex)
            min_max = numpy.empty((2,), dtype=float)
            for tidx, idx in enumerate(visible_idxs):
                layer = self.layer_stack.layers[idx]
                image = layer.image
                min_max[0], min_max[1] = layer.min, layer.max
                min_max = self._normalize_for_gl(min_max, image)
                tidxstr = str(tidx)
                prog.setUniformValue('tex_'+tidxstr, tidx)
                prog.setUniformValue('rescale_min_'+tidxstr, min_max[0])
                prog.setUniformValue('rescale_range_'+tidxstr, min_max[1] - min_max[0])
                prog.setUniformValue('gamma_'+tidxstr, layer.gamma)
                prog.setUniformValue('tint_'+tidxstr, Qt.QVector4D(*layer.tint))
            self.set_blend(estack)
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)
        self.painted.emit()

    @staticmethod
    def _normalize_for_gl(v, image):
        """Some things to note:
        * OpenGL normalizes uint16 data uploaded to float32 texture for the full uint16 range.  We store
        our unpacked 12-bit images in uint16 arrays.  Therefore, OpenGL will normalize by dividing by
        65535, even though no 12-bit image will have a component value larger than 4095.
        * float32 data uploaded to float32 texture is not normalized"""
        if image.dtype == numpy.uint16:
            v /= 65535
        elif image.dtype == numpy.uint8 or image.dtype == bool:
            v /= 255
        elif image.dtype == numpy.float32:
            pass
        else:
            raise NotImplementedError('OpenGL-compatible normalization for {} missing.'.format(image.dtype))
        return v

    def _get_visible_idxs_and_update_texs(self, estack):
        """Meant to be executed between a pair of QPainter.beginNativePainting() QPainter.endNativePainting() calls or,
        at the very least, when an OpenGL context is current, _get_nonmuted_idxs_and_update_texs does whatever is required,
        for every visible layer with non-None .layer in self.layer_stack, in order that self._texs[layer] represents layer, including texture
        object creation and texture data uploading, and it leaves self._texs[layer] bound to texture unit n, where n is
        the associated visible_idx."""
        layer_stack = self.layer_stack
        if layer_stack.examine_layer_mode_enabled:
            idx = layer_stack.focused_layer_idx
            visible_idxs = [] if idx is None or layer_stack[idx].image is None else [idx]
        elif layer_stack.layers:
            visible_idxs = [idx for idx, layer in enumerate(layer_stack.layers) if layer.visible and layer.image is not None]
        else:
            visible_idxs = []
        for tex_unit, idx in enumerate(visible_idxs):
            layer = layer_stack.layers[idx]
            image = layer.image
            image.async_texture.bind(tex_unit, estack)
        return visible_idxs