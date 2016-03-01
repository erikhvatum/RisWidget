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
from .shader_item import ShaderItem, ShaderTexture
from ..shared_resources import QGL, UNIQUE_QGRAPHICSITEM_TYPE

class HistogramItem(ShaderItem):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()

    def __init__(self, layer_stack, graphics_item_parent=None):
        super().__init__(graphics_item_parent)
        self.layer_stack = layer_stack
        layer_stack.layer_focus_changed.connect(self._on_layer_focus_changed)
        self.layer = None
        self._layer_data_serial = 0
        self._bounding_rect = Qt.QRectF(0, 0, 1, 1)
        self._tex = None
        self._gl_widget = None
        self.min_item = MinMaxItem(self, 'min')
        self.max_item = MinMaxItem(self, 'max')
        self.gamma_item = GammaItem(self, self.min_item, self.max_item)
        self.gamma_gamma = 1.0
        self.hide()

    def _do_update(self):
        self.update()

    def _on_layer_focus_changed(self, layer_stack, old_layer, layer):
        assert layer_stack is self.layer_stack
        if old_layer is not None:
            self.layer.image_changed.disconnect(self._on_layer_image_changed)
            self.layer.min_changed.disconnect(self.min_item.arrow_item._on_value_changed)
            self.layer.max_changed.disconnect(self.max_item.arrow_item._on_value_changed)
            self.layer.histogram_min_changed.disconnect(self.min_item.arrow_item._on_value_changed)
            self.layer.histogram_min_changed.disconnect(self._do_update)
            self.layer.histogram_max_changed.disconnect(self.max_item.arrow_item._on_value_changed)
            self.layer.histogram_max_changed.disconnect(self._do_update)
            self.layer.gamma_changed.disconnect(self.gamma_item._on_value_changed)
            self.layer = None
        if layer is None:
            self.hide()
        else:
            layer.image_changed.connect(self._on_layer_image_changed)
            layer.min_changed.connect(self.min_item.arrow_item._on_value_changed)
            layer.max_changed.connect(self.max_item.arrow_item._on_value_changed)
            layer.histogram_min_changed.connect(self.min_item.arrow_item._on_value_changed)
            layer.histogram_min_changed.connect(self._do_update)
            layer.histogram_max_changed.connect(self.max_item.arrow_item._on_value_changed)
            layer.histogram_max_changed.connect(self._do_update)
            layer.gamma_changed.connect(self.gamma_item._on_value_changed)
            self.layer = layer
            self.show()
            self.gamma_item._on_value_changed()
            self._on_layer_image_changed()

    def type(self):
        return HistogramItem.QGRAPHICSITEM_TYPE

    def boundingRect(self):
        return self._bounding_rect

    def paint(self, qpainter, option, widget):
        assert widget is not None, 'histogram_scene.HistogramItem.paint called with widget=None.  Ensure that view caching is disabled.'
        if self._gl_widget is None:
            self._gl_widget = widget
            widget.context_about_to_change.connect(self._on_gl_widget_context_about_to_change, Qt.Qt.DirectConnection)
        else:
            assert self._gl_widget is widget
        layer = self.layer
        if layer is None or layer.image is None:
            if self._tex is not None:
                self._tex.destroy()
                self._tex = None
        else:
            image = layer.image
            view = widget.view
            layer = self.layer
            scene = self.scene()
            with ExitStack() as estack:
                qpainter.beginNativePainting()
                estack.callback(qpainter.endNativePainting)
                GL = QGL()
                desired_shader_type = 'G'
                if desired_shader_type in self.progs:
                    prog = self.progs[desired_shader_type]
                    if not GL.glIsProgram(prog.programId()):
                        # The current GL context is in a state of flux, likely because a histogram view is in a dock widget that is in
                        # the process of being floated or docked.
                        return
                else:
                    prog = self.build_shader_prog(desired_shader_type,
                                                  'planar_quad_vertex_shader.glsl',
                                                  'histogram_item_fragment_shader.glsl')
                desired_tex_width = image.histogram.shape[-1]
                tex = self._tex
                if tex is not None:
                    if tex.width != desired_tex_width:
                        tex.destroy()
                        tex = self._tex = None
                if tex is None:
                    tex = ShaderTexture(GL.GL_TEXTURE_1D)
                    tex.bind()
                    estack.callback(tex.release)
                    GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                    GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                    # tex stores histogram bin counts - values that are intended to be addressed by element without
                    # interpolation.  Thus, nearest neighbor for texture filtering.
                    GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
                    GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
                    tex.serial = -1
                else:
                    tex.bind()
                    estack.callback(tex.release)
                histogram = image.histogram
                h_r = layer.histogram_min, layer.histogram_max
                h_w = h_r[1] - h_r[0]
                r = image.range
                w = r[1] - r[0]
                bin_count = histogram.shape[-1] * h_w / w
                if image.dtype is not numpy.float32:
                    bin_count = int(bin_count)
                bin_idx_offset = int((h_r[0] - r[0])/(w/bin_count))
                print(bin_count, bin_idx_offset)
                if image.num_channels == 1:
                    pass
                elif image.num_channels == 2:
                    histogram = histogram[0,:]
                elif image.num_channels >= 3:
                    histogram = 0.2126 * histogram[0,:] + 0.7152 * histogram[1,:] + 0.0722 * histogram[2,:]
                max_bin_val = histogram[bin_idx_offset:bin_idx_offset + bin_count].max()
                if tex.serial != self._layer_data_serial:
                    orig_unpack_alignment = GL.glGetIntegerv(GL.GL_UNPACK_ALIGNMENT)
                    if orig_unpack_alignment != 1:
                        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
                        # QPainter font rendering for OpenGL surfaces will become broken if we do not restore GL_UNPACK_ALIGNMENT
                        # to whatever QPainter had it set to (when it prepared the OpenGL context for our use as a result of
                        # qpainter.beginNativePainting()).
                        estack.callback(lambda oua=orig_unpack_alignment: GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, oua))
                    # TODO: zero copy upload.
                    GL.glTexImage1D(GL.GL_TEXTURE_1D, 0,
                                    GL.GL_LUMINANCE32UI_EXT, desired_tex_width, 0,
                                    GL.GL_LUMINANCE_INTEGER_EXT, GL.GL_UNSIGNED_INT,
                                    list(histogram))
                    tex.serial = self._layer_data_serial
                    tex.width = desired_tex_width
                    self._tex = tex
                if not view.quad_buffer.bind():
                    Qt.qDebug('view.quad_buffer.bind() failed')
                    return
                estack.callback(view.quad_buffer.release)
                if view.quad_vao is None:
                    pass
                view.quad_vao.bind()
                estack.callback(view.quad_vao.release)
                if not prog.bind():
                    Qt.qDebug('prog.bind() failed')
                    return
                estack.callback(prog.release)
                vert_coord_loc = prog.attributeLocation('vert_coord')
                if vert_coord_loc < 0:
                    Qt.qDebug('vert_coord_loc < 0')
                    return
                prog.enableAttributeArray(vert_coord_loc)
                prog.setAttributeBuffer(vert_coord_loc, GL.GL_FLOAT, 0, 2, 0)
                prog.setUniformValue('tex', 0)
                dpi_ratio = widget.devicePixelRatio()
                prog.setUniformValue('inv_view_size', 1/(dpi_ratio * widget.size().width()), 1/(dpi_ratio * widget.size().height()))
                prog.setUniformValue('x_offset', (h_r[0] - r[0]) / w)
                prog.setUniformValue('x_factor', h_w / w)
                inv_max_transformed_bin_val = max_bin_val**-self.gamma_gamma
                prog.setUniformValue('inv_max_transformed_bin_val', inv_max_transformed_bin_val)
                prog.setUniformValue('gamma_gamma', self.gamma_gamma)
                prog.setUniformValue('opacity', self.opacity())
                self.set_blend(estack)
                GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
                GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)

    def hoverMoveEvent(self, event):
        layer = self.layer
        if layer is None:
            return
        image = layer.image
        if image is None:
            return
        x = event.pos().x()
        if x >= 0 and x <= 1:
            image_type = image.type
            histogram = image.histogram
            h_r = layer.histogram_min, layer.histogram_max
            h_w = h_r[1] - h_r[0]
            r = image.range
            w = r[1] - r[0]
            # TODO: fix this code when I'm not so tired
            bin_count = histogram.shape[-1] * h_w / w
            if image.dtype is not numpy.float32:
                bin_count = int(bin_count)
            bin = int(x * bin_count)
            # TODO: verify this more
            bin_width = (h_r[1] - h_r[0]) / (bin_count)# - 1)
            # print(bin_width)
            if image.dtype is numpy.float32:
                mst = '[{},{}{} '.format(h_r[0] + bin*bin_width, h_r[0] + (bin+1)*bin_width, ']' if bin == bin_count - 1 else ')')
            else:
                l, r = math.ceil(h_r[0] + bin*bin_width), math.floor(h_r[0] + (bin+1)*bin_width)
                mst = '{} '.format(l) if image.dtype is numpy.uint8 else '[{},{}] '.format(l, r)
            vt = '(' + ' '.join((c + ':{}' for c in image_type)) + ')'
            if image.num_channels > 1:
                vt = vt.format(*histogram[..., bin])
            else:
                vt = vt.format(histogram[bin])
            self.scene().update_contextual_info(mst + vt, self)

    def hoverLeaveEvent(self, event):
        self.scene().clear_contextual_info(self)

    def _on_layer_image_changed(self):
        self._layer_data_serial += 1
        self.min_item.arrow_item._on_value_changed()
        self.max_item.arrow_item._on_value_changed()
        self.update()

    def _on_gl_widget_context_about_to_change(self, gl_widget):
        assert gl_widget is self._gl_widget
        self.progs = {}
        if self._tex is not None:
            self._tex.destroy()
            self._tex = None

class MinMaxItem(Qt.QGraphicsObject):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()

    def __init__(self, histogram_item, name):
        super().__init__(histogram_item)
        self._bounding_rect = Qt.QRectF(-0.1, 0, .2, 1)
        self.arrow_item = MinMaxArrowItem(self, histogram_item, name)
        self.setFlag(Qt.QGraphicsItem.ItemIgnoresParentOpacity)

    def type(self):
        return MinMaxItem.QGRAPHICSITEM_TYPE

    def boundingRect(self):
        return self._bounding_rect

    def paint(self, qpainter, option, widget):
        pen = Qt.QPen(Qt.QColor(255,0,0,128))
        pen.setWidth(0)
        qpainter.setPen(pen)
        br = self.boundingRect()
        x = (br.left() + br.right()) / 2
        qpainter.drawLine(x, br.top(), x, br.bottom())

class MinMaxArrowItem(Qt.QGraphicsObject):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()

    def __init__(self, min_max_item, histogram_item, name):
        super().__init__(histogram_item)
        self.name = name
        self._path = Qt.QPainterPath()
        self._min_max_item = min_max_item
        if self.name.startswith('min'):
            polygonf = Qt.QPolygonF((Qt.QPointF(0.5, -12), Qt.QPointF(8, 0), Qt.QPointF(0.5, 12)))
        else:
            polygonf = Qt.QPolygonF((Qt.QPointF(-0.5, -12), Qt.QPointF(-8, 0), Qt.QPointF(-0.5, 12)))
        self._path.addPolygon(polygonf)
        self._path.closeSubpath()
        self._bounding_rect = self._path.boundingRect()
        self.setFlag(Qt.QGraphicsItem.ItemIgnoresParentOpacity)
        self.setFlag(Qt.QGraphicsItem.ItemIgnoresTransformations)
        self.setFlag(Qt.QGraphicsItem.ItemIsMovable)
        # GUI behavior is much more predictable with min/max arrow item selectability disabled:
        # with ItemIsSelectable enabled, min/max items can exhibit some very unexpected behaviors, as we
        # do not do anything differently in our paint function if the item is selected vs not, making
        # it unlikely one would realize one or more items are selected.  If multiple items are selected,
        # they will move together when one is dragged.  Additionally, arrow key presses would move
        # selected items if their viewport has focus (viewport focus is also not indicated).
        # Items are non-selectable by default; the following line is present only to make intent clear.
        #self.setFlag(Qt.QGraphicsItem.ItemIsSelectable, False)
        self._ignore_x_change = False
        self.setY(0.5)
        self.xChanged.connect(self._on_x_changed)
        self.yChanged.connect(self._on_y_changed)

    def type(self):
        return MinMaxArrowItem.QGRAPHICSITEM_TYPE

    def boundingRect(self):
        return self._bounding_rect

    def shape(self):
        return self._path

    def paint(self, qpainter, option, widget):
        c = Qt.QColor(255,0,0,128)
        qpainter.setPen(Qt.QPen(c))
        qpainter.setBrush(Qt.QBrush(c))
        qpainter.drawPath(self._path)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.scene().update_contextual_info('{}: {}'.format(self.name, getattr(self.parentItem().layer, self.name)), self)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.scene().update_contextual_info('{}: {}'.format(self.name, getattr(self.parentItem().layer, self.name)), self)

    def _on_x_changed(self):
        x = self.x()
        if not self._ignore_x_change:
            if x < 0:
                self.setX(0)
                x = 0
            elif x > 1:
                self.setX(1)
                x = 1
            layer = self.parentItem().layer
            r = layer.histogram_min, layer.histogram_max
            setattr(layer, self.name, r[0] + x * float(r[1] - r[0]))
        self._min_max_item.setX(x)

    def _on_y_changed(self):
        if self.y() != 0.5:
            self.setY(0.5)

    def _on_value_changed(self):
        self._ignore_x_change = True
        try:
            layer = self.parentItem().layer
            r = layer.histogram_min, layer.histogram_max
            self.setX( (getattr(layer, self.name) - r[0]) / (r[1] - r[0]) )
        finally:
            self._ignore_x_change = False

class GammaItem(Qt.QGraphicsObject):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()
    CURVE_VERTEX_Y_INCREMENT = 1 / 100

    def __init__(self, histogram_item, min_item, max_item):
        super().__init__(histogram_item)
        self._bounding_rect = Qt.QRectF(0, 0, 1, 1)
        self.min_item = min_item
        self.min_item.xChanged.connect(self._on_min_or_max_x_changed)
        self.max_item = max_item
        self.max_item.xChanged.connect(self._on_min_or_max_x_changed)
        self._path = Qt.QPainterPath()
        self.setFlag(Qt.QGraphicsItem.ItemIgnoresParentOpacity)
        self.setFlag(Qt.QGraphicsItem.ItemIsMovable)
        self.setZValue(-1)
        # This is a convenient way to ensure that only primary mouse button clicks cause
        # invocation of mouseMoveEvent(..).  Without this, it would be necessary to
        # override mousePressEvent(..) and check which buttons are down, in addition to
        # checking which buttons remain down in mouseMoveEvent(..).
        self.setAcceptedMouseButtons(Qt.Qt.LeftButton)

    def type(self):
        return GammaItem.QGRAPHICSITEM_TYPE

    def boundingRect(self):
        return self._bounding_rect

    def shape(self):
        pen = Qt.QPen()
        pen.setWidthF(0)
        stroker = Qt.QPainterPathStroker(pen)
        stroker.setWidth(0.2)
        return stroker.createStroke(self._path)

    def paint(self, qpainter, option, widget):
        if not self._path.isEmpty():
            pen = Qt.QPen(Qt.QColor(255,255,0,128))
            pen.setWidth(2)
            pen.setCosmetic(True)
            qpainter.setPen(pen)
            qpainter.setBrush(Qt.Qt.NoBrush)
            qpainter.drawPath(self._path)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.scene().update_contextual_info('gamma: {}'.format(self.parentItem().layer.gamma), self)

    def mouseMoveEvent(self, event):
        current_x, current_y = map(lambda v: min(max(v, 0.001), 0.999),
                                   (event.pos().x(), event.pos().y()))
        current_y = 1-current_y
        layer = self.parentItem().layer
        layer.gamma = gamma = min(max(math.log(current_y, current_x), layer.GAMMA_RANGE[0]), layer.GAMMA_RANGE[1])
        self.scene().update_contextual_info('gamma: {}'.format(gamma), self)

    def _on_min_or_max_x_changed(self):
        min_x = self.min_item.x()
        max_x = self.max_item.x()
        t = Qt.QTransform()
        t.translate(min_x, 0)
        t.scale(max_x - min_x, 1)
        self.setTransform(t)

    def _on_value_changed(self):
        self.prepareGeometryChange()
        self._path = Qt.QPainterPath(Qt.QPointF(0, 1))
        gamma = self.parentItem().layer.gamma
        # Compute sample point x locations such that the y increment from one sample point to the next is approximately
        # the constant, resulting in a fairly smooth gamma plot.  This is not particularly fast, but it only happens when 
        # gamma value changes, and it's fast enough that there is no noticable choppiness when dragging the gamma curve
        # up and down on a mac mini.
        xs = []
        x = 0
        while x < 1:
            xs.append(x)
            x += (GammaItem.CURVE_VERTEX_Y_INCREMENT + x**gamma)**(1/gamma) - x
        del xs[0]
        for x, y in zip(xs, (x**gamma for x in xs)):
            self._path.lineTo(x, 1.0-y)
        self._path.lineTo(1, 0)
        self.update()