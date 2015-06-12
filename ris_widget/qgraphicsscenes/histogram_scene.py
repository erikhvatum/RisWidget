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
import sys
from ..qgraphicsitems.shader_item import ShaderItem, ShaderTexture
from ..shared_resources import UNIQUE_QGRAPHICSITEM_TYPE
from .base_scene import BaseScene

class HistogramScene(BaseScene):
    def __init__(self, parent, image_item, HistogramItemClass, ContextualInfoItemClass):
        super().__init__(parent, ContextualInfoItemClass)
        self.setSceneRect(0, 0, 1, 1)
#       self.histogram_item = HistogramItemClass()
#       self.addItem(self.histogram_item)
#       self._image_item = None
#       self.image_item = image_item
#       self.gamma_gamma = 1.0
#
#   @property
#   def image_item(self):
#       return self._image_item
#
#   @image_item.setter
#   def image_item(self, image_item):
#       assert image_item is not None
#       if image_item is not self._image_item:
#           if self._image_item is not None:
#               self._image_item.image_changed.disconnect(self.histogram_item._on_image_changed)
#               self._image_item.min_changed.disconnect(self.histogram_item.min_item.arrow_item._on_value_changed)
#               self._image_item.max_changed.disconnect(self.histogram_item.max_item.arrow_item._on_value_changed)
#               self._image_item.gamma_changed.disconnect(self.histogram_item.gamma_item._on_value_changed)
#           self._image_item = image_item
#           self._image_item.image_changed.connect(self.histogram_item._on_image_changed)
#           self._image_item.min_changed.connect(self.histogram_item.min_item.arrow_item._on_value_changed)
#           self._image_item.max_changed.connect(self.histogram_item.max_item.arrow_item._on_value_changed)
#           self._image_item.gamma_changed.connect(self.histogram_item.gamma_item._on_value_changed)
#           self.histogram_item.min_item.arrow_item._on_value_changed()
#           self.histogram_item.max_item.arrow_item._on_value_changed()
#           self.histogram_item.gamma_item._on_value_changed()

class HistogramItem(ShaderItem):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()

    def __init__(self, graphics_item_parent=None):
        super().__init__(graphics_item_parent)
        self._image_id = 0
        self._bounding_rect = Qt.QRectF(0, 0, 1, 1)
        self._tex = None
#       self.min_item = MinMaxItem(self, 'min', ImageItem.min, ImageItem.normalized_min)
#       self.max_item = MinMaxItem(self, 'max', ImageItem.max, ImageItem.normalized_max)
#       self.gamma_item = GammaItem(self, ImageItem.gamma, self.min_item, self.max_item)

    def type(self):
        return HistogramItem.QGRAPHICSITEM_TYPE

    def boundingRect(self):
        return self._bounding_rect

    def paint(self, qpainter, option, widget):
        pass
#       assert widget is not None, 'histogram_scene.HistogramItem.paint called with widget=None.  Ensure that view caching is disabled.'
#       image = self.scene()._image_item._image
#       if image is None:
#           if self._tex is not None:
#               self._tex.destroy()
#               self._tex = None
#               self._tex_is_stale = False
#       else:
#           view = widget.view
#           scene = self.scene()
#           with ExitStack() as stack:
#               qpainter.beginNativePainting()
#               stack.callback(qpainter.endNativePainting)
#               GL = widget.GL
#               desired_shader_type = 'g'
#               if desired_shader_type in self.progs:
#                   prog = self.progs[desired_shader_type]
#               else:
#                   prog = self.build_shader_prog(desired_shader_type,
#                                                 'histogram_widget_vertex_shader.glsl',
#                                                 'histogram_widget_fragment_shader_{}.glsl'.format(desired_shader_type))
#               desired_tex_width = image.histogram.shape[-1]
#               tex = self._tex
#               if tex is not None:
#                   if tex.width != desired_tex_width:
#                       tex.destroy()
#                       tex = self._tex = None
#               if tex is None:
#                   tex = ShaderTexture(GL.GL_TEXTURE_1D, widget.GL)
#                   tex.bind()
#                   stack.callback(tex.release)
#                   GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
#                   GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
#                   # tex stores histogram bin counts - values that are intended to be addressed by element without
#                   # interpolation.  Thus, nearest neighbor for texture filtering.
#                   GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
#                   GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
#                   tex.image_id = -1
#               else:
#                   tex.bind()
#                   stack.callback(tex.release)
#               histogram = image.histogram
#               max_bin_val = histogram[image.max_histogram_bin]
#               if tex.image_id != self._image_id:
#                   orig_unpack_alignment = GL.glGetIntegerv(GL.GL_UNPACK_ALIGNMENT)
#                   if orig_unpack_alignment != 1:
#                       GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
#                       # QPainter font rendering for OpenGL surfaces will become broken if we do not restore GL_UNPACK_ALIGNMENT
#                       # to whatever QPainter had it set to (when it prepared the OpenGL context for our use as a result of
#                       # qpainter.beginNativePainting()).
#                       stack.callback(lambda oua=orig_unpack_alignment: GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, oua))
#                   GL.glTexImage1D(GL.GL_TEXTURE_1D, 0,
#                                   GL.GL_LUMINANCE32UI_EXT, desired_tex_width, 0,
#                                   GL.GL_LUMINANCE_INTEGER_EXT, GL.GL_UNSIGNED_INT,
#                                   histogram.data)
#                   tex.image_id = self._image_id
#                   tex.width = desired_tex_width
#                   self._tex = tex
#               view.quad_buffer.bind()
#               stack.callback(view.quad_buffer.release)
#               view.quad_vao.bind()
#               stack.callback(view.quad_vao.release)
#               prog.bind()
#               stack.callback(prog.release)
#               vert_coord_loc = prog.attributeLocation('vert_coord')
#               prog.enableAttributeArray(vert_coord_loc)
#               prog.setAttributeBuffer(vert_coord_loc, GL.GL_FLOAT, 0, 2, 0)
#               prog.setUniformValue('tex', 0)
#               prog.setUniformValue('inv_view_size', 1/widget.size().width(), 1/widget.size().height())
#               inv_max_transformed_bin_val = max_bin_val**-scene.gamma_gamma
#               prog.setUniformValue('inv_max_transformed_bin_val', inv_max_transformed_bin_val)
#               prog.setUniformValue('gamma_gamma', scene.gamma_gamma)
#               GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
#               GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)

    def hoverMoveEvent(self, event):
        pass
#       image = self.scene()._image_item._image
#       if image is not None:
#           x = event.pos().x()
#           if x >= 0 and x <= 1:
#               image_type = image.type
#               histogram = image.histogram
#               range_ = image.range
#               bin_count = histogram.shape[-1]
#               bin = int(x * bin_count)
#               bin_width = (range_[1] - range_[0]) / bin_count
#               if image.dtype == numpy.float32:
#                   mst = '[{},{}) '.format(range_[0] + bin*bin_width, range_[0] + (bin+1)*bin_width)
#               else:
#                   mst = '[{},{}] '.format(math.ceil(bin*bin_width), math.floor((bin+1)*bin_width))
#               vt = '(' + ' '.join((c + ':{}' for c in image_type)) + ')'
#               vt = vt.format(histogram[bin])
#               self.scene().update_contextual_info(mst + vt, self)

    def hoverLeaveEvent(self, event):
        self.scene().clear_contextual_info(self)

    def _on_image_changed(self):
        self._image_id += 1
        self.update()

class MinMaxItem(Qt.QGraphicsObject):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()

    def __init__(self, histogram_item, name, value_prop, normalized_value_prop):
        super().__init__(histogram_item)
        self._bounding_rect = Qt.QRectF(-0.1, 0, .2, 1)
        self.arrow_item = MinMaxArrowItem(self, histogram_item, name, value_prop, normalized_value_prop)

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

    def __init__(self, min_max_item, histogram_item, name, value_prop, normalized_value_prop):
        super().__init__(histogram_item)
        self.name = name
        self._value_prop = value_prop
        self._normalized_value_prop = normalized_value_prop
        self._path = Qt.QPainterPath()
        self._min_max_item = min_max_item
        if self.name.startswith('min'):
            polygonf = Qt.QPolygonF((Qt.QPointF(0.5, -12), Qt.QPointF(8, 0), Qt.QPointF(0.5, 12)))
        else:
            polygonf = Qt.QPolygonF((Qt.QPointF(-0.5, -12), Qt.QPointF(-8, 0), Qt.QPointF(-0.5, 12)))
        self._path.addPolygon(polygonf)
        self._path.closeSubpath()
        self._bounding_rect = self._path.boundingRect()
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
        self.scene().update_contextual_info('{}: {}'.format(self.name, self._value_prop.fget(self.scene()._image_item)), self)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.scene().update_contextual_info('{}: {}'.format(self.name, self._value_prop.fget(self.scene()._image_item)), self)

    def _on_x_changed(self):
        x = self.x()
        if not self._ignore_x_change:
            if x < 0:
                self.setX(0)
                x = 0
            elif x > 1:
                self.setX(1)
                x = 1
            self._normalized_value_prop.fset(self.scene().image_item, x)
        self._min_max_item.setX(x)

    def _on_y_changed(self):
        if self.y() != 0.5:
            self.setY(0.5)

    def _on_value_changed(self):
        self._ignore_x_change = True
        try:
            self.setX(self._normalized_value_prop.fget(self.scene().image_item))
        finally:
            self._ignore_x_change = False

class GammaItem(Qt.QGraphicsObject):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()
    CURVE_VERTEX_Y_INCREMENT = 1 / 100

    def __init__(self, histogram_item, value_prop, min_item, max_item):
        super().__init__(histogram_item)
        self._bounding_rect = Qt.QRectF(0, 0, 1, 1)
        self._value_prop = value_prop
        self.min_item = min_item
        self.min_item.xChanged.connect(self._on_min_or_max_x_changed)
        self.max_item = max_item
        self.max_item.xChanged.connect(self._on_min_or_max_x_changed)
        self._path = Qt.QPainterPath()
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
        self.scene().update_contextual_info('gamma: {}'.format(self._value_prop.fget(self.scene()._image_item)), self)

    def mouseMoveEvent(self, event):
        current_x, current_y = map(lambda v: min(max(v, 0.001), 0.999),
                                   (event.pos().x(), event.pos().y()))
        current_y = 1-current_y
        scene = self.scene()
        image_item = scene._image_item
        image_item.gamma = min(max(math.log(current_y, current_x), ImageItem.GAMMA_RANGE[0]), ImageItem.GAMMA_RANGE[1])
        scene.update_contextual_info('gamma: {}'.format(self._value_prop.fget(image_item)), self)

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
        gamma = self._value_prop.fget(self.scene()._image_item)
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