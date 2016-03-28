# The MIT License (MIT)
#
# Copyright (c) 2016 WUSTL ZPLAB
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

from PyQt5 import Qt
from ..shared_resources import UNIQUE_QGRAPHICSITEM_TYPE

class PointItem(Qt.QGraphicsRectItem):
    # Omitting .type() or failing to return a unique causes PyQt to return a wrapper of the wrong type when retrieving an instance of this item as a base
    # class pointer from C++.  For example, if this item has a child and that child calls self.parentItem(), it would receive a Python object of type
    # Qt.QGraphicsRectItem rather than PointItem unless PointItem has a correct .type() implementation.
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()

    def __init__(self, picker, x, y, w, h, parent_item):
        super().__init__(x, y, w, h, parent_item)
        self.picker = picker
        flags = self.flags()
        self.setFlags(
            flags |
            Qt.QGraphicsItem.ItemIsFocusable | # Necessary in order for item to receive keyboard events
            Qt.QGraphicsItem.ItemIsSelectable |
            Qt.QGraphicsItem.ItemIsMovable |
            Qt.QGraphicsItem.ItemSendsGeometryChanges # Necessary in order for .itemChange to be called when item is moved
        )

    def itemChange(self, change, value):
        if change == Qt.QGraphicsItem.ItemPositionHasChanged:
            self.picker.point_item_position_has_changed.emit(self)
        return super().itemChange(change, value)

    def keyPressEvent(self, event):
        if event.key() == Qt.Qt.Key_Delete and event.modifiers() == Qt.Qt.NoModifier:
            self.picker.delete_selected()

    def type(self):
        return self.QGRAPHICSITEM_TYPE

# NB: deriving from Qt.QGraphicsObject is necessary in order to be a scene event filter target
class SimplePointPicker(Qt.QGraphicsObject):
    """ex:
    from ris_widget.ris_widget import RisWidget
    from ris_widget.examples.simple_point_picker import SimplePointPicker
    rw = RisWidget()
    simple_point_picker = SimplePointPicker(rw.main_view, rw.main_scene.layer_stack_item)"""
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()

    point_item_position_has_changed = Qt.pyqtSignal(PointItem)
    point_item_list_content_reset = Qt.pyqtSignal()

    def __init__(self, general_view, parent_item, points=None):
        super().__init__(parent_item)
        self.view = general_view
        self.view.viewport_rect_item.size_changed.connect(self.on_viewport_size_changed)
        self.point_items = []
        self.pen = Qt.QPen(Qt.Qt.red)
        self.pen.setWidth(2)
        color = Qt.QColor(Qt.Qt.yellow)
        color.setAlphaF(0.5)
        self.brush = Qt.QBrush(color)
        self.brush_selected = Qt.QBrush(Qt.QColor(255, 0, 255, 127))
        parent_item.installSceneEventFilter(self)
        if points:
            for point in points:
                self.make_and_store_point_item(Qt.QPointF(point[0], point[1]))

    def boundingRect(self):
        return Qt.QRectF()

    def paint(self, QPainter, QStyleOptionGraphicsItem, QWidget_widget=None):
        pass

    def type(self):
        return self.QGRAPHICSITEM_TYPE

    def make_and_store_point_item(self, pos):
        point_item = PointItem(self, -7, -7, 15, 15, self.parentItem())
        point_item.setScale(1 / self.view.transform().m22())
        point_item.setPen(self.pen)
        point_item.setBrush(self.brush)
        flags = point_item.flags()
        point_item.setFlags(
            flags |
            Qt.QGraphicsItem.ItemIsFocusable | # Necessary in order for item to receive keyboard events
            Qt.QGraphicsItem.ItemIsSelectable |
            Qt.QGraphicsItem.ItemIsMovable |
            Qt.QGraphicsItem.ItemSendsGeometryChanges
        )
        point_item.installSceneEventFilter(self)
        self.point_items.append(point_item)
        point_item.setPos(pos)

    def delete_selected(self):
        for idx, item in reversed(list(enumerate((self.point_items)))):
            if item.isSelected():
                self.scene().removeItem(item)
                del self.point_items[idx]
        self.point_item_list_content_reset.emit()

    def sceneEventFilter(self, watched, event):
        if watched is self.parentItem():
            if event.type() == Qt.QEvent.GraphicsSceneMousePress and event.button() == Qt.Qt.RightButton:
                self.make_and_store_point_item(event.pos())
                return True
            if event.type() == Qt.QEvent.KeyPress and event.key() == Qt.Qt.Key_Delete and event.modifiers() == Qt.Qt.NoModifier:
                self.delete_selected()
        return False

    def on_viewport_size_changed(self):
        scale = 1 / self.view.transform().m22()
        for point_item in self.point_items:
            point_item.setScale(scale)

    def clear(self):
        for point_item in self.point_items:
            self.view.scene().removeItem(point_item)
        self.point_items = []
        self.point_item_list_content_reset.emit()

    @property
    def points(self):
        return [(point_item.pos().x(), point_item.pos().y()) for point_item in self.point_items]

    @points.setter
    def points(self, points):
        self.clear()
        for point in points:
            self.make_and_store_point_item(Qt.QPointF(point[0], point[1]))