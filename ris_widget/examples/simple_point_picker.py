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

class SimplePointPicker:
    """ex:
    from ris_widget.ris_widget import RisWidget
    from ris_widget.examples.simple_point_picker import SimplePointPicker
    rw = RisWidget()
    simple_point_picker = SimplePointPicker(rw.main_view, rw.main_scene.layer_stack_item)"""

    def __init__(self, view, item):
        self.view = view
        self.item = item
        self.point_items = []
        self.pen = Qt.QPen(Qt.Qt.red)
        self.pen.setWidth(2)
        color = Qt.QColor(Qt.Qt.yellow)
        color.setAlphaF(0.5)
        self.brush = Qt.QBrush(color)
        self.view.mouse_event_signal.connect(self.on_mouse_event_in_view)
        self.view.key_event_signal.connect(self.on_key_event_in_view)
        self.view.scene_region_changed.connect(self.on_scene_region_changed)

    def make_and_store_point_item(self, pos):
        point_item = Qt.QGraphicsRectItem(-7, -7, 15, 15, self.item)
        point_item.setPos(pos)
        point_item.setScale(1 / self.view.transform().m22())
        point_item.setPen(self.pen)
        point_item.setBrush(self.brush)
        flags = point_item.flags()
        point_item.setFlags(
            flags |
            Qt.QGraphicsItem.ItemIsSelectable |
            Qt.QGraphicsItem.ItemIsMovable
        )
        self.point_items.append(point_item)

    def on_mouse_event_in_view(self, event_type, event, scene_pos):
        if event_type == 'press' and event.buttons() == Qt.Qt.RightButton:
            self.make_and_store_point_item(self.item.mapFromScene(scene_pos))

    def on_key_event_in_view(self, event_type, event):
        if event_type == 'press' and event.key() == Qt.Qt.Key_Delete and event.modifiers() == Qt.Qt.NoModifier:
            for idx in reversed(range(len(self.point_items))):
                if self.point_items[idx].isSelected():
                    self.view.scene().removeItem(self.point_items[idx])
                    del self.point_items[idx]
            event.accept()

    def on_scene_region_changed(self, view):
        assert view is self.view
        scale = 1 / self.view.transform().m22()
        for point_item in self.point_items:
            point_item.setScale(scale)

    def clear(self):
        for point_item in self.point_items:
            self.view.scene().removeItem(point_item)
        self.point_items = []

    @property
    def points(self):
        return [(point_item.pos().x(), point_item.pos().y()) for point_item in self.point_items]

    @points.setter
    def points(self, points):
        self.clear()
        for point in points:
            self.make_and_store_point_item(Qt.QPointF(point[0], point[1]))