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
from .simple_point_picker import SimplePointPicker

class SimplePolyLinePointPicker(SimplePointPicker):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()

    def __init__(self, general_view, parent_item, points=None):
        self.line_items = []
        self.line_pen = Qt.QPen(Qt.Qt.green)
        self.line_pen.setWidth(5)
        self.line_pen.setCosmetic(True)
        super().__init__(general_view, parent_item, points)
        self.point_item_position_has_changed.connect(self._on_point_item_position_has_changed)
        self.point_item_list_content_reset.connect(self._on_point_item_list_content_reset)
        self._ignore_point_item_position_changed = False

    def make_and_store_point_item(self, pos):
        self._ignore_point_item_position_changed = True
        try:
            super().make_and_store_point_item(pos)
        finally:
            self._ignore_point_item_position_changed = False
        if len(self.point_items) > 1:
            p1 = self.point_items[-2].pos()
            line_item = Qt.QGraphicsLineItem(Qt.QLineF(p1, pos), self.parentItem())
            line_item.setPen(self.line_pen)
            line_item.installSceneEventFilter(self)
            line_item.setZValue(-1)
            self.line_items.append(line_item)

    def sceneEventFilter(self, watched, event):
        is_line_click = (
            isinstance(watched, Qt.QGraphicsLineItem) and
            event.type() == Qt.QEvent.GraphicsSceneMousePress and
            event.button() == Qt.Qt.LeftButton and
            event.modifiers() == Qt.Qt.NoModifier
        )
        if is_line_click:
            for point_item in self.point_items:
                point_item.setSelected(True)
            # Focus a point item so that the delete key shortcut works
            self.point_items[0].setFocus()
            return True
        return super().sceneEventFilter(watched, event)

    def _on_point_item_position_has_changed(self, point_item):
        if not self._ignore_point_item_position_changed:
            idx = self.point_items.index(point_item)
            if idx > 0:
                line_item = self.line_items[idx - 1]
                line = line_item.line()
                line.setP2(point_item.pos())
                line_item.setLine(line)
            if idx < len(self.point_items) - 1:
                line_item = self.line_items[idx]
                line = line_item.line()
                line.setP1(point_item.pos())
                line_item.setLine(line)

    def _on_point_item_list_content_reset(self):
        for line_item in self.line_items:
            self.view.scene().removeItem(line_item)
        self.line_items = []
        if len(self.point_items) > 1:
            for point_item1, point_item2 in zip(self.point_items, self.point_items[1:]):
                line_item = Qt.QGraphicsLineItem(Qt.QLineF(point_item1.pos(), point_item2.pos()), self.parentItem())
                line_item.setPen(self.line_pen)
                line_item.installSceneEventFilter(self)
                line_item.setZValue(-1)
                self.line_items.append(line_item)