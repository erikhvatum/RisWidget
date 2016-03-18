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
from ..point_list_picker import PointList, PointListPicker
from ..shared_resources import UNIQUE_QGRAPHICSITEM_TYPE

class PolyLinePointPicker(PointListPicker):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()

    def __init__(self, general_view, parent_item, points=None):
        super().__init__(general_view, parent_item, points)
        self.line_pen = Qt.QPen(Qt.Qt.green)
        self.line_pen.setWidth(5)
        self.line_pen.setCosmetic(True)
        self.point_list_contents_changed.connect(self._on_point_list_contents_changed)
        self._on_point_list_contents_changed()

    def sceneEventFilter(self, watched, event):
        if isinstance(watched, Qt.QGraphicsLineItem) and event.type() == Qt.QEvent.GraphicsSceneMousePress and event.modifiers() == Qt.Qt.NoModifier:
            if event.button() == Qt.Qt.LeftButton:
                for point_item in self.point_items.values():
                    point_item.setSelected(True)
                # Focus a point item so that the delete key shortcut works
                self.point_items[self._points[0]].setFocus()
                return True
            if event.button() == Qt.Qt.RightButton:
                # Find the point this line extends from
                for idx, point in enumerate(self._points):
                    try:
                        point_item = self.point_items[point]
                    except KeyError:
                        continue
                    try:
                        edge_item = point_item.edge_item
                    except AttributeError:
                        continue
                    if edge_item is watched:
                        break
                else:
                    return False
                # point is now the value we're looking for - otherwise, we wouldn't have broken out of the for loop and the else clause would have
                # executed, returning us from this function before we got here
                inserted_point_pos = self.parentItem().mapFromScene(event.scenePos())
                self._points.insert(idx + 1, inserted_point_pos)
                return True
        return super().sceneEventFilter(watched, event)

    def _on_point_list_contents_changed(self):
        if not self._points:
            return
        if len(self._points) == 1:
            # Delete only point's line segment
            try:
                edge_item = self.point_items[self._points[0]].edge_item
            except AttributeError:
                return
            if edge_item.scene():
                edge_item.scene().removeItem(edge_item)
            del self.point_items[self._points[0]].edge_item
            return
        # Make and update edges (line segments) as needed
        pis = [(point, self.point_items[point]) for point in self._points]
        for (point1, point_item1), (point2, point_item2) in zip(pis, pis[1:]):
            line = Qt.QLineF(point1.x, point1.y, point2.x, point2.y)
            if not hasattr(point_item1, 'edge_item'):
                point_item1.edge_item = Qt.QGraphicsLineItem(line, self.parentItem())
                point_item1.edge_item.setPen(self.line_pen)
                point_item1.edge_item.installSceneEventFilter(self)
                point_item1.edge_item.setZValue(-1)
            elif point_item1.edge_item.line() != line:
                point_item1.edge_item.setLine(line)
        # Delete last point's line segment
        try:
            edge_item = self.point_items[self._points[-1]].edge_item
        except AttributeError:
            return
        if edge_item.scene():
            edge_item.scene().removeItem(edge_item)
        del self.point_items[self._points[-1]].edge_item

    def _detach_point(self, point):
        # Destroy orphaned edge item (if one exists) before handing control to base implementation
        try:
            point_item = self.point_items[point]
        except KeyError:
            pass
        else:
            try:
                edge_item = point_item.edge_item
            except AttributeError:
                pass
            else:
                if edge_item.scene():
                    edge_item.scene().removeItem(edge_item)
                del point_item.edge_item
        super()._detach_point(point)