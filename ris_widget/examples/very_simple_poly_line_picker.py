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

class VerySimplePolyLinePicker(Qt.QGraphicsObject):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()

    def __init__(self, general_view, parent_item):
        super().__init__(parent_item)
        self.view = general_view
        self.path_item = Qt.QGraphicsPathItem(parent_item)
        pen = Qt.QPen(Qt.Qt.green)
        pen.setWidth(5)
        pen.setCosmetic(True)
        self.path_item.setPen(pen)
        self.path = Qt.QPainterPath()
        self.points = []
        parent_item.installSceneEventFilter(self)

    def type(self):
        return self.QGRAPHICSITEM_TYPE

    def boundingRect(self):
        return Qt.QRectF()

    def paint(self, QPainter, QStyleOptionGraphicsItem, QWidget_widget=None):
        pass

    def sceneEventFilter(self, watched, event):
        if watched is self.parentItem() and event.type() == Qt.QEvent.GraphicsSceneMousePress and event.button() == Qt.Qt.RightButton:
            pos = event.pos()
            if not self.points:
                self.path.moveTo(pos)
            else:
                self.path.lineTo(pos)
            self.points.append((pos.x(), pos.y()))
            self.path_item.setPath(self.path)
            return True
        return False

    def clear(self):
        self.path = Qt.QPainterPath()
        self.path_item.setPath(self.path)
        self.points = []