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

class ViewportRectItem(Qt.QGraphicsObject):
    size_changed = Qt.pyqtSignal(Qt.QSizeF)
    scene_position_changed = Qt.pyqtSignal(Qt.QPointF)

    def __init__(self):
        super().__init__()
        self._is_visible = False
        self.setFlags(
            Qt.QGraphicsItem.ItemIgnoresTransformations |
            Qt.QGraphicsItem.ItemSendsGeometryChanges |
            Qt.QGraphicsItem.ItemSendsScenePositionChanges |
            Qt.QGraphicsItem.ItemHasNoContents
        )
        self._size = Qt.QSizeF()
        # Children are generally overlay items that should appear over anything else rather than z-fighting
        self.setZValue(10)

    @property
    def is_visible(self):
        return self._is_visible

    @is_visible.setter
    def is_visible(self, v):
        v = bool(v)
        if self._is_visible != v:
            self.setFlag(Qt.QGraphicsItem.ItemHasNoContents, not v)
            self.setOpacity(0.25 if v else 1)
            self._is_visible = v
            self.update()

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, v):
        if not isinstance(v, Qt.QSizeF):
            v = Qt.QSizeF(v)
        if self._size != v:
            self.prepareGeometryChange()
            self._size = v
        self.size_changed.emit(v)

    def boundingRect(self):
        return Qt.QRectF(Qt.QPointF(), self._size)

    def itemChange(self, change, value):
        if change == Qt.QGraphicsItem.ItemScenePositionHasChanged:
            self.scene_position_changed.emit(value)
            return
        return super().itemChange(change, value)

    def paint(self, qpainter, option, widget):
        if self._is_visible:
            qpainter.fillRect(option.rect, Qt.Qt.white)