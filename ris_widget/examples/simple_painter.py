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

class SimplePainter:
    def __init__(self, view, item):
        self.view = view
        self.item = item
        self.path_item = Qt.QGraphicsPathItem(item)
        pen = Qt.QPen(Qt.Qt.green)
        pen.setWidth(5)
        self.path_item.setPen(pen)
        self.path = Qt.QPainterPath()
        self.view.mouse_event_signal.connect(self.on_mouse_event_in_view)
        self.points = []

    def on_mouse_event_in_view(self, event_type, event, scene_pos):
        if event_type == 'press' and event.buttons() == Qt.Qt.RightButton:
            pos = self.item.mapFromScene(scene_pos)
            if not self.points:
                self.path.moveTo(pos)
            else:
                self.path.lineTo(pos)
            self.points.append((pos.x(), pos.y()))
            self.path_item.setPath(self.path)
            event.accept()

    def clear(self):
        self.path = Qt.QPainterPath()
        self.path_item.setPath(self.path)
        self.points = []