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

# class PolyRegionMaskPicker(PointListPicker):
#     def __init__(self, general_view, parent_item, points=None, PointListType=PointList, parent=None):
#         super().__init__(general_view, parent_item, points, PointListType, parent)
#         self.path_item = Qt.QGraphicsPathItem(parent_item)
#         pen = Qt.QPen(Qt.Qt.green)
#         pen.setWidth(5)
#         pen.setCosmetic(True)
#         self.path_item.setPen(pen)
#         self.point_list_contents_changed.connect(self.on_point_list_contents_changed)
#
#     def on_point_list_contents_changed(self):
#         path = Qt.QPainterPath()
#         if len(self.points) >= 2:
#             path.moveTo(self.points[0].x, self.points[0].y)
#             for point in self.points[1:]:
#                 path.lineTo(point.x, point.y)
#         self.path_item.setPath(path)

# @property
#     def mask(self):
#         pass