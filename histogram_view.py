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

from PyQt5 import Qt
from .shader_view import ShaderView

class HistogramView(ShaderView):
    @classmethod
    def make_histogram_view_and_frame(cls, shader_scene, parent):
        histogram_frame = Qt.QFrame(parent)
        histogram_frame.setMinimumSize(Qt.QSize(120, 60))
        histogram_frame.setFrameShape(Qt.QFrame.StyledPanel)
        histogram_frame.setFrameShadow(Qt.QFrame.Sunken)
        histogram_frame.setLayout(Qt.QHBoxLayout())
        histogram_frame.layout().setSpacing(0)
        histogram_frame.layout().setContentsMargins(Qt.QMargins(0,0,0,0))
        histogram_view = cls(shader_scene, histogram_frame)
        histogram_frame.layout().addWidget(histogram_view)
        return (histogram_view, histogram_frame)

    def __init__(self, shader_scene, parent):
        super().__init__(shader_scene, parent)
        self.resized.connect(self.on_resized)
#       self.add_overlay_min_max_arrow_items()

    def on_resized(self, size):
        self.resetTransform()
        self.scale(size.width(), size.height())

#   def add_overlay_min_max_arrow_items(self):
#       pen = Qt.QPen(Qt.QColor(Qt.Qt.transparent))
#       color = Qt.QColor(Qt.Qt.red)
#       color.setAlphaF(0.5)
#       brush = Qt.QBrush(color)
#       histogram_scene = self.scene()
#
#       polygon = Qt.QPolygonF((Qt.QPointF(0.5, -10), Qt.QPointF(6, 0), Qt.QPointF(0.5, 10)))
#       self.overlay_min_arrow_item = self.overlay_scene.addPolygon(polygon, pen, brush)
#       min_max_item = histogram_scene.get_prop_item('min')
#       min_max_item.xChanged.connect(lambda min_max_item=min_max_item: \
#                                            self.update_min_max_arrow_item_pos(self.overlay_min_arrow_item, min_max_item))
#       self.resized.connect(lambda _,
#                                   min_max_item=min_max_item: \
#                                   self.update_min_max_arrow_item_pos(self.overlay_min_arrow_item, min_max_item))
#
#       polygon = Qt.QPolygonF((Qt.QPointF(-0.5, -10), Qt.QPointF(-6, 0), Qt.QPointF(-0.5, 10)))
#       self.overlay_max_arrow_item = self.overlay_scene.addPolygon(polygon, pen, brush)
#       min_max_item = histogram_scene.get_prop_item('max')
#       min_max_item.xChanged.connect(lambda min_max_item=min_max_item: \
#                                            self.update_min_max_arrow_item_pos(self.overlay_max_arrow_item, min_max_item))
#       self.resized.connect(lambda _,
#                                   min_max_item=min_max_item: \
#                                   self.update_min_max_arrow_item_pos(self.overlay_max_arrow_item, min_max_item))
#
#   def update_min_max_arrow_item_pos(self, overlay_min_max_arrow_item, min_max_item):
#       view_size = self.viewport().size()
#       overlay_min_max_arrow_item.setPos(min_max_item.x() * view_size.width(), view_size.height() / 2)
