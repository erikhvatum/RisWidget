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

from .histogram_scene import MinMaxArrowItem
from .shader_view import ShaderView
from PyQt5 import Qt

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

    def on_resize(self, size):
        super().on_resize(size)
        # Adjust this view's transform such that unit square scene rect fills resized viewport
        self.resetTransform()
        self.scale(size.width(), size.height())

    def on_resize_done(self, size):
        """HistogramView always displays the same region of the scene (the unit square) regardless of
        view size, so there is no need for our base class's implementation of this function to
        emit scene_view_rect_changed.  For this reason, this override does not call super().on_resize_done(..)."""
        pass
