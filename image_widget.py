# The MIT License (MIT)
#
# Copyright (c) 2014 WUSTL ZPLAB
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

import enum
from PyQt5 import Qt
import sys

class ImageWidgetScroller(Qt.QAbstractScrollArea):
    scroll_contents_by = Qt.pyqtSignal(int, int)
    def scrollContentsBy(dx, dy):
        scroll_contents_by.emit(dx, dy)

class ImageWidget(Qt.QOpenGLWidget):
    def __init__(self, parent):
        super().__init__(parent)
        format = Qt.QSurfaceFormat()
        format.setRenderableType(Qt.QSurfaceFormat.OpenGL)
        format.setVersion(4, 1)
        format.setProfile(Qt.QSurfaceFormat.CoreProfile)
        format.setSwapBehavior(Qt.QSurfaceFormat.DoubleBuffer)
        format.setStereo(False)
        format.setSwapInterval(1)
        self.setFormat(format)

    def initializeGL(self):
        pass

    def paintGL(self):
        pass

    def resizeGL(self, resize_event):
        pass


#def make_image_widget_in_scroller(parent):

