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
    def __init__(self, parent):
        super().__init__(parent)
        self.setFrameShape(Qt.QFrame.StyledPanel)
        self.setFrameShadow(Qt.QFrame.Raised)
        self.image_widget = ImageWidget(self)
        self.setViewport(self.image_widget)

    def scrollContentsBy(dx, dy):
        self.image_widget.scroll_contents_by(dx, dy)

class ImageWidget(Qt.QOpenGLWidget):
    def __init__(self, parent):
        super().__init__(parent)
        format = Qt.QSurfaceFormat()
        format.setRenderableType(Qt.QSurfaceFormat.OpenGL)
        format.setVersion(2, 1)
        format.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
        format.setSwapBehavior(Qt.QSurfaceFormat.DoubleBuffer)
        format.setStereo(False)
        format.setSwapInterval(1)
        self.setFormat(format)

    def initializeGL(self):
        # PyQt5 provides access to OpenGL functions up to OpenGL 2.0, but we have made a 2.1
        # context.  QOpenGLContext.versionFunctions(..) will, by default, attempt to return
        # a wrapper around QOpenGLFunctions2_1, which will fail, as there is no
        # PyQt5._QOpenGLFunctions_2_1 implementation.  Therefore, we explicitly request 2.0
        # functions, and any 2.1 calls that we want to make can not occur through self.glfs.
        vp = Qt.QOpenGLVersionProfile()
        vp.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
        vp.setVersion(2, 0)
        self.glfs = self.context().versionFunctions(vp)

    def paintGL(self):
        pass

    def resizeGL(self, x, y):
        pass

    def scroll_contents_by(self, dx, dy):
        pass
