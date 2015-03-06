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

from .shader_resources import GL_QSURFACE_FORMAT, init_GL_QSURFACE_FORMAT, GL, init_GL

class Viewport(Qt.QGraphicsView):
    pass

class _ViewportGLWidget(Qt.QOpenGLWidget):
    """In order to obtain a QGraphicsView instance that renders into an OpenGL 2.1
    compatibility context (OS X does not support OpenGL 3.0+ compatibility profile,
    and Qt 5.4.1 QPainter support relies upon legacy calls, limiting us to 2.1 on
    OS X - and everywhere else as a practicality), I know of two supported methods:

    * Call the static method Qt.QSurfaceFormat.setDefaultFormat(our_format).  All
    widgets created thereafter will render into OpenGL 2.1 compat contexts.  That can
    possibly be avoided, by attempting to stash and restore the value of
    Qt.QSurfaceFormat.defaultFormat(), but this should must be done with care.  On
    your particular platform, when does QGraphicsView read
    Qt.QSurfaceFormat.defaultFormat()?  When you show() the widget?  During the next
    iteration of the event loop after you show() the widget?  Also, you don't want
    to interfere with intervening instantiation of a 3rd party widget (eg a,
    matplotlib chart, which may want the software rasterizer).

    * Make a QGLWidget (old-school) or QOpenGLWidget (new school) and feed it to
    QGraphicsView's setViewport method.  _ViewportGLWidget is that QOpenGLWidget.
    _ViewportGLWidget doesn't really do anything except set up an OpenGL context.
    All relevant QEvents are filtered by the QGraphicsView-derived instance.
    _ViewportGLWidget has paintGL and resizeGL methods only because older PyQt5
    versions verify that these exist in order to enforce the
    no-class-instances-with-pure-virtual-methods C++ contract."""

    gl_initializing = Qt.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        init_GL_QSURFACE_FORMAT()
        self.setFormat(GL_QSURFACE_FORMAT)
        self.view = None

    def initializeGL(self):
        # PyQt5 provides access to OpenGL functions up to OpenGL 2.0, but we have made a 2.1
        # context.  QOpenGLContext.versionFunctions(..) will, by default, attempt to return
        # a wrapper around QOpenGLFunctions2_1, which may fail, as there is not necessarily
        # a PyQt5._QOpenGLFunctions_2_1 implementation.  Therefore, we explicitly request 2.0
        # functions, and any 2.1 calls that we want to make can not occur through self.glfs.
        vp = Qt.QOpenGLVersionProfile()
        vp.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
        vp.setVersion(2, 0)
        self.glfs = self.context().versionFunctions(vp)
        if not self.glfs:
            raise RuntimeError('Failed to retrieve OpenGL function bundle.')
        if not self.glfs.initializeOpenGLFunctions():
            raise RuntimeError('Failed to initialize OpenGL function bundle.')
        self.gl_initializing.emit()

    def paintGL(self):
        print('paintGL(self)')

    def resizeGL(self, w, h):
        print('resizeGL(self, w, h)')
