from PyQt5 import Qt

class GLW(Qt.QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        qsurface_format = Qt.QSurfaceFormat()
        qsurface_format.setRenderableType(Qt.QSurfaceFormat.OpenGL)
        qsurface_format.setVersion(2, 1)
        qsurface_format.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
        qsurface_format.setSwapBehavior(Qt.QSurfaceFormat.DoubleBuffer)
        qsurface_format.setStereo(False)
        qsurface_format.setSwapInterval(1)
        self.setFormat(qsurface_format)

    def initializeGL(self):
        vp = Qt.QOpenGLVersionProfile()
        vp.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
        vp.setVersion(2, 0)
        self._glfs = self.context().versionFunctions(vp)
        if not self._glfs:
            raise RuntimeError('Failed to retrieve OpenGL function bundle.')
        if not self._glfs.initializeOpenGLFunctions():
            raise RuntimeError('Failed to initialize OpenGL function bundle.')

    def paint(self, p, rect):
        p.beginNativePainting()
        self._glfs.glClearColor(0,0,0,1)
        self._glfs.glClearDepth(1)
        self._glfs.glClear(self._glfs.GL_COLOR_BUFFER_BIT | self._glfs.GL_DEPTH_BUFFER_BIT)
        p.endNativePainting()

#       color = Qt.QColor(Qt.Qt.red)
#       color.setAlphaF(0.5)
#       brush = Qt.QBrush(color)
#       p.setBrush(brush)
#       p.drawRect(10, 10, 100, 100)

    def paintGL(self):
        pass

    def resizeGL(self, w, h):
        pass

class GV(Qt.QGraphicsView):
    def __init__(self, gs, parent, glw):
        super().__init__(gs, parent)
        self.glw = glw
        self.setDragMode(Qt.QGraphicsView.ScrollHandDrag);
        self.setTransformationAnchor(Qt.QGraphicsView.AnchorUnderMouse);
        self.setResizeAnchor(Qt.QGraphicsView.AnchorViewCenter);

    def drawBackground(self, painter, rect):
        self.glw.paint(painter, rect)

if __name__ == '__main__':
    import sys
    app = Qt.QApplication(sys.argv)
    mw = Qt.QMainWindow()

    gs = Qt.QGraphicsScene()
    c = Qt.QColor(Qt.Qt.red)
    c.setAlphaF(0.5)
    b = Qt.QBrush(c)
    gs.addRect(10, 10, 100, 100, Qt.QPen(Qt.Qt.blue), b)

    glw = GLW()

    gv = GV(gs, mw, glw)
    gv.setViewport(glw)

    mw.setCentralWidget(gv)
    mw.show()

    app.exec()
