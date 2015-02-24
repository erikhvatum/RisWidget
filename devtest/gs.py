import numpy
from pathlib import Path
from PyQt5 import Qt

class MW(Qt.QMainWindow):
    def __init__(self, parent=None, window_flags=Qt.Qt.WindowFlags(0)):
        super().__init__(parent, window_flags)
        self.iw = ImageWidget(self)
        self.gs = Qt.QGraphicsScene(self)
        c = Qt.QColor(Qt.Qt.red)
        c.setAlphaF(0.5)
        b = Qt.QBrush(c)
        self.gs.addRect(10, 10, 100, 100, Qt.QPen(Qt.Qt.blue), b)
        self.gv = GV(self.gs, self, self.iw)
        self.gv.setViewport(self.iw)
        self.setCentralWidget(self.gv)

class GV(Qt.QGraphicsView):
    def __init__(self, gs, parent, iw):
        super().__init__(gs, parent)
        self.iw = iw
        self.setDragMode(Qt.QGraphicsView.ScrollHandDrag);
        self.setTransformationAnchor(Qt.QGraphicsView.AnchorUnderMouse);
        self.setResizeAnchor(Qt.QGraphicsView.AnchorViewCenter);

    def drawBackground(self, painter, rect):
        self.iw.paint(painter, rect)

class ImageWidget(Qt.QOpenGLWidget):
    def __init__(self, parent):
        super().__init__(parent)
        qsurface_format = Qt.QSurfaceFormat()
        qsurface_format.setRenderableType(Qt.QSurfaceFormat.OpenGL)
        qsurface_format.setVersion(2, 1)
        qsurface_format.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
        qsurface_format.setSwapBehavior(Qt.QSurfaceFormat.DoubleBuffer)
        qsurface_format.setStereo(False)
        qsurface_format.setSwapInterval(1)
        self.setFormat(qsurface_format)
#       self.setMouseTracking(True)
        self._glfs = None
        self._quad_vao = None
        self._quad_buffer = None

    def _init_glfs(self):
        # PyQt5 provides access to OpenGL functions up to OpenGL 2.0, but we have made a 2.1
        # context.  QOpenGLContext.versionFunctions(..) will, by default, attempt to return
        # a wrapper around QOpenGLFunctions2_1, which will fail, as there is no
        # PyQt5._QOpenGLFunctions_2_1 implementation.  Therefore, we explicitly request 2.0
        # functions, and any 2.1 calls that we want to make can not occur through self.glfs.
        vp = Qt.QOpenGLVersionProfile()
        vp.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
        vp.setVersion(2, 0)
        self._glfs = self.context().versionFunctions(vp)
        if not self._glfs:
            raise RuntimeError('Failed to retrieve OpenGL function bundle.')
        if not self._glfs.initializeOpenGLFunctions():
            raise RuntimeError('Failed to initialize OpenGL function bundle.')

    def _build_shader_prog(self, desc, vert_fn, frag_fn):
        source_dpath = Path(__file__).parent.parent / 'shaders'
        prog = Qt.QOpenGLShaderProgram(self)
        if not prog.addShaderFromSourceFile(Qt.QOpenGLShader.Vertex, str(source_dpath / vert_fn)):
            raise RuntimeError('Failed to compile vertex shader "{}" for {} {} shader program.'.format(vert_fn, type(self).__name__, desc))
        if not prog.addShaderFromSourceFile(Qt.QOpenGLShader.Fragment, str(source_dpath / frag_fn)):
            raise RuntimeError('Failed to compile fragment shader "{}" for {} {} shader program.'.format(frag_fn, type(self).__name__, desc))
        if not prog.link():
            raise RuntimeError('Failed to link {} {} shader program.'.format(type(self).__name__, desc))
        return prog

    def _make_quad_vao(self):
        self._quad_vao = Qt.QOpenGLVertexArrayObject()
        self._quad_vao.create()
        quad_vao_binder = Qt.QOpenGLVertexArrayObject.Binder(self._quad_vao)
        quad = numpy.array([1.1, -1.1,
                            -1.1, -1.1,
                            -1.1, 1.1,
                            1.1, 1.1], dtype=numpy.float32)
        self._quad_buffer = Qt.QOpenGLBuffer(Qt.QOpenGLBuffer.VertexBuffer)
        self._quad_buffer.create()
        self._quad_buffer.bind()
        self._quad_buffer.setUsagePattern(Qt.QOpenGLBuffer.StaticDraw)
        self._quad_buffer.allocate(quad.ctypes.data, quad.nbytes)

    def initializeGL(self):
        print('initializeGL')
        self._init_glfs()
        self._glfs.glClearColor(0,0,0,1)
        self._glfs.glClearDepth(1)
        self._glsl_prog_g = self._build_shader_prog('g',
                                                    'image_widget_vertex_shader.glsl',
                                                    'image_widget_fragment_shader_g.glsl')
        self._glsl_prog_ga = self._build_shader_prog('ga',
                                                     'image_widget_vertex_shader.glsl',
                                                     'image_widget_fragment_shader_ga.glsl')
        self._glsl_prog_rgb = self._build_shader_prog('rgb',
                                                      'image_widget_vertex_shader.glsl',
                                                      'image_widget_fragment_shader_rgb.glsl')
        self._glsl_prog_rgba = self._build_shader_prog('rgba',
                                                       'image_widget_vertex_shader.glsl',
                                                       'image_widget_fragment_shader_rgba.glsl')
        self._image_type_to_glsl_prog = {'g'   : self._glsl_prog_g,
                                         'ga'  : self._glsl_prog_ga,
                                         'rgb' : self._glsl_prog_rgb,
                                         'rgba': self._glsl_prog_rgba}
        self._make_quad_vao()

    def paintGL(self):
        print('paintGL')

    def resizeGL(self, x, y):
        print('resizeGL')

    def paint(self, painter, rect):
        painter.beginNativePainting()
        self._glfs.glClearColor(0,0,0,1)
        self._glfs.glClearDepth(1)
        self._glfs.glClear(self._glfs.GL_COLOR_BUFFER_BIT | self._glfs.GL_DEPTH_BUFFER_BIT)
        painter.endNativePainting()
