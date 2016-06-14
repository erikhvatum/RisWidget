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

from contextlib import ExitStack
import numpy
from pathlib import Path
from PyQt5 import Qt
import warnings

NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE = {
    numpy.bool8: Qt.QOpenGLTexture.UInt8,
    numpy.uint8: Qt.QOpenGLTexture.UInt8,
    numpy.uint16: Qt.QOpenGLTexture.UInt16,
    numpy.float32: Qt.QOpenGLTexture.Float32
}
IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT = {
    'G': Qt.QOpenGLTexture.R32F,
    'Ga': Qt.QOpenGLTexture.RG32F,
    'rgb': Qt.QOpenGLTexture.RGB32F,
    'rgba': Qt.QOpenGLTexture.RGBA32F
}
IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT = {
    'G': Qt.QOpenGLTexture.Red,
    'Ga': Qt.QOpenGLTexture.RG,
    'rgb': Qt.QOpenGLTexture.RGB,
    'rgba': Qt.QOpenGLTexture.RGBA
}

def qtransform_to_ndarray(t):
    return numpy.array([
        [t.m11(), t.m21(), t.m31()],
        [t.m12(), t.m22(), t.m32()],
        [t.m13(), t.m23(), t.m33()]])

def ndarray_to_qtransform(a):
    assert a.ndim == 2 and a.shape[0] == 3 and a.shape[1] == 3
    return Qt.QTransform(
        a[0,0], a[1,0], a[2,0],
        a[0,1], a[1,1], a[2,1],
        a[0,2], a[1,2], a[2,2])

_NEXT_QGRAPHICSITEM_USERTYPE = Qt.QGraphicsItem.UserType + 1

def UNIQUE_QGRAPHICSITEM_TYPE():
    """Returns a value to return from QGraphicsItem.type() overrides (which help
    Qt and PyQt return objects of the right type from any call returning QGraphicsItem
    references; for details see http://www.riverbankcomputing.com/pipermail/pyqt/2015-January/035302.html
    and https://bugreports.qt.io/browse/QTBUG-45064)

    This function will not return the same value twice and should be
    used to generate type values for all custom item classes that may
    have instances in the same scene."""
    global _NEXT_QGRAPHICSITEM_USERTYPE
    ret = _NEXT_QGRAPHICSITEM_USERTYPE
    _NEXT_QGRAPHICSITEM_USERTYPE += 1
    return ret

_NEXT_QITEMDATA_ROLE = Qt.Qt.UserRole + 1

def UNIQUE_QITEMDATA_ROLE():
    global _NEXT_QITEMDATA_ROLE
    ret = _NEXT_QITEMDATA_ROLE
    _NEXT_QITEMDATA_ROLE += 1
    return ret

CHOICES_QITEMDATA_ROLE = UNIQUE_QITEMDATA_ROLE()
SPECIAL_SELECTION_HIGHLIGHT_QITEMDATA_ROLE = UNIQUE_QITEMDATA_ROLE()

class NoGLContextIsCurrentError(RuntimeError):
    DEFAULT_MESSAGE = (
        'QOpenGLContext.currentContext() returned None, indicating that no OpenGL '
        'context is current.  This usually indicates that a routine that makes '
        'OpenGL calls was invoked in an unanticipated manner (EG, at-exit execution '
        'of a destructor for an module-level object that wraps an OpenGL primitive).')
    def __init__(self, message=None):
        if message is None:
            message = NoGLContextIsCurrentError.DEFAULT_MESSAGE
        super().__init__(message)

_GL_CACHE = {}

def QGL():
    current_thread = Qt.QThread.currentThread()
    if current_thread is None:
        # We are probably being called by a destructor being called by an at-exit cleanup routine, but too much
        # Qt infrastructure has already been torn down for whatever is calling us to complete its cleanup.
        return
    context = Qt.QOpenGLContext.currentContext()
    if context is None:
        raise NoGLContextIsCurrentError()
    assert current_thread is context.thread()
    # Attempt to return cache entry, a Qt.QOpenGLVersionFunctions object...
    try:
        return _GL_CACHE[context]
    except KeyError:
        pass
    # There is no entry for the current OpenGL context in our cache.  Acquire, cache, and return a
    # Qt.QOpenGLVersionFunctions object.
    try:
        GL = context.versionFunctions()
        if GL is None:
            # Some platforms seem to need version profile specification
            vp = Qt.QOpenGLVersionProfile()
            vp.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
            vp.setVersion(2, 1)
            GL = context.versionFunctions(vp)
    except ImportError:
        # PyQt5 v5.4.0 and v5.4.1 provide access to OpenGL functions up to OpenGL 2.0, but we have made
        # an OpenGL 2.1 context.  QOpenGLContext.versionFunctions(..) will, by default, attempt to return
        # a wrapper around QOpenGLFunctions2_1, which has failed in the try block above.  Therefore,
        # we fall back to explicitly requesting 2.0 functions.  We don't need any of the C _GL 2.1
        # constants or calls, anyway - these address non-square shader uniform transformation matrices and
        # specification of sRGB texture formats, neither of which we use.
        vp = Qt.QOpenGLVersionProfile()
        vp.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
        vp.setVersion(2, 0)
        GL = context.versionFunctions(vp)
    if GL is None:
        raise RuntimeError('Failed to retrieve QOpenGL.')
    if not GL.initializeOpenGLFunctions():
        raise RuntimeError('Failed to initialize OpenGL wrapper namespace.')
    _GL_CACHE[context] = GL
    context.destroyed[Qt.QObject].connect(_on_destruction_of_context_with_cached_gl)
    return GL

def _on_destruction_of_context_with_cached_gl(context):
    del _GL_CACHE[context]

_GL_LOGGERS = {}

def GL_LOGGER():
    context = Qt.QOpenGLContext.currentContext()
    if context is None:
        raise NoGLContextIsCurrentError()
    assert Qt.QThread.currentThread() is context.thread()
    try:
        return _GL_LOGGERS[context]
    except KeyError:
        pass
    gl_logger = Qt.QOpenGLDebugLogger()
    if not gl_logger.initialize():
        raise RuntimeError('Failed to initialize QOpenGLDebugLogger.')
    gl_logger.messageLogged.connect(_on_gl_logger_message)
    context.destroyed.connect(_on_destroyed_context_with_gl_logger)
    gl_logger.enableMessages()
    gl_logger.startLogging(Qt.QOpenGLDebugLogger.SynchronousLogging)
    _GL_LOGGERS[context] = gl_logger
    return gl_logger

_GL_LOGGER_MESSAGE_SEVERITIES = {
    Qt.QOpenGLDebugMessage.InvalidSeverity : 'Invalid',
    Qt.QOpenGLDebugMessage.HighSeverity : 'High',
    Qt.QOpenGLDebugMessage.MediumSeverity : 'Medium',
    Qt.QOpenGLDebugMessage.LowSeverity : 'Low',
    Qt.QOpenGLDebugMessage.NotificationSeverity : 'Notification',
    Qt.QOpenGLDebugMessage.AnySeverity : 'Any'}

_GL_LOGGER_MESSAGE_SOURCES = {
    Qt.QOpenGLDebugMessage.InvalidSource : 'Invalid',
    Qt.QOpenGLDebugMessage.APISource : 'API',
    Qt.QOpenGLDebugMessage.WindowSystemSource : 'WindowSystem',
    Qt.QOpenGLDebugMessage.ShaderCompilerSource : 'ShaderCompiler',
    Qt.QOpenGLDebugMessage.ThirdPartySource : 'ThirdParty',
    Qt.QOpenGLDebugMessage.ApplicationSource : 'Application',
    Qt.QOpenGLDebugMessage.OtherSource : 'Other',
    Qt.QOpenGLDebugMessage.AnySource : 'Any'}

_GL_LOGGER_MESSAGE_TYPES = {
    Qt.QOpenGLDebugMessage.InvalidType : 'Invalid',
    Qt.QOpenGLDebugMessage.ErrorType : 'Error',
    Qt.QOpenGLDebugMessage.DeprecatedBehaviorType : 'DeprecatedBehavior',
    Qt.QOpenGLDebugMessage.UndefinedBehaviorType : 'UndefinedBehavior',
    Qt.QOpenGLDebugMessage.PortabilityType : 'Portability',
    Qt.QOpenGLDebugMessage.PerformanceType : 'Performance',
    Qt.QOpenGLDebugMessage.OtherType : 'Other',
    Qt.QOpenGLDebugMessage.MarkerType : 'Marker',
    Qt.QOpenGLDebugMessage.GroupPushType : 'GroupPush',
    Qt.QOpenGLDebugMessage.GroupPopType : 'GroupPop',
    Qt.QOpenGLDebugMessage.AnyType : 'Any'}

def _on_gl_logger_message(message):
    Qt.qDebug('GL LOG MESSAGE (severity: {}, source: {}, type: {}, GL ID: {}): "{}"'.format(
        _GL_LOGGER_MESSAGE_SEVERITIES[message.severity()],
        _GL_LOGGER_MESSAGE_SOURCES[message.source()],
        _GL_LOGGER_MESSAGE_TYPES[message.type()],
        message.id(),
        message.message()))

def _on_destroyed_context_with_gl_logger(context):
    del _GL_LOGGERS[context]

_GL_EXTS_QUERIED = False
NV_PATH_RENDERING_AVAILABLE = False
NVX_GPU_MEMORY_INFO_AVAILABLE = False

def query_gl_exts():
    global NV_PATH_RENDERING_AVAILABLE
    global NVX_GPU_MEMORY_INFO_AVAILABLE
    if _GL_EXTS_QUERIED:
        return
    try:
        with ExitStack() as estack:
            glw = Qt.QOpenGLWidget()
            estack.callback(glw.deleteLater)
            glf = Qt.QSurfaceFormat()
            glf.setRenderableType(Qt.QSurfaceFormat.OpenGL)
            glf.setVersion(2, 1)
            glf.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
            glf.setSwapBehavior(Qt.QSurfaceFormat.SingleBuffer)
            glf.setStereo(False)
            glf.setSwapInterval(1)
            glw.setFormat(glf)
            glw.show()
            estack.callback(glw.hide)

            if glw.context().hasExtension('GL_NV_path_rendering'.encode('utf-8')):
                try:
                    import OpenGL
                    import OpenGL.GL.NV.path_rendering as PR
                    if PR.glInitPathRenderingNV():
                        NV_PATH_RENDERING_AVAILABLE = True
                except:
                    pass

            if glw.context().hasExtension('GL_NVX_gpu_memory_info'.encode('utf-8')):
                try:
                    import OpenGL
                    import OpenGL.GL.NVX.gpu_memory_info as GMI
                    if GMI.glInitGpuMemoryInfoNVX():
                        NVX_GPU_MEMORY_INFO_AVAILABLE = True
                except:
                    pass
    except:
        warnings.warn('An error occurred while querying OpenGL extension availability.')

_GL_QSURFACE_FORMAT = None

def GL_QSURFACE_FORMAT(msaa_sample_count=None, swap_interval=None):
    global _GL_QSURFACE_FORMAT
    if _GL_QSURFACE_FORMAT is None:
        _GL_QSURFACE_FORMAT = Qt.QSurfaceFormat()
        _GL_QSURFACE_FORMAT.setRenderableType(Qt.QSurfaceFormat.OpenGL)
        _GL_QSURFACE_FORMAT.setVersion(2, 1)
        _GL_QSURFACE_FORMAT.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
        _GL_QSURFACE_FORMAT.setSwapBehavior(Qt.QSurfaceFormat.DoubleBuffer)
        _GL_QSURFACE_FORMAT.setStereo(False)
        if swap_interval is not None:
            _GL_QSURFACE_FORMAT.setSwapInterval(swap_interval)
        if msaa_sample_count is not None:
            _GL_QSURFACE_FORMAT.setSamples(msaa_sample_count)
        if NV_PATH_RENDERING_AVAILABLE:
            _GL_QSURFACE_FORMAT.setStencilBufferSize(4)
        _GL_QSURFACE_FORMAT.setRedBufferSize(8)
        _GL_QSURFACE_FORMAT.setGreenBufferSize(8)
        _GL_QSURFACE_FORMAT.setBlueBufferSize(8)
        _GL_QSURFACE_FORMAT.setAlphaBufferSize(8)
    return _GL_QSURFACE_FORMAT

_freeimage = None

def FREEIMAGE(show_messagebox_on_error=False, error_messagebox_owner=None, is_read=True):
    """If show_messagebox_on_error is true and importing freeimage fails with an exception, a modal QMessageBox is displayed
    describing the error and None is returned.  If show_messagebox_on_error is false and importing freeimage fails with
    and exception, the exception is allowed to propagate."""
    global _freeimage
    if _freeimage is None:
        if show_messagebox_on_error:
            try:
                import freeimage
                _freeimage = freeimage
            except ImportError:
                Qt.QMessageBox.information(
                    error_messagebox_owner,
                    'freeimage-py Module Not Found',
                    """Zach's <a href=https://github.com/zpincus/freeimage-py>freeimage-py module</a> is required for {} image files with RisWidget.
                    Even without freeimage-py, RisWidget accepts image data (<i>rw.image = numpy.zeros((400,400),dtype=numpy.uint8)</i>, for example), but
                    freeimage-py is required if RisWidget is to {} image files on your behalf.""".format(*(('loading', 'load') if is_read else ('saving', 'save'))))
                return
            except RuntimeError as e:
                estr = '\n'.join((
                    "freeimage.py was found, but an error occurred while importing it " + \
                    "(likely because freeimage.so/dylib/dll could not be found):\n",) + e.args)
                Qt.QMessageBox.information(error_messagebox_owner, 'Error While Importing freeimage-py Module', estr)
                return
        else:
            import freeimage
            _freeimage = freeimage
    return _freeimage

_icons = None

def ICONS():
    global _icons
    if _icons is None:
        _icons = {}
        fns = (
            'image_icon.svg',
            'layer_icon.svg',
            'layer_stack_icon.svg',

            'checked_box_icon.svg',
            'disabled_checked_box_icon.svg',

            'pseudo_checked_box_icon.svg',
            'disabled_pseudo_checked_box_icon.svg',

            'unchecked_box_icon.svg',
            'disabled_unchecked_box_icon.svg',

            'wrong_type_checked_box_icon.svg',
            'disabled_wrong_type_checked_box_icon.svg'
        )
        for fn in fns:
            fpath = Path(__file__).parent / 'icons' / fn
            _icons[fpath.stem] = Qt.QIcon(str(fpath))
    return _icons

_FPSD = None
def FPSD():
    global _FPSD
    if _FPSD is None:
        from .qwidgets.fps_display import FPSDisplay
        _FPSD = FPSDisplay()
        _FPSD.show()
    return _FPSD

class _GlQuad:
    def __init__(self):
        if Qt.QOpenGLContext.currentContext() is None:
            raise RuntimeError("A QOpenGLContext must be current when a _GlQuad is instantiated.")
        self.vao = Qt.QOpenGLVertexArrayObject()
        self.vao.create()
        vao_binder = Qt.QOpenGLVertexArrayObject.Binder(self.vao)
        quad = numpy.array([1.1, -1.1,
                            -1.1, -1.1,
                            -1.1, 1.1,
                            1.1, 1.1], dtype=numpy.float32)
        self.buffer = Qt.QOpenGLBuffer(Qt.QOpenGLBuffer.VertexBuffer)
        self.buffer.create()
        self.buffer.bind()
        try:
            self.buffer.setUsagePattern(Qt.QOpenGLBuffer.StaticDraw)
            self.buffer.allocate(quad.ctypes.data, quad.nbytes)
        finally:
            # Note: the following release call is essential.  Without it, if a QPainter is active, QPainter will never work for
            # again for the widget with the active painter!
            self.buffer.release()
        Qt.QApplication.instance().aboutToQuit.connect(self._on_qapplication_about_to_quit)

    def _on_qapplication_about_to_quit(self):
        # Unlike __init__, _on_qapplication_about_to_quit is not called directly by us, and we can not guarantee that
        # an OpenGL context is current
        with ExitStack() as estack:
            if Qt.QOpenGLContext.currentContext() is None:
                offscreen_surface = Qt.QOffscreenSurface()
                offscreen_surface.setFormat(GL_QSURFACE_FORMAT())
                offscreen_surface.create()
                gl_context = Qt.QOpenGLContext()
                gl_context.setShareContext(Qt.QOpenGLContext.globalShareContext())
                gl_context.setFormat(GL_QSURFACE_FORMAT())
                gl_context.create()
                gl_context.makeCurrent(offscreen_surface)
                estack.callback(gl_context.doneCurrent)
            self.vao.destroy()
            self.vao = None
            self.buffer.destroy()
            self.buffer = None

_GL_QUAD = None
def GL_QUAD():
    global _GL_QUAD
    if _GL_QUAD is None:
        _GL_QUAD = _GlQuad()
    return _GL_QUAD