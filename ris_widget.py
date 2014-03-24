import ctypes
import numpy
from OpenGL import GL
from OpenGL.arrays import vbo
import OpenGL.GL.ARB.texture_rg
import OpenGL.GL.shaders as GLS
import os
from PyQt5 import QtCore, QtGui, QtWidgets, QtOpenGL
import sip

from ris_widget_exceptions import *

class RisWidget(QtOpenGL.QGLWidget):
    '''RisWidget stands for Rapid Image Stream Widget.  If tearing is visible, try enabling vsync in your OS's display
    settings.  If that doesn't help, supply True for the enableSwapInterval1_ argument.'''
    def __init__(self, parent_ = None, windowTitle_ = 'RisWidget', enableSwapInterval1_ = False):
        super().__init__(RisWidget._makeGlFormat(enableSwapInterval1_), parent_)
        self.enableSwapInterval1 = enableSwapInterval1_
        self.imTex = None
        self.prevWindowSize = None
        self.setWindowTitle(windowTitle_)

    @staticmethod
    def _makeGlFormat(enableSwapInterval1_):
        glFormat = QtOpenGL.QGLFormat()
        # Our weakest target platform is Macmini6,1 having Intel HD 4000 graphics supporting up to OpenGL 4.1 on OS X
        glFormat.setVersion(4, 3)
        # We avoid relying on depcrecated fixed-function pipeline functionality; any attempt to use legacy OpenGL calls
        # should fail.
        glFormat.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        # It's highly likely that enabling swap interval 1 will not ameliorate tearing: any display supporting GL 4.1
        # supports double buffering, and tearing should not be visible with double buffering.  Therefore, the tearing
        # is caused by vsync being off or some fundamental brain damage in your out-of-date X11 display server; further
        # breaking things with swap interval won't help.  But, perhaps you can manage to convince yourself that it's
        # tearing less, and by the simple expedient of displaying less, it will be.
        glFormat.setSwapInterval(enableSwapInterval1_)
        # Want hardware rendering (should be enabled by default, but this can't hurt)
        glFormat.setDirectRendering(True)
        # Likewise, double buffering should be enabled by default
        glFormat.setDoubleBuffer(True)
        return glFormat

    def _loadSource(self, sourceFileName):
        with open(os.path.join(os.path.dirname(__file__), sourceFileName), 'r') as src:
            return src.read()

    def _getAttribLoc(self, attribName):
        loc = GLS.glGetAttribLocation(self.shaderProgram, attribName)
        if loc == -1:
            raise RisWidgetException('Could not find location of panel.glslf attribute "{}".'.format(attribName))
        return loc

    def initializeGL(self):
        self.shaderProgram = GLS.compileProgram(
            GLS.compileShader([self._loadSource('panel.glslv')], GLS.GL_VERTEX_SHADER),
            GLS.compileShader([self._loadSource('image.glslf')], GLS.GL_FRAGMENT_SHADER))
        GLS.glUseProgram(self.shaderProgram)

        self.panelVao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.panelVao)

        quad = numpy.array([
            # Vertex positions
            0.98, -0.98,
            -0.98, -0.98,
            -0.98, 0.98,
            0.98, 0.98,
            # Texture coordinates
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0], numpy.float32)

        self.quadShaderBuff = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.quadShaderBuff)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad, GL.GL_STATIC_DRAW)

        # quad is deleted to make clear that quad's data has been copied to the GPU by the glBufferData directly above
        del quad

        vertPosLoc = self._getAttribLoc('vertPos')
        GLS.glEnableVertexAttribArray(vertPosLoc)
        GLS.glVertexAttribPointer(vertPosLoc, 2, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))

        texCoordLoc = self._getAttribLoc('texCoord')
        GLS.glEnableVertexAttribArray(texCoordLoc)
        GLS.glVertexAttribPointer(texCoordLoc, 2, GL.GL_FLOAT, False, 0, ctypes.c_void_p(2 * 4 * 4))

        checkerboard = numpy.array([
            [0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000],
            [0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff],
            [0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000],
            [0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff],
            [0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000],
            [0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff],
            [0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000],
            [0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff]], numpy.uint16)

        self.imTex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.imTex)

        GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, OpenGL.GL.ARB.texture_rg.GL_R16UI, checkerboard.shape[1], checkerboard.shape[0])
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, checkerboard.shape[1], checkerboard.shape[0], GL.GL_RED_INTEGER, GL.GL_UNSIGNED_SHORT, checkerboard)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        self.gtpEnabledLoc = GLS.glGetUniformLocation(self.shaderProgram, b'gtp.isEnabled')
        self.gtpMinLoc     = GLS.glGetUniformLocation(self.shaderProgram, b'gtp.minVal')
        self.gtpMaxLoc     = GLS.glGetUniformLocation(self.shaderProgram, b'gtp.maxVal')
        self.gtpGammaLoc   = GLS.glGetUniformLocation(self.shaderProgram, b'gtp.gammaVal')
        self.projectionModelViewMatrixLoc = GLS.glGetUniformLocation(self.shaderProgram, b'projectionModelViewMatrix')

        self.qglClearColor(QtGui.QColor(255/3, 255/3, 255/3, 255))

#       samplers = []
#       GL.glGenSamplers(1, samplers)

    def paintGL(self):
        GL.glClearDepth(1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self.imTex is not None:
            GLS.glUseProgram(self.shaderProgram)
            GL.glBindVertexArray(self.panelVao)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.imTex)

            # Rescale projection matrix such that display aspect ratio is preserved and contents fill either horizontal
            # or vertical (whichever is smaller)
            ws = self.size()
            if ws != self.prevWindowSize:
                wsw = float(ws.width())
                wsh = float(ws.height())
                pmv = numpy.identity(4, numpy.float32)
                if wsw >= wsh:
                    pmv[0, 0] = wsh/wsw
                else:
                    pmv[1, 1] = wsw/wsh
                GLS.glUniformMatrix4fv(self.projectionModelViewMatrixLoc, 1, True, pmv)
                self.prevWindowSize = ws

            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)

    def resizeGL(self, width, height):
        GL.glViewport(0, 0, width, height)

    def showImage(self, imageData, recycleTexture = False):
        '''Enable rycleTexture when replacing an image that differs only by pixel value.  For example,
        do enable recycleTexture when showing frames from the same video file; do not enable recycleTexture
        when displaying a random assortment of images of various sizes and color depths. '''
        if type(imageData) is not numpy.ndarray:
            raise TypeError('type(imageData) is not numpy.ndarray')
        if imageData.dtype != numpy.uint16:
            raise TypeError('imageData.dtype != numpy.uint16')
        if imageData.ndim != 2:
            raise ValueError('imageData.ndim != 2')

        self.context().makeCurrent()
        GLS.glUseProgram(self.shaderProgram)

        if not recycleTexture and self.imTex is not None:
            GL.glDeleteTextures([self.imTex])
            self.imTex = None

        if self.imTex is None:
            self.imTex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.imTex)
            GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, OpenGL.GL.ARB.texture_rg.GL_R16UI, imageData.shape[1], imageData.shape[0])
        else:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.imTex)

        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, imageData.shape[1], imageData.shape[0], GL.GL_RED_INTEGER, GL.GL_UNSIGNED_SHORT, imageData)
#       GL.glTexImage2Dui(GL.GL_TEXTURE_2D, 0, OpenGL.GL.ARB.texture_rg.GL_R16UI, imageData.shape[1], imageData.shape[0], 0, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_SHORT, imageData)
#       GL.glTexParameteriv(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_RGBA, [GL.GL_RED, GL.GL_RED, GL.GL_RED, GL.GL_ONE])
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
#       GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

        self.update()

    def setGtpEnabled(self, gtpEnabled, update=True):
        '''Enable or disable gamma transformation.  If update is true, the widget will be refreshed.'''
        self.context().makeCurrent()
        GLS.glUseProgram(self.shaderProgram)
        GLS.glUniform1i(self.gtpEnabledLoc, gtpEnabled)
        if update:
            self.update()

    def setGtpMinimum(self, gtpMinimum, update=True):
        '''Set gamma transformation minimum intensity parameter.  If update is true, the widget will be refreshed.'''
        self.context().makeCurrent()
        GLS.glUseProgram(self.shaderProgram)
        GLS.glUniform1f(self.gtpMinLoc, gtpMinimum)
        if update:
            self.update()

    def setGtpMaximum(self, gtpMaximum, update=True):
        '''Set gamma transformation minimum intensity parameter.  If update is true, the widget will be refreshed.'''
        self.context().makeCurrent()
        GLS.glUseProgram(self.shaderProgram)
        GLS.glUniform1f(self.gtpMaxLoc, gtpMaximum)
        if update:
            self.update()

    def setGtpGamma(self, gtpGamma, update=True):
        '''Set gamma transformation gamma parameter.  If update is true, the widget will be refreshed.'''
        self.context().makeCurrent()
        GLS.glUseProgram(self.shaderProgram)
        GLS.glUniform1f(self.gtpGammaLoc, gtpGamma)
        if update:
            self.update()

    def setGtpParams(self, gtpEnabled, gtpMinimum, gtpMaximum, gtpGamma, update=True):
        '''Set all gamma transformation parameters.  If update is true, the widget will be refreshed.'''
        self.context().makeCurrent()
        GLS.glUseProgram(self.shaderProgram)
        GLS.glUniform1i(self.gtpEnabledLoc, gtpEnabled)
        GLS.glUniform1f(self.gtpMinLoc, gtpMinimum)
        GLS.glUniform1f(self.gtpMaxLoc, gtpMaximum)
        GLS.glUniform1f(self.gtpGammaLoc, gtpGamma)
        if update:
            self.update()

    def getGtpParams(self):
        '''Returns a dict containing all gamma transformation parameter values.'''
        self.context().makeCurrent()
        ret = {}

        v = numpy.array([-1], numpy.int32)
        GLS.glGetUniformiv(self.shaderProgram, self.gtpEnabledLoc, v)
        ret['gtpEnabled'] = True if v[0] == 1 else False

        v = numpy.array([None], numpy.float32)
        GLS.glGetUniformfv(self.shaderProgram, self.gtpMaxLoc, v)
        ret['gtpMaximum'] = v[0]

        v[0] = numpy.nan
        GLS.glGetUniformfv(self.shaderProgram, self.gtpMinLoc, v)
        ret['gtpMinimum'] = v[0]

        v[0] = numpy.nan
        GLS.glGetUniformfv(self.shaderProgram, self.gtpGammaLoc, v)
        ret['gtpGamma'] = v[0]

        return ret
