# The MIT License (MIT)
#
# Copyright (c) 2014 Erik Hvatum
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

import ctypes
import math
import numpy
from OpenGL import GL
from OpenGL.arrays import vbo
import OpenGL.GL.ARB.texture_rg
import OpenGL.GL.shaders as GLS
import os
from PyQt5 import QtCore, QtGui, QtWidgets, QtOpenGL
import sip
import sys
import transformations

from ris_widget.ris import Ris
from ris_widget.ris_exceptions import *
from ris_widget.ris_widget_exceptions import *

class RisWidget(QtOpenGL.QGLWidget):
    '''RisWidget stands for Rapid Image Stream Widget.  If tearing is visible, try enabling vsync in your OS's display
    settings.  If that doesn't help, supply True for the enableSwapInterval1_ argument.'''
    def __init__(self, parent_ = None, windowTitle_ = 'RisWidget', enableSwapInterval1_ = False):
        super().__init__(RisWidget._makeGlFormat(enableSwapInterval1_), parent_)
        self.enableSwapInterval1 = enableSwapInterval1_
        self.imTex = None
        self.prevWindowSize = None
        self.currWindowSize = None
        self.windowSizeChanged = None
        self.setWindowTitle(windowTitle_)
        self.iinfo_uint32 = numpy.iinfo(numpy.uint32)
        self.ris = None

    @staticmethod
    def _makeGlFormat(enableSwapInterval1_):
        glFormat = QtOpenGL.QGLFormat()
        # Our weakest target platform is Macmini6,1, having Intel HD 4000 graphics, supporting up to OpenGL 4.1 on OS X.
        # But, we want to use compute shaders, which is just too bad for OS X, because Apple can't be bothered to
        # support GL 4.3.  There's even a change.org petition requesting that Apple support 4.3.  However, that's a bad
        # idea, because it's an unambiguous indication that Apple customers want it - and as we all know, whatsoever
        # Apple customers want, Apple will go to extreme lengths to withhold forever.  Like two button mice.  That took
        # 25 years.  Not unlike fixing the framework relative path bug that existed in NeXT and remained unfixed through
        # OS X 10.3.  So, if you, dear Apple user, aren't used to waiting, you're going to GET used to waiting, and
        # you're going to like it, which is fortunate in that you payed a fortune to do so.
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
        loc = GLS.glGetAttribLocation(self.panelProg, attribName)
        if loc == -1:
            raise RisWidgetException('Could not find location of panel.glslf attribute "{}".'.format(attribName))
        return loc

    def _initPanelProg(self):
        try:
            self.panelProg = GLS.compileProgram(
                GLS.compileShader([self._loadSource('panel.glslv')], GLS.GL_VERTEX_SHADER),
                GLS.compileShader([self._loadSource('panel.glslf')], GLS.GL_FRAGMENT_SHADER))
        except GL.GLError as e:
            print('In panel.glslv or panel.glslf:\n' + e.description.decode('utf-8'))
            sys.exit(-1)

        GLS.glUseProgram(self.panelProg)

        self.panelVao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.panelVao)

        quad = numpy.array([
            # Vertex positions
            1.0, -1.0,
            -1.0, -1.0,
            -1.0, 1.0,
            1.0, 1.0,
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

        self.gtpEnabled = False
        self.gtpMinLoc     = GLS.glGetUniformLocation(self.panelProg, b'gtp.minVal')
        self.gtpMaxLoc     = GLS.glGetUniformLocation(self.panelProg, b'gtp.maxVal')
        self.gtpGammaLoc   = GLS.glGetUniformLocation(self.panelProg, b'gtp.gammaVal')
        self.projectionModelViewMatrixLoc = GLS.glGetUniformLocation(self.panelProg, b'projectionModelViewMatrix')

        self.imPmvMat = None
        self.histoPmvMat = None

        self.panelColorerLoc                    = GL.glGetSubroutineUniformLocation(self.panelProg, GL.GL_FRAGMENT_SHADER, b'panelColorer')
        self.imagePanelGammaTransformColorerLoc = GL.glGetSubroutineIndex(self.panelProg, GL.GL_FRAGMENT_SHADER, b'imagePanelGammaTransformColorer')
        self.imagePanelPassthroughColorerLoc    = GL.glGetSubroutineIndex(self.panelProg, GL.GL_FRAGMENT_SHADER, b'imagePanelPassthroughColorer')
        self.histogramPanelColorerLoc           = GL.glGetSubroutineIndex(self.panelProg, GL.GL_FRAGMENT_SHADER, b'histogramPanelColorer')

        self.qglClearColor(QtGui.QColor(255/3, 255/3, 255/3, 255))

    def _initHistoCalcProg(self):
        try:
            self.histoCalcProg = GLS.compileProgram(GLS.compileShader([self._loadSource('histogramCalc.glslc')], GL.GL_COMPUTE_SHADER))
        except GL.GLError as e:
            print('In histogramCalc.glslc:\n' + e.description.decode('utf-8'))
            sys.exit(-1)
        self.histoBinCountLoc = GLS.glGetUniformLocation(self.histoCalcProg, b'binCount')
        self.histoInvocationRegionSizeLoc = GLS.glGetUniformLocation(self.histoCalcProg, b'invocationRegionSize')
        self.histoImageLoc = 0
        self.histogramBlocksLoc  = 1
        self.histogramBlocksTex = None
        # Hardcode work group count parameter for now
        self.histoWgCountPerAxis = 8
        # This value must match local_size_x and local_size_y in histogramCalc.glslc
        self.histoLiCountPerAxis = 4

    def _initHistoConsolidateProg(self):
        try:
            self.histoConsolidateProg = GLS.compileProgram(GLS.compileShader([self._loadSource('histogramConsolidate.glslc')], GL.GL_COMPUTE_SHADER))
        except GL.GLError as e:
            print('In histogramConsolidate.glslc:\n' + e.description.decode('utf-8'))
            sys.exit(-1)
        self.histoConsolidateHistogramBlocksLoc = 0
        self.histoConsolidateHistogramLoc = 1
        self.histogramTex = None
        self.histoConsolidateExtremaLoc = 0
        self.histoConsolidateBinCountLoc = GLS.glGetUniformLocation(self.histoConsolidateProg, b'binCount')
        self.histoConsolidateInvocationBinCountLoc = GLS.glGetUniformLocation(self.histoConsolidateProg, b'invocationBinCount')
        # This value must match local_size_x in histogramConsolidate.glslc
        self.histoConsolidateLiCount = 16

        GLS.glUseProgram(self.histoConsolidateProg)
        self.histoConsolidateExtremaBuff = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.histoConsolidateExtremaBuff)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, 2 * 4, None, GL.GL_DYNAMIC_COPY)
        self.histoMinMax = None
        self.histo = None

    def _initHistoDrawProg(self):
        try:
            self.histoDrawProg = GLS.compileProgram(
                GLS.compileShader([self._loadSource('histogram.glslv')], GLS.GL_VERTEX_SHADER),
                GLS.compileShader([self._loadSource('histogram.glslf')], GLS.GL_FRAGMENT_SHADER))
        except GL.GLError as e:
            print('In histogram.glslv or histogram.glslf:\n' + e.description.decode('utf-8'))
            sys.exit(-1)
        GLS.glUseProgram(self.histoDrawProg)
        self.histoDrawProjectionModelViewMatrixLoc = GLS.glGetUniformLocation(self.histoDrawProg, b'projectionModelViewMatrix')
        self.histoDrawBinCountLoc = GLS.glGetUniformLocation(self.histoDrawProg, b'binCount')
        self.histoDrawBinScaleLoc = GLS.glGetUniformLocation(self.histoDrawProg, b'binScale')
        self.histoDrawBinIndexLoc = GLS.glGetAttribLocation(self.histoDrawProg, b'binIndex')
        self.histoDrawPointBuff = None
        self.histoDrawPointVao = None
        self.drawHistogramLoc = 0

    def initializeGL(self):
        self._initPanelProg()
        self._initHistoCalcProg()
        self._initHistoConsolidateProg()
        self._initHistoDrawProg()
        self.setBinCount(2048, update=False)

    def _loadImageData(self, imageData, reallocate):
        GLS.glUseProgram(self.panelProg)
        if reallocate and self.imTex is not None:
            GL.glDeleteTextures([self.imTex])
            self.imTex = None

        if self.imTex is None:
            self.imTex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.imTex)
            GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, OpenGL.GL.ARB.texture_rg.GL_R16UI, imageData.shape[1], imageData.shape[0])
        else:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.imTex)

        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, imageData.shape[1], imageData.shape[0], GL.GL_RED_INTEGER, GL.GL_UNSIGNED_SHORT, imageData)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        self.imAspectRatio = imageData.shape[1] / imageData.shape[0]
        self.imSize = type('', (), {'w' : imageData.shape[1], 'h' : imageData.shape[0]})

        GLS.glUseProgram(self.histoCalcProg)

        if reallocate and self.histogramBlocksTex is not None:
            GL.glDeleteTextures([self.histogramBlocksTex])
            self.histogramBlocksTex = None

        if self.histogramBlocksTex is None:
            self.histogramBlocksTex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.histogramBlocksTex)
            GL.glTexStorage3D(
                GL.GL_TEXTURE_2D_ARRAY,
                1,
                OpenGL.GL.ARB.texture_rg.GL_R32UI,
                self.histoWgCountPerAxis, self.histoWgCountPerAxis, self.histoBinCount)
        else:
            GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.histogramBlocksTex)

        # Zero-out block histogram data... this is slow and should be improved
        GL.glTexSubImage3D(
            GL.GL_TEXTURE_2D_ARRAY,
            0,
            0, 0, 0,
            self.histoWgCountPerAxis, self.histoWgCountPerAxis, self.histoBinCount,
            GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT,
            numpy.zeros((self.histoWgCountPerAxis, self.histoWgCountPerAxis, self.histoBinCount), dtype=numpy.uint32))

        if reallocate and self.histogramTex is not None:
            GL.glDeleteTextures([self.histogramTex])
            self.histogramTex = None

        if self.histogramTex is None:
            self.histogramTex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_1D, self.histogramTex)
            GL.glTexStorage1D(
                GL.GL_TEXTURE_1D,
                1,
                OpenGL.GL.ARB.texture_rg.GL_R32UI,
                self.histoBinCount)
        else:
            GL.glBindTexture(GL.GL_TEXTURE_1D, self.histogramTex)

        axisInvocations = self.histoWgCountPerAxis * self.histoLiCountPerAxis
        GL.glUniform2i(self.histoInvocationRegionSizeLoc, math.ceil(imageData.shape[1] / axisInvocations), math.ceil(imageData.shape[0] / axisInvocations))

        # Another pessimal zeroing...
        GL.glTexSubImage1D(
            GL.GL_TEXTURE_1D,
            0,
            0,
            self.histoBinCount,
            GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT,
            numpy.zeros((self.histoBinCount), dtype=numpy.uint32))

    def setBinCount(self, binCount, update=True):
        self.histoBinCount = binCount
        self.context().makeCurrent()

        GL.glProgramUniform1f(self.histoCalcProg, self.histoBinCountLoc, self.histoBinCount)

        GL.glProgramUniform1ui(self.histoConsolidateProg, self.histoConsolidateBinCountLoc, self.histoBinCount)
        GL.glProgramUniform1ui(self.histoConsolidateProg, self.histoConsolidateInvocationBinCountLoc, math.ceil(self.histoBinCount / self.histoConsolidateLiCount))

        GLS.glUseProgram(self.histoDrawProg)

        GL.glUniform1ui(self.histoDrawBinCountLoc, self.histoBinCount)

        if self.histoDrawPointBuff is not None:
            GL.glDeleteVertexArrays(1, [self.histoDrawPointVao])
            GL.glDeleteBuffers(1, [self.histoDrawPointBuff])

        self.histoDrawPointVao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.histoDrawPointVao)

        self.histoDrawPointBuff = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.histoDrawPointBuff)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, numpy.arange(self.histoBinCount, dtype=numpy.float32), GL.GL_STATIC_DRAW)

        GLS.glEnableVertexAttribArray(self.histoDrawBinIndexLoc)
        GLS.glVertexAttribPointer(self.histoDrawBinIndexLoc, 1, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))

        self.histo = numpy.zeros((self.histoBinCount), dtype=numpy.uint32)

        if update:
            self._execHistoCalcProg()
            self._execHistoConsolidateProg()
            self.update()

    def _execPanelProg(self):
        GL.glClearDepth(1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self.imTex is not None:
            GLS.glUseProgram(self.panelProg)
            GL.glBindVertexArray(self.panelVao)

            # Draw image panel
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.imTex)
            GL.glUniformSubroutinesuiv(GL.GL_FRAGMENT_SHADER, 1, [self.imagePanelGammaTransformColorerLoc if self.gtpEnabled else self.imagePanelPassthroughColorerLoc])
            GLS.glUniformMatrix4fv(self.projectionModelViewMatrixLoc, 1, True, self.imPmvMat)
            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)

            # Draw histogram panel
            GL.glUniformSubroutinesuiv(GL.GL_FRAGMENT_SHADER, 1, [self.histogramPanelColorerLoc])
            GLS.glUniformMatrix4fv(self.projectionModelViewMatrixLoc, 1, True, self.histoPmvMat)
            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)

    def _execHistoCalcProg(self):
        GLS.glUseProgram(self.histoCalcProg)

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindImageTexture(self.histoImageLoc, self.imTex, 0, False, 0, GL.GL_READ_ONLY, OpenGL.GL.ARB.texture_rg.GL_R16UI)

        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, 0)
        GL.glBindImageTexture(self.histogramBlocksLoc , self.histogramBlocksTex, 0, True, 0, GL.GL_WRITE_ONLY, OpenGL.GL.ARB.texture_rg.GL_R32UI)

        GL.glDispatchCompute(self.histoWgCountPerAxis, self.histoWgCountPerAxis, 1)

        # Wait for compute shader execution to complete
        GL.glMemoryBarrier(GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    def _execHistoConsolidateProg(self):
        GLS.glUseProgram(self.histoConsolidateProg)

        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, 0)
        GL.glBindImageTexture(self.histoConsolidateHistogramBlocksLoc, self.histogramBlocksTex, 0, True, 0, GL.GL_READ_ONLY, OpenGL.GL.ARB.texture_rg.GL_R32UI)

        GL.glBindTexture(GL.GL_TEXTURE_1D, 0)
        GL.glBindImageTexture(self.histoConsolidateHistogramLoc, self.histogramTex, 0, False, 0, GL.GL_READ_WRITE, OpenGL.GL.ARB.texture_rg.GL_R32UI)

        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.histoConsolidateExtremaBuff)
        if self.histoMinMax is None:
            self.histoMinMax = numpy.array([self.iinfo_uint32.max, self.iinfo_uint32.min], dtype=numpy.uint32)
        else:
            self.histoMinMax[0], self.histoMinMax[1] = self.iinfo_uint32.max, self.iinfo_uint32.min
        GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, self.histoMinMax)

        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, self.histoConsolidateExtremaLoc, self.histoConsolidateExtremaBuff)

        GL.glDispatchCompute(self.histoWgCountPerAxis, self.histoWgCountPerAxis, 1)

        # Wait for compute shader execution to complete
        GL.glMemoryBarrier(GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL.GL_BUFFER_UPDATE_BARRIER_BIT)

        GL.glBindTexture(GL.GL_TEXTURE_1D, self.histogramTex)
        GL.glGetTexImage(GL.GL_TEXTURE_1D, 0, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT, self.histo)

        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.histoConsolidateExtremaBuff)
        GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, self.histoMinMax.nbytes, self.histoMinMax)

    def _execHistoDrawProg(self):
        GLS.glUseProgram(self.histoDrawProg)
        GL.glUniform1ui(self.histoDrawBinCountLoc, self.histoBinCount)
        GL.glUniform1f(self.histoDrawBinScaleLoc, self.histoMinMax[1])
        GLS.glUniformMatrix4fv(self.histoDrawProjectionModelViewMatrixLoc, 1, True, self.histoPmvMat)

        GL.glBindTexture(GL.GL_TEXTURE_1D, 0)
        GL.glBindImageTexture(self.drawHistogramLoc, self.histogramTex, 0, True, 0, GL.GL_READ_ONLY, OpenGL.GL.ARB.texture_rg.GL_R32UI)

        GL.glBindVertexArray(self.histoDrawPointVao)
        GL.glDrawArrays(GL.GL_LINE_STRIP, 0, self.histoBinCount)
        GL.glPointSize(4)
        GL.glDrawArrays(GL.GL_POINTS, 0, self.histoBinCount)

    def getHistogram(self):
        '''Returns a reference to the local memory histogram cache as a numpy array.  This data is
        modified when a new image is shown, so copy the array returned by this function if you want
        to retain it.  NB: the local memory cache of histogram data is copied from the GPU
        immediately after histogram calculation.  NNB: "histogram calculation" is done blockwise,
        so it would be more accurate to say, "after histogram consolidation;" ie, after every
        block's histogram has been summed into a single histogram representing the entire image.'''
        return self.histo

    def getHistogramMinMax(self):
        '''Returns a tuple (min, max) copied from data cached in local memory.'''
        return (self.histoMinMax[0], self.histoMinMax[1])

    def _updateMats(self):
        if self.windowSizeChanged:
            # Image view aspect ratio is always maintained.  The image is centered along whichever axis
            # does not fit.
            imViewAspectRatio = 3/2 * self.windowAspectRatio
            correctionFactor = self.imAspectRatio / imViewAspectRatio
            self.imPmvMat = numpy.dot(
                transformations.translation_matrix([0, 1/3, 0]),
                transformations.scale_matrix(2/3, direction=[0, 1, 0]))
            if correctionFactor <= 1:
                self.imPmvMat = numpy.dot(
                    transformations.scale_matrix(correctionFactor, direction=[1, 0, 0]),
                    self.imPmvMat).astype(numpy.float32)
            else:
                self.imPmvMat = numpy.dot(
                    transformations.scale_matrix(1 / correctionFactor, direction=[0, 1, 0]),
                    self.imPmvMat).astype(numpy.float32)

            # Histogram is always 1/3 window height and fills window width
            self.histoPmvMat = numpy.dot(
                transformations.translation_matrix([0, -2/3, 0]),
                transformations.scale_matrix(1/3, direction=[0, 1, 0]))
            self.histoPmvMat = self.histoPmvMat.astype(numpy.float32)

    def paintGL(self):
        ws = self.size()
        self.currWindowSize = type('', (), {'w' : float(ws.width()), 'h' : float(ws.height())}) # so sue me
        del ws
        self.windowSizeChanged = self.currWindowSize != self.prevWindowSize
        self.windowAspectRatio = self.currWindowSize.w / self.currWindowSize.h

        self._updateMats()
        self._execPanelProg()
        self._execHistoDrawProg()

        self.prevWindowSize = self.currWindowSize

    def resizeGL(self, width, height):
        GL.glViewport(0, 0, width, height)

    def showImage(self, imageData, reallocate=True):
        '''Disable reallocate when replacing an image that differs only by pixel value.  For example,
        do disable reallocate when showing frames from the same video file; do not disable reallocate
        when displaying a random assortment of images of various sizes and color depths. '''
        if type(imageData) is not numpy.ndarray:
            raise TypeError('type(imageData) is not numpy.ndarray')
        if imageData.dtype != numpy.uint16:
            raise TypeError('imageData.dtype != numpy.uint16')
        if imageData.ndim != 2:
            raise ValueError('imageData.ndim != 2')
        self.context().makeCurrent()
        self._loadImageData(imageData, reallocate)
        self._execHistoCalcProg()
        self._execHistoConsolidateProg()
        self.update()

    def attachRis(self, ris, startRis=True, showRisFrames=True):
        if self.ris is not None:
            self.detachRis(self.ris)
        self.ris = ris
        self.showRisFrames = showRisFrames
        self.ris.attachSink(self)
        if startRis:
            self.ris.start()

    def detachRis(self, stopRis=True):
        if stopRis:
            self.ris.stop()
        self.ris.detachSink(self)
        self.ris = None

    def risImageAcquired(self, ris, image):
        pass

    def setGtpEnabled(self, gtpEnabled, update=True):
        '''Enable or disable gamma transformation.  If update is true, the widget will be refreshed immediately.'''
        self.gtpEnabled = gtpEnabled
        if update:
            self.update()

    def setGtpMinimum(self, gtpMinimum, update=True):
        '''Set gamma transformation minimum intensity parameter.  If update is true, the widget will be refreshed immediately.'''
        self.context().makeCurrent()
        GLS.glUseProgram(self.panelProg)
        GLS.glUniform1f(self.gtpMinLoc, gtpMinimum)
        if update:
            self.update()

    def setGtpMaximum(self, gtpMaximum, update=True):
        '''Set gamma transformation minimum intensity parameter.  If update is true, the widget will be refreshed immediately.'''
        self.context().makeCurrent()
        GLS.glUseProgram(self.panelProg)
        GLS.glUniform1f(self.gtpMaxLoc, gtpMaximum)
        if update:
            self.update()

    def setGtpGamma(self, gtpGamma, update=True):
        '''Set gamma transformation gamma parameter.  If update is true, the widget will be refreshed immediately.'''
        self.context().makeCurrent()
        GLS.glUseProgram(self.panelProg)
        GLS.glUniform1f(self.gtpGammaLoc, gtpGamma)
        if update:
            self.update()

    def setGtpParams(self, gtpEnabled, gtpMinimum, gtpMaximum, gtpGamma, update=True):
        '''Set all gamma transformation parameters.  If update is true, the widget will be refreshed immediately.'''
        self.context().makeCurrent()
        GLS.glUseProgram(self.panelProg)
        self.gtpEnabled = gtpEnabled
        GLS.glUniform1f(self.gtpMinLoc, gtpMinimum)
        GLS.glUniform1f(self.gtpMaxLoc, gtpMaximum)
        GLS.glUniform1f(self.gtpGammaLoc, gtpGamma)
        if update:
            self.update()

    def getGtpParams(self):
        '''Returns a dict containing all gamma transformation parameter values.'''
        self.context().makeCurrent()
        ret = {}

        ret['gtpEnabled'] = self.gtpEnabled

        v = numpy.array([None], numpy.float32)
        GLS.glGetUniformfv(self.panelProg, self.gtpMaxLoc, v)
        ret['gtpMaximum'] = v[0]

        v[0] = numpy.nan
        GLS.glGetUniformfv(self.panelProg, self.gtpMinLoc, v)
        ret['gtpMinimum'] = v[0]

        v[0] = numpy.nan
        GLS.glGetUniformfv(self.panelProg, self.gtpGammaLoc, v)
        ret['gtpGamma'] = v[0]

        return ret
