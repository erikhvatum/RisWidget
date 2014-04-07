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
import time
import transformations

from ris_widget.ris import Ris
from ris_widget.ris_exceptions import *
from ris_widget.ris_widget_exceptions import *
from ris_widget.shader_program import ShaderProgram
from ris_widget.size import Size

class RisWidget(QtOpenGL.QGLWidget):
    '''RisWidget stands for Rapid Image Stream Widget.  If tearing is visible, try enabling vsync in your OS's display
    settings.  If that doesn't help, supply True for the enableSwapInterval1_ argument.'''
    def __init__(self, parent_ = None, windowTitle_ = 'RisWidget', enableSwapInterval1_ = False):
        super().__init__(RisWidget._makeGlFormat(enableSwapInterval1_), parent_)
        self.enableSwapInterval1 = enableSwapInterval1_
        self.prevWindowSize = None
        self.currWindowSize = None
        self.windowSizeChanged = None
        self.setWindowTitle(windowTitle_)
        self.iinfo_uint32 = numpy.iinfo(numpy.uint32)
        self.ris = None
        self.showRisFrames = None
        self.timeAtLastFrameEnd = None

        ##TODO REMOVE
        self.debugPrintFps = False

        self.image = None
        self.imageSize = None
        self.prevImageSize = None
        self.imageSizeChanged = None
        self.imageAspectRatio = None
        self.histogramIsStale = True
        self.histogramDataStructuresAreStale = True
        self.histogramBinCount = 2048
        self.histogramBlocks = None
        self.histogram = None
        self.histogramData = None
        self.imagePmv = None
        self.histogramPmv = None

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

    def _initHistoCalcProg(self):
        self.histoCalcProg = ShaderProgram(
            self.context(),
            [('histogramCalc.glslc', GL.GL_COMPUTE_SHADER)],
            [
                'binCount',
                'invocationRegionSize'
            ]
        )
        self.histoCalcProg.imageLoc = 0
        self.histoCalcProg.blocksLoc = 1
        # Hardcode work group count parameter for now
        self.histoCalcProg.wgCountPerAxis = 8
        # This value must match local_size_x and local_size_y in histogramCalc.glslc
        self.histoCalcProg.liCountPerAxis = 4

    def _initHistoConsolidateProg(self):
        self.histoConsolidateProg = ShaderProgram(
            self.context(),
            [('histogramConsolidate.glslc', GL.GL_COMPUTE_SHADER)],
            [
                'binCount',
                'invocationBinCount'
            ]
        )
        self.histoConsolidateProg.blocksLoc = 0
        self.histoConsolidateProg.histogramLoc = 1
        self.histoConsolidateProg.extremaLoc = 0
        # This value must match local_size_x in histogramConsolidate.glslc
        self.histoConsolidateProg.liCount = 16

        self.histoConsolidateProg.extremaBuff = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.histoConsolidateProg.extremaBuff)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, 2 * 4, None, GL.GL_DYNAMIC_COPY)
        self.histoConsolidateProg.extrema = numpy.ndarray((2), numpy.uint32)

    def _initPanelProg(self):
        self.panelProg = ShaderProgram(
            self.context(),
            [
                ('panel.glslv', GLS.GL_VERTEX_SHADER),
                ('panel.glslf', GLS.GL_FRAGMENT_SHADER,
                    ('panelColorer',
                     'imagePanelGammaTransformColorer',
                     'imagePanelPassthroughColorer',
                     'histogramPanelColorer'))
            ],
            [
                ('gtp.minVal', 'gtpMin'),
                ('gtp.maxVal', 'gtpMax'),
                ('gtp.gammaVal', 'gtpGamma'),
                'projectionModelViewMatrix'
            ],
            [
                'vertPos',
                'texCoord'
            ]
        )

        self.panelProg.gtpEnabled = False

        with self.panelProg:
            self.panelProg.vao = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(self.panelProg.vao)

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

            GLS.glEnableVertexAttribArray(self.panelProg.vertPosLoc)
            GLS.glVertexAttribPointer(self.panelProg.vertPosLoc, 2, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))

            GLS.glEnableVertexAttribArray(self.panelProg.texCoordLoc)
            GLS.glVertexAttribPointer(self.panelProg.texCoordLoc, 2, GL.GL_FLOAT, False, 0, ctypes.c_void_p(2 * 4 * 4))

    def _initHistoDrawProg(self):
        self.histoDrawProg = ShaderProgram(
            self.context(),
            [
                ('histogram.glslv', GLS.GL_VERTEX_SHADER),
                ('histogram.glslf', GLS.GL_FRAGMENT_SHADER)
            ],
            [
                'projectionModelViewMatrix',
                'binCount',
                'binScale'
            ],
            [   
                'binIndex'
            ]
        )
        self.histoDrawProg.pointBuff = None
        self.histoDrawProg.pointVao = None
        self.histoDrawProg.histogramLoc = 0

    def initializeGL(self):
        self.qglClearColor(QtGui.QColor(255/3, 255/3, 255/3, 255))
        self._initPanelProg()
        self._initHistoCalcProg()
        self._initHistoConsolidateProg()
        self._initHistoDrawProg()
        checkerboard = numpy.zeros((1600), dtype=numpy.uint16)
        f = 65535 / 1599
        a = True
        i = 0
        for r in range(40):
            for c in range(40):
                if a:
                    checkerboard[i] = round(i * f)
                a = not a
                i += 1
            a = not a
        checkerboard = checkerboard.reshape((40,40))
        #checkerboard = numpy.tile(numpy.array([[0xffff, 0x0000],
        #                                       [0x0000, 0xffff]], dtype=numpy.uint16), (8, 8))
        self.showImage(checkerboard, filterTexture=False)

    def setBinCount(self, binCount, update=True):
        self.histogramBinCount = binCount
        self.histogramIsStale = True
        self.histogramDataStructuresAreStale = True
        if update:
            self.update()

    def _execHistoCalcProg(self):
        with self.histoCalcProg:
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glBindImageTexture(self.histoCalcProg.imageLoc, self.image, 0, False, 0, GL.GL_READ_ONLY, OpenGL.GL.ARB.texture_rg.GL_R16UI)

            GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, 0)
            GL.glBindImageTexture(self.histoCalcProg.blocksLoc, self.histogramBlocks, 0, True, 0, GL.GL_WRITE_ONLY, OpenGL.GL.ARB.texture_rg.GL_R32UI)

            GL.glDispatchCompute(self.histoCalcProg.wgCountPerAxis, self.histoCalcProg.wgCountPerAxis, 1)

            # Wait for compute shader execution to complete
            GL.glMemoryBarrier(GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    def _execHistoConsolidateProg(self):
        with self.histoConsolidateProg:
            GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, 0)
            GL.glBindImageTexture(self.histoConsolidateProg.blocksLoc, self.histogramBlocks, 0, True, 0, GL.GL_READ_ONLY, OpenGL.GL.ARB.texture_rg.GL_R32UI)

            GL.glBindTexture(GL.GL_TEXTURE_1D, 0)
            GL.glBindImageTexture(self.histoConsolidateProg.histogramLoc, self.histogram, 0, False, 0, GL.GL_READ_WRITE, OpenGL.GL.ARB.texture_rg.GL_R32UI)

            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.histoConsolidateProg.extremaBuff)
            self.histoConsolidateProg.extrema[0], self.histoConsolidateProg.extrema[1] = self.iinfo_uint32.max, self.iinfo_uint32.min
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, self.histoConsolidateProg.extrema)

            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, self.histoConsolidateProg.extremaLoc, self.histoConsolidateProg.extremaBuff)

            GL.glDispatchCompute(self.histoCalcProg.wgCountPerAxis, self.histoCalcProg.wgCountPerAxis, 1)

            # Wait for compute shader execution to complete
            GL.glMemoryBarrier(GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL.GL_BUFFER_UPDATE_BARRIER_BIT)

            GL.glBindTexture(GL.GL_TEXTURE_1D, self.histogram)
            GL.glGetTexImage(GL.GL_TEXTURE_1D, 0, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT, self.histogramData)

            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.histoConsolidateProg.extremaBuff)
            GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, self.histoConsolidateProg.extrema.nbytes, self.histoConsolidateProg.extrema)

    def _execPanelProg(self):
        with self.panelProg:
            GL.glBindVertexArray(self.panelProg.vao)

            # Draw image panel
            if self.panelProg.gtpEnabled:
                sub = self.panelProg.imagePanelGammaTransformColorerLoc
            else:
                sub = self.panelProg.imagePanelPassthroughColorerLoc
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.image)
            GL.glUniformSubroutinesuiv(GL.GL_FRAGMENT_SHADER, 1, [sub])
            GLS.glUniformMatrix4fv(self.panelProg.projectionModelViewMatrixLoc, 1, True, self.imagePmv)
            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)

            # Draw histogram panel
            GL.glUniformSubroutinesuiv(GL.GL_FRAGMENT_SHADER, 1, [self.panelProg.histogramPanelColorerLoc])
            GLS.glUniformMatrix4fv(self.panelProg.projectionModelViewMatrixLoc, 1, True, self.histogramPmv)
            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)

    def _execHistoDrawProg(self):
        with self.histoDrawProg:
            GL.glUniform1ui(self.histoDrawProg.binCountLoc, self.histogramBinCount)
            GL.glUniform1f(self.histoDrawProg.binScaleLoc, self.histoConsolidateProg.extrema[1])
            GLS.glUniformMatrix4fv(self.histoDrawProg.projectionModelViewMatrixLoc, 1, True, self.histogramPmv)

            GL.glBindTexture(GL.GL_TEXTURE_1D, 0)
            GL.glBindImageTexture(self.histoDrawProg.histogramLoc, self.histogram, 0, True, 0, GL.GL_READ_ONLY, OpenGL.GL.ARB.texture_rg.GL_R32UI)

            GL.glBindVertexArray(self.histoDrawProg.pointVao)
            GL.glDrawArrays(GL.GL_LINE_STRIP, 0, self.histogramBinCount)
            GL.glPointSize(4)
            GL.glDrawArrays(GL.GL_POINTS, 0, self.histogramBinCount)

    def getHistogram(self):
        '''Returns a reference to the local memory histogram cache as a numpy array.  This data is
        modified when a new image is shown, so copy the array returned by this function if you want
        to retain it.  NB: the local memory cache of histogram data is copied from the GPU
        immediately after histogram calculation.  NNB: "histogram calculation" is done blockwise,
        so it would be more accurate to say, "after histogram consolidation;" ie, after every
        block's histogram has been summed into a single histogram representing the entire image.'''
        return self.histogramData

    def getHistogramMinMax(self):
        '''Returns a tuple (min, max) copied from data cached in local memory.'''
        return (self.histoConsolidateProg.extrema[0], self.histoConsolidateProg.extrema[1])

    def _updateHisto(self):
        if self.histogramIsStale:
            if self.histogramDataStructuresAreStale and self.histogramBlocks is not None:
                GL.glDeleteTextures([self.histogramBlocks])
                self.histogramBlocks = None

            if self.histogramBlocks is None:
                self.histogramBlocks = GL.glGenTextures(1)
                GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.histogramBlocks)
                GL.glTexStorage3D(
                    GL.GL_TEXTURE_2D_ARRAY,
                    1,
                    OpenGL.GL.ARB.texture_rg.GL_R32UI,
                    self.histoCalcProg.wgCountPerAxis, self.histoCalcProg.wgCountPerAxis, self.histogramBinCount)
            else:
                GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.histogramBlocks)

            # Zero-out block histogram data... this is slow and should be improved
            GL.glTexSubImage3D(
                GL.GL_TEXTURE_2D_ARRAY,
                0,
                0, 0, 0,
                self.histoCalcProg.wgCountPerAxis, self.histoCalcProg.wgCountPerAxis, self.histogramBinCount,
                GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT,
                numpy.zeros((self.histoCalcProg.wgCountPerAxis, self.histoCalcProg.wgCountPerAxis, self.histogramBinCount), dtype=numpy.uint32))

            if self.histogramDataStructuresAreStale and self.histogram is not None:
                GL.glDeleteTextures([self.histogram])
                self.histogram = None

            if self.histogram is None:
                self.histogram = GL.glGenTextures(1)
                GL.glBindTexture(GL.GL_TEXTURE_1D, self.histogram)
                GL.glTexStorage1D(
                    GL.GL_TEXTURE_1D,
                    1,
                    OpenGL.GL.ARB.texture_rg.GL_R32UI,
                    self.histogramBinCount)
            else:
                GL.glBindTexture(GL.GL_TEXTURE_1D, self.histogram)

            if self.histogramDataStructuresAreStale:
                axisInvocations = self.histoCalcProg.wgCountPerAxis * self.histoCalcProg.liCountPerAxis
                GL.glProgramUniform2i(self.histoCalcProg.prog, self.histoCalcProg.invocationRegionSizeLoc, math.ceil(self.imageSize.w / axisInvocations), math.ceil(self.imageSize.h / axisInvocations))
                GL.glProgramUniform1f(self.histoCalcProg.prog, self.histoCalcProg.binCountLoc, self.histogramBinCount)
                GL.glProgramUniform1ui(self.histoConsolidateProg.prog, self.histoConsolidateProg.binCountLoc, self.histogramBinCount)
                GL.glProgramUniform1ui(self.histoConsolidateProg.prog, self.histoConsolidateProg.invocationBinCountLoc, math.ceil(self.histogramBinCount / self.histoConsolidateProg.liCount))
                with self.histoDrawProg:
                    GL.glUniform1ui(self.histoDrawProg.binCountLoc, self.histogramBinCount)

                    if self.histoDrawProg.pointBuff is not None:
                        GL.glDeleteVertexArrays(1, [self.histoDrawProg.pointVao])
                        GL.glDeleteBuffers(1, [self.histoDrawProg.pointBuff])

                    self.histoDrawProg.pointVao = GL.glGenVertexArrays(1)
                    GL.glBindVertexArray(self.histoDrawProg.pointVao)

                    self.histoDrawProg.pointBuff = GL.glGenBuffers(1)
                    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.histoDrawProg.pointBuff)
                    GL.glBufferData(GL.GL_ARRAY_BUFFER, numpy.arange(self.histogramBinCount, dtype=numpy.float32), GL.GL_STATIC_DRAW)

                    GLS.glEnableVertexAttribArray(self.histoDrawProg.binIndexLoc)
                    GLS.glVertexAttribPointer(self.histoDrawProg.binIndexLoc, 1, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))

            # Another pessimal zeroing...
            GL.glTexSubImage1D(
                GL.GL_TEXTURE_1D,
                0,
                0,
                self.histogramBinCount,
                GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT,
                numpy.zeros((self.histogramBinCount), dtype=numpy.uint32))

            self.histogramData = numpy.zeros((self.histogramBinCount), dtype=numpy.uint32)

            self.histogramDataStructuresAreStale = False
            self._execHistoCalcProg()
            self._execHistoConsolidateProg()
            self.histogramIsStale = False

    def _updateMats(self):
        if self.windowSizeChanged or self.imageSizeChanged:
            # Image view aspect ratio is always maintained.  The image is centered along whichever axis
            # does not fit.
            imViewAspectRatio = 3/2 * self.windowAspectRatio
            correctionFactor = self.imageAspectRatio / imViewAspectRatio
            self.imagePmv = numpy.dot(
                transformations.translation_matrix([0, 1/3, 0]),
                transformations.scale_matrix(2/3, direction=[0, 1, 0]))
            if correctionFactor <= 1:
                self.imagePmv = numpy.dot(
                    transformations.scale_matrix(correctionFactor, direction=[1, 0, 0]),
                    self.imagePmv).astype(numpy.float32)
            else:
                self.imagePmv = numpy.dot(
                    transformations.scale_matrix(1 / correctionFactor, direction=[0, 1, 0]),
                    self.imagePmv).astype(numpy.float32)

            # Histogram is always 1/3 window height and fills window width
            self.histogramPmv = numpy.dot(
                transformations.translation_matrix([0, -2/3, 0]),
                transformations.scale_matrix(1/3, direction=[0, 1, 0]))
            self.histogramPmv = self.histogramPmv.astype(numpy.float32)

    def paintGL(self):
        GL.glClearDepth(1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self.image is not None:
            ws = self.size()
            self.currWindowSize = Size(float(ws.width()), float(ws.height()))
            del ws
            self.windowSizeChanged = self.currWindowSize != self.prevWindowSize
            self.windowAspectRatio = self.currWindowSize.w / self.currWindowSize.h
            self.imageSizeChanged = self.imageSize != self.prevImageSize

            self._updateHisto()
            self._updateMats()
            self._execPanelProg()
            self._execHistoDrawProg()

            self.prevWindowSize = self.currWindowSize
            self.prevImageSize = self.imageSize

            t = time.time()
            if self.debugPrintFps:
                if self.timeAtLastFrameEnd is not None:
                    d = t - self.timeAtLastFrameEnd
                    print('{}s\t{}fps'.format(d, 1/d))
            self.timeAtLastFrameEnd = t

    def resizeGL(self, width, height):
        GL.glViewport(0, 0, width, height)

    def _loadImageData(self, imageData, filterTexture):
        self.imageAspectRatio = imageData.shape[1] / imageData.shape[0]
        newImageSize = Size(imageData.shape[1], imageData.shape[0])
        reallocate = newImageSize != self.imageSize
        self.imageSize = newImageSize

        self.histogramIsStale = True
        self.histogramDataStructuresAreStale = reallocate

        if reallocate and self.image is not None:
            GL.glDeleteTextures([self.image])
            self.image = None

        if self.image is None:
            self.image = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.image)
            GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, OpenGL.GL.ARB.texture_rg.GL_R16UI, imageData.shape[1], imageData.shape[0])
        else:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.image)

        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, imageData.shape[1], imageData.shape[0], GL.GL_RED_INTEGER, GL.GL_UNSIGNED_SHORT, imageData)
        filterType = GL.GL_LINEAR if filterTexture else GL.GL_NEAREST
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, filterType)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, filterType)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

    def showImage(self, imageData, filterTexture=True):
        '''filterTexture controls whether filtering is applied when rendering an image that
        does not have 1 to 1 texture pixel to screen pixel correspondence.'''
        if type(imageData) is not numpy.ndarray:
            raise TypeError('type(imageData) is not numpy.ndarray')
        if imageData.dtype != numpy.uint16:
            raise TypeError('imageData.dtype != numpy.uint16')
        if imageData.ndim != 2:
            raise ValueError('imageData.ndim != 2')
        self.context().makeCurrent()
        self._loadImageData(imageData, filterTexture)
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

    def risImageAcquired(self, ris, imageData):
        if self.showRisFrames:
            self.showImage(imageData)

    def setGtpEnabled(self, gtpEnabled, update=True):
        '''Enable or disable gamma transformation.  If update is true, the widget will be refreshed immediately.'''
        self.panelProg.gtpEnabled = gtpEnabled
        if update:
            self.update()

    def setGtpMinimum(self, gtpMinimum, update=True):
        '''Set gamma transformation minimum intensity parameter.  If update is true, the widget will be refreshed immediately.'''
        self.context().makeCurrent()
        GLS.glProgramUniform1f(self.panelProg.prog, self.panelProg.gtpMinLoc, gtpMinimum)
        if update:
            self.update()

    def setGtpMaximum(self, gtpMaximum, update=True):
        '''Set gamma transformation minimum intensity parameter.  If update is true, the widget will be refreshed immediately.'''
        self.context().makeCurrent()
        GLS.glProgramUniform1f(self.panelProg.prog, self.panelProg.gtpMaxLoc, gtpMaximum)
        if update:
            self.update()

    def setGtpGamma(self, gtpGamma, update=True):
        '''Set gamma transformation gamma parameter.  If update is true, the widget will be refreshed immediately.'''
        self.context().makeCurrent()
        GLS.glProgramUniform1f(self.panelProg.prog, self.panelProg.gtpGammaLoc, gtpGamma)
        if update:
            self.update()

    def setGtpParams(self, gtpEnabled, gtpMinimum, gtpMaximum, gtpGamma, update=True):
        '''Set all gamma transformation parameters.  If update is true, the widget will be refreshed immediately.'''
        self.context().makeCurrent()
        self.panelProg.gtpEnabled = gtpEnabled
        with self.panelProg:
            GLS.glUniform1f(self.panelProg.gtpMinLoc, gtpMinimum)
            GLS.glUniform1f(self.panelProg.gtpMaxLoc, gtpMaximum)
            GLS.glUniform1f(self.panelProg.gtpGammaLoc, gtpGamma)
        if update:
            self.update()

    def getGtpParams(self):
        '''Returns a dict containing all gamma transformation parameter values.'''
        self.context().makeCurrent()
        ret = {}

        ret['gtpEnabled'] = self.panelProg.gtpEnabled

        v = numpy.array([None], numpy.float32)
        GLS.glGetUniformfv(self.panelProg.prog, self.panelProg.gtpMaxLoc, v)
        ret['gtpMaximum'] = v[0]

        v[0] = numpy.nan
        GLS.glGetUniformfv(self.panelProg.prog, self.panelProg.gtpMinLoc, v)
        ret['gtpMinimum'] = v[0]

        v[0] = numpy.nan
        GLS.glGetUniformfv(self.panelProg.prog, self.panelProg.gtpGammaLoc, v)
        ret['gtpGamma'] = v[0]

        return ret
