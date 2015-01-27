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

import ctypes
import numpy
from pathlib import Path
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
    _NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE = {
        numpy.uint8  : Qt.QOpenGLTexture.UInt8,
        numpy.uint16 : Qt.QOpenGLTexture.UInt16,
        numpy.float32: Qt.QOpenGLTexture.Float32}
    _IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT = {
        'g'   : Qt.QOpenGLTexture.R32F,
        'ga'  : Qt.QOpenGLTexture.R32F,
        'rgb' : Qt.QOpenGLTexture.RGB32F,
        'rgba': Qt.QOpenGLTexture.RGB32F}
    _IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT = {
        'g'   : Qt.QOpenGLTexture.Red,
        'ga'  : Qt.QOpenGLTexture.RG,
        'rgb' : Qt.QOpenGLTexture.RGB,
        'rgba': Qt.QOpenGLTexture.RGBA}

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
        self._image = None
        self._aspect_ratio = None
        self._glfs = None
        self._glsl_prog_g = None
        self._glsl_prog_rgb = None
        self._tex = None

    def _build_shader_prog(self, prog_n, desc, vert_fn, frag_fn):
        source_dpath = Path(__file__).parent
        setattr(self, prog_n, Qt.QOpenGLShaderProgram(self))
        prog = getattr(self, prog_n)
        if not prog.addShaderFromSourceFile(Qt.QOpenGLShader.Vertex, str(source_dpath / vert_fn)):
            raise RuntimeError('Failed to compile vertex shader "{}" for ImageWidget {} shader program.'.format(vert_fn, desc))
        if not prog.addShaderFromSourceFile(Qt.QOpenGLShader.Fragment, str(source_dpath / frag_fn)):
            raise RuntimeError('Failed to compile fragment shader "{}" for ImageWidget {} shader program.'.format(frag_fn, desc))
        if not prog.link():
            raise RuntimeError('Failed to link ImageWidget {} program.'.format(desc))

    def initializeGL(self):
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
        self._glfs.glClearColor(0,0,0,1)
        self._build_shader_prog('_glsl_prog_g', 'grayscale',
                                  'image_widget_vertex_shader.glsl',
                                  'image_widget_fragment_shader_gray.glsl')
        self._build_shader_prog('_glsl_prog_rgb', 'rgb',
                                  'image_widget_vertex_shader.glsl',
                                  'image_widget_fragment_shader_rgb.glsl')

    def paintGL(self):
        self._glfs.glClear(self._glfs.GL_COLOR_BUFFER_BIT | self._glfs.GL_DEPTH_BUFFER_BIT)

    def resizeGL(self, x, y):
        pass

    def scroll_contents_by(self, dx, dy):
        pass

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        try:
            self.makeCurrent()
            if self._image is not None and (image is None or self._image.size != image.size):
                self._tex = None
                self._image = None
            if image is not None:
                desired_texture_format = ImageWidget._IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT[image.type]
                if self._tex is None or not self._tex.isCreated() or self._tex.format() != desired_texture_format:
                    self._tex = Qt.QOpenGLTexture(Qt.QOpenGLTexture.Target2D)
                    self._tex.setFormat(desired_texture_format)
                    self._tex.setWrapMode(Qt.QOpenGLTexture.ClampToEdge)
                    self._tex.setAutoMipMapGenerationEnabled(True)
                    self._tex.setSize(image.size.width(), image.size.height(), 1)
                    self._tex.setMipLevels(4)
                    self._tex.allocateStorage()
                self._tex.setMinMagFilters(Qt.QOpenGLTexture.LinearMipMapLinear, Qt.QOpenGLTexture.Nearest)
                self._tex.bind()
                pixel_transfer_opts = Qt.QOpenGLPixelTransferOptions()
                pixel_transfer_opts.setAlignment(1)
                self._tex.setData(ImageWidget._IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT[image.type],
                                  ImageWidget._NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE[image.dtype],
                                  ctypes.c_void_p(image.data.ctypes.data),
                                  pixel_transfer_opts)
                self._tex.release()
                self._image = image
                self._aspect_ratio = image.size.width() / image.size.height()
        finally:
            self.doneCurrent()
