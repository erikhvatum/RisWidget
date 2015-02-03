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

from .canvas_widget import CanvasWidget
import numpy
from PyQt5 import Qt

class ImageWidgetScroller(Qt.QAbstractScrollArea):
    def __init__(self, parent, qsurface_format):
        super().__init__(parent)
        self.setFrameShape(Qt.QFrame.StyledPanel)
        self.setFrameShadow(Qt.QFrame.Raised)
        self.image_widget = ImageWidget(self, qsurface_format)
        self.setViewport(self.image_widget)

    def scrollContentsBy(dx, dy):
        self.image_widget.scroll_contents_by(dx, dy)

class ImageWidget(CanvasWidget):
    _NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE = {
        numpy.uint8  : Qt.QOpenGLTexture.UInt8,
        numpy.uint16 : Qt.QOpenGLTexture.UInt16,
        numpy.float32: Qt.QOpenGLTexture.Float32}
    _IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT = {
        'g'   : Qt.QOpenGLTexture.R32F,
        'ga'  : Qt.QOpenGLTexture.RG32F,
        'rgb' : Qt.QOpenGLTexture.RGB32F,
        'rgba': Qt.QOpenGLTexture.RGBA32F}
    _IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT = {
        'g'   : Qt.QOpenGLTexture.Red,
        'ga'  : Qt.QOpenGLTexture.RG,
        'rgb' : Qt.QOpenGLTexture.RGB,
        'rgba': Qt.QOpenGLTexture.RGBA}

    def __init__(self, parent, qsurface_format):
        super().__init__(parent, qsurface_format)
        self.setFormat(qsurface_format)
        self._image = None
        self._aspect_ratio = None
        self._glsl_prog_g = None
        self._glsl_prog_ga = None
        self._glsl_prog_rgb = None
        self._glsl_prog_rgba = None
        self._image_type_to_glsl_prog = None
        self._tex = None
        self._frag_to_tex = Qt.QMatrix3x3()
        self.setMinimumSize(Qt.QSize(100,100))

    def initializeGL(self):
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
        self._make_quad_buffer()

    def paintGL(self):
        self._glfs.glClear(self._glfs.GL_COLOR_BUFFER_BIT | self._glfs.GL_DEPTH_BUFFER_BIT)
        if self._image is not None:
            prog = self._image_type_to_glsl_prog[self._image.type]
            prog.bind()
            self._quad_buffer.bind()
            self._tex.bind()
            vert_coord_loc = prog.attributeLocation('vert_coord')
            quad_vao_binder = Qt.QOpenGLVertexArrayObject.Binder(self._quad_vao)
            prog.enableAttributeArray(vert_coord_loc)
            prog.setAttributeBuffer(vert_coord_loc, self._glfs.GL_FLOAT, 0, 2, 0)
            prog.setUniformValue('tex', 0)
            self._frag_to_tex.setToIdentity()
            self._frag_to_tex[0,0] = 1/self._image.size.width()
            self._frag_to_tex[1,1] = 1/self._image.size.height()
            prog.setUniformValue('frag_to_tex', self._frag_to_tex)
            prog.setUniformValue('mvp', self._mvp)
            self._glfs.glEnableClientState(self._glfs.GL_VERTEX_ARRAY)
            self._glfs.glDrawArrays(self._glfs.GL_TRIANGLE_FAN, 0, 4)
            self._tex.release()
            self._quad_buffer.release()
            prog.release()

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
                                  image.data.ctypes.data,
                                  pixel_transfer_opts)
                self._tex.release()
                self._image = image
                self._aspect_ratio = image.size.width() / image.size.height()
                self.update()
        finally:
            self.doneCurrent()
