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
from .shared_resources import GL, UNIQUE_QGRAPHICSITEM_TYPE
import math
import numpy
from PyQt5 import Qt
from .shader_scene import ShaderScene, ShaderItem, ShaderQOpenGLTexture
from .shader_view import ShaderView
import sys

class ImageScene(ShaderScene):
    def __init__(self, parent):
        super().__init__(parent)
        self.image_item = ImageItem()
        self.addItem(self.image_item)
        self._histogram_scene = None

    def on_image_changing(self, image):
        self.image_item.on_image_changing(image)
        self.setSceneRect(self.image_item.boundingRect())
        for view in self.views():
            view.on_image_changing(image)

    def on_histogram_gamma_or_min_max_changed(self):
        self.image_item.update()

    @property
    def histogram_scene(self):
        return self._histogram_scene

    @histogram_scene.setter
    def histogram_scene(self, histogram_scene):
        if self._histogram_scene is not None:
            self._histogram_scene.gamma_or_min_max_changed.disconnect(self.on_histogram_gamma_or_min_max_changed)
        histogram_scene.gamma_or_min_max_changed.connect(self.on_histogram_gamma_or_min_max_changed)
        self._histogram_scene = histogram_scene
        self.on_histogram_gamma_or_min_max_changed()

class ImageItem(ShaderItem):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()
    NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE = {
        numpy.uint8  : Qt.QOpenGLTexture.UInt8,
        numpy.uint16 : Qt.QOpenGLTexture.UInt16,
        numpy.float32: Qt.QOpenGLTexture.Float32}
    IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT = {
        'g'   : Qt.QOpenGLTexture.R32F,
        'ga'  : Qt.QOpenGLTexture.RG32F,
        'rgb' : Qt.QOpenGLTexture.RGB32F,
        'rgba': Qt.QOpenGLTexture.RGBA32F}
    IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT = {
        'g'   : Qt.QOpenGLTexture.Red,
        'ga'  : Qt.QOpenGLTexture.RG,
        'rgb' : Qt.QOpenGLTexture.RGB,
        'rgba': Qt.QOpenGLTexture.RGBA}
    IMAGE_TYPE_TO_COMBINE_VT_COMPONENTS = {
        'g'   : 'vec4(vcomponents, vcomponents, vcomponents, 1.0f)',
        'ga'  : 'vec4(vcomponents, vcomponents, vcomponents, tcomponents.a)',
        'rgb' : 'vec4(vcomponents.rgb, 1.0f)',
        'rgba': 'vec4(vcomponents.rgb, tcomponents.a)'}

    def __init__(self, parent_item=None):
        super().__init__(parent_item)
        self.tex = None
        self._show_frame = False

    def type(self):
        return GammaItem.QGRAPHICSITEM_TYPE

    def boundingRect(self):
        return Qt.QRectF(0,0,1,1) if self.image is None else Qt.QRectF(Qt.QPointF(), Qt.QSizeF(self.image.size))

    def paint(self, qpainter, option, widget):
        if widget is None:
            print('WARNING: image_view.ImageItem.paint called with widget=None.  Ensure that view caching is disabled.')
        elif self.image is None:
            if self.tex is not None:
                self.tex.destroy()
                self.tex = None
        else:
            with ExitStack() as stack:
                qpainter.beginNativePainting()
                stack.callback(qpainter.endNativePainting)
                gl = GL()
                image = self.image
                if image.type in self.progs:
                    prog = self.progs[image.type]
                else:
                    if image.is_grayscale:
                        vcomponents_t = 'float'
                        extract_vcomponents = 'tcomponents.r'
                        vcomponents_ones_vector = '1.0f'
                    else:
                        vcomponents_t = 'vec3'
                        extract_vcomponents = 'tcomponents.rgb'
                        vcomponents_ones_vector = 'vec3(1.0f, 1.0f, 1.0f)'
                    prog = self.build_shader_prog(image.type,
                                                  'image_widget_vertex_shader.glsl',
                                                  'image_widget_fragment_shader_template.glsl',
                                                  vcomponents_t=vcomponents_t,
                                                  extract_vcomponents=extract_vcomponents,
                                                  vcomponents_ones_vector=vcomponents_ones_vector,
                                                  combine_vt_components=ImageItem.IMAGE_TYPE_TO_COMBINE_VT_COMPONENTS[image.type])
                prog.bind()
                stack.callback(prog.release)
                desired_texture_format = ImageItem.IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT[image.type]
                tex = self.tex
                if tex is not None:
                    if image.size != Qt.QSize(tex.width(), tex.height()) or tex.format() != desired_texture_format:
                        tex.destroy()
                        tex = self.tex = None
                if tex is None:
                    tex = ShaderQOpenGLTexture(Qt.QOpenGLTexture.Target2D)
                    tex.setFormat(desired_texture_format)
                    tex.setWrapMode(Qt.QOpenGLTexture.ClampToEdge)
                    tex.setMipLevels(1)
                    tex.setAutoMipMapGenerationEnabled(False)
                    tex.setSize(image.size.width(), image.size.height(), 1)
                    tex.allocateStorage()
                    tex.setMinMagFilters(Qt.QOpenGLTexture.Linear, Qt.QOpenGLTexture.Nearest)
                    tex.image_id = -1
                tex.bind()
                stack.callback(tex.release)
                if tex.image_id != self._image_id:
#                   import time
#                   t0=time.time()
                    pixel_transfer_opts = Qt.QOpenGLPixelTransferOptions()
                    pixel_transfer_opts.setAlignment(1)
                    tex.setData(ImageItem.IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT[image.type],
                                ImageItem.NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE[image.dtype],
                                image.data.ctypes.data,
                                pixel_transfer_opts)
#                   t1=time.time()
#                   print('tex.setData {}ms / {}fps'.format(1000*(t1-t0), 1/(t1-t0)))
                    tex.image_id = self._image_id
                    # self.tex is updated here and not before so that any failure preparing tex results in a retry the next time self.tex
                    # is needed
                    self.tex = tex
                view = widget.view
                view.quad_buffer.bind()
                stack.callback(view.quad_buffer.release)
                view.quad_vao.bind()
                stack.callback(view.quad_vao.release)
                vert_coord_loc = prog.attributeLocation('vert_coord')
                prog.enableAttributeArray(vert_coord_loc)
                prog.setAttributeBuffer(vert_coord_loc, gl.GL_FLOAT, 0, 2, 0)
                prog.setUniformValue('tex', 0)
                frag_to_tex = Qt.QTransform()
                frame = Qt.QPolygonF(view.mapFromScene(self.boundingRect()))
                if not qpainter.transform().quadToSquare(frame, frag_to_tex):
                    raise RuntimeError('Failed to compute gl_FragCoord to texture coordinate transformation matrix.')
                prog.setUniformValue('frag_to_tex', frag_to_tex)
                prog.setUniformValue('viewport_height', float(widget.size().height()))
                histogram_scene = view.scene().histogram_scene
                if image.is_grayscale:
                    gamma = histogram_scene.gamma
                    min_max = numpy.array((histogram_scene.min, histogram_scene.max), dtype=float)
                    if image.dtype != numpy.float32:
                        self._normalize_min_max(min_max)
                    prog.setUniformValue('gammas', gamma)
                    prog.setUniformValue('vcomponent_rescale_mins', min_max[0])
                    prog.setUniformValue('vcomponent_rescale_ranges', min_max[1] - min_max[0])
                else:
                    gammas = (histogram_scene.gamma_red, histogram_scene.gamma_green, histogram_scene.gamma_blue)
                    min_maxs = numpy.array(((histogram_scene.min_red, histogram_scene.min_green, histogram_scene.min_blue),
                                            (histogram_scene.max_red, histogram_scene.max_green, histogram_scene.max_blue)), dtype=float)
                    prog.setUniformValue('gammas', *gammas)
                    if image.dtype != numpy.float32:
                        self._normalize_min_max(min_maxs)
                    prog.setUniformValue('vcomponent_rescale_mins', *min_maxs[0])
                    prog.setUniformValue('vcomponent_rescale_ranges', *(min_maxs[1]-min_maxs[0]))
                gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
                gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)
            if self._show_frame:
                qpainter.setBrush(Qt.QBrush(Qt.Qt.transparent))
                color = Qt.QColor(Qt.Qt.red)
                color.setAlphaF(0.5)
                pen = Qt.QPen(color)
                pen.setWidth(2)
                pen.setCosmetic(True)
                pen.setStyle(Qt.Qt.DotLine)
                qpainter.setPen(pen)
                qpainter.drawRect(0, 0, image.size.width(), image.size.height())

    def hoverMoveEvent(self, event):
        if self.image is not None:
            # NB: event.pos() is a QPointF, and one may call QPointF.toPoint(), as in the following line,
            # to get a QPoint from it.  However, toPoint() rounds x and y coordinates to the nearest int,
            # which would cause us to erroneously report mouse position as being over the pixel to the
            # right and/or below if the view with the mouse cursor is zoomed in such that an image pixel
            # occupies more than one screen pixel and the cursor is over the right and/or bottom half
            # of a pixel.
#           pos = event.pos().toPoint()
            pos = Qt.QPoint(event.pos().x(), event.pos().y())
            if Qt.QRect(Qt.QPoint(), self.image.size).contains(pos):
                mst = 'x:{} y:{} '.format(pos.x(), pos.y())
                image_type = self.image.type
                vt = '(' + ' '.join((c + ':{}' for c in image_type)) + ')'
                if len(image_type) == 1:
                    vt = vt.format(self.image.data[pos.x(), pos.y()])
                else:
                    vt = vt.format(*self.image.data[pos.x(), pos.y()])
                self.scene().update_contextual_info(mst + vt, False, self)

    def hoverLeaveEvent(self, event):
        self.scene().clear_contextual_info(self)

    def on_image_changing(self, image):
        if self.image is None and image is not None or \
           self.image is not None and (image is None or self.image.size != image.size):
            self.prepareGeometryChange()
        super().on_image_changing(image)

    @property
    def show_frame(self):
        return self._show_frame

    @show_frame.setter
    def show_frame(self, show_frame):
        if show_frame != self.show_frame:
            self._show_frame = show_frame
            self.update()

#class ImageOverlayItem(Qt.QGraphicsObject):

