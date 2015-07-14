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
import ctypes
import numpy
from pathlib import Path
import OpenGL
import OpenGL.GL as PyGL
import OpenGL.GL.NV.path_rendering as PR
from PyQt5 import Qt
from ..shared_resources import UNIQUE_QGRAPHICSITEM_TYPE

c_float32_p = ctypes.POINTER(ctypes.c_float)
c_uint8_p = ctypes.POINTER(ctypes.c_uint8)

class ContextualInfoItemNV(Qt.QGraphicsObject):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()
    FIXED_POSITION_IN_VIEW = Qt.QPoint(10, 5)

    text_changed = Qt.pyqtSignal()

    def __init__(self, parent_item=None):
        super().__init__(parent_item)
        self.setFlag(Qt.QGraphicsItem.ItemIgnoresTransformations)
#       self._font = Qt.QFont('Courier', pointSize=16, weight=Qt.QFont.Bold)
#       self._font.setKerning(False)
#       self._font.setStyleHint(Qt.QFont.Monospace, Qt.QFont.OpenGLCompatible | Qt.QFont.PreferQuality)
#       self._pen = Qt.QPen(Qt.QColor(Qt.Qt.black))
#       self._pen.setWidth(2)
#       self._pen.setCosmetic(True)
#       self._brush = Qt.QBrush(Qt.QColor(45,255,70,255))
        self._text = None
        self._text_serial = 0
        self._glyph_base = None
        self._path = None
        self._path_serial = None
#       self._text_flags = Qt.Qt.AlignLeft | Qt.Qt.AlignTop | Qt.Qt.AlignAbsolute
        self._bounding_rect = None
        # Necessary to prevent context information from disappearing when mouse pointer passes over
        # context info text
        self.setAcceptHoverEvents(False)
        self.setAcceptedMouseButtons(Qt.Qt.NoButton)
        # Info text generally should appear over anything else rather than z-fighting
        self.setZValue(10)
        self.hide()

    def __del__(self):
        scene = self.scene()
        if scene is None:
            return
        views = scene.views()
        if not views:
            return
        view = views[0]
        gl_widget = view.gl_widget
        context = gl_widget.context()
        if not context:
            return
        gl_widget.makeCurrent()
        try:
            if self._glyph_base is not None:
                PR.glDeletePathsNV(self._glyph_base, 256)
        finally:
            gl_widget.doneCurrent()

    def type(self):
        return ContextualInfoItem.QGRAPHICSITEM_TYPE

    def boundingRect(self):
        if 0:#self._text:
            self._update_picture()
            return self._bounding_rect
        else:
            # TODO: compute bounding rect in _update_paths()
            return Qt.QRectF(0,0,1,1)

    def paint(self, qpainter, option, widget):
        with ExitStack() as estack:
            qpainter.beginNativePainting()
            estack.callback(qpainter.endNativePainting)
            self._update_paths()
            PyGL.glClearStencil(0)
            PyGL.glClearColor(0,0,0,0)
            PyGL.glStencilMask(~0)
            PyGL.glEnable(PyGL.GL_STENCIL_TEST)
            PyGL.glStencilFunc(PyGL.GL_NOTEQUAL, 0, 0x1)
            PyGL.glStencilOp(PyGL.GL_KEEP, PyGL.GL_KEEP, PyGL.GL_ZERO)
            PyGL.glPushMatrix()
            estack.callback(PyGL.glPopMatrix)
            PyGL.glScale(1,-1,1)
            PyGL.glTranslate(0,-30,0)

            # Draw text outline
            PR.glStencilStrokePathInstancedNV(
                self._text_encoded.shape[0], PyGL.GL_UNSIGNED_BYTE, self._text_encoded.ctypes.data_as(c_uint8_p),
                self._glyph_base,
                1, ~0,
                PR.GL_TRANSLATE_X_NV, self._text_kerning.ctypes.data_as(c_float32_p))
            PyGL.glColor3f(0,0,0)
            PR.glCoverStrokePathInstancedNV(
                self._text_encoded.shape[0], PyGL.GL_UNSIGNED_BYTE, self._text_encoded.ctypes.data_as(c_uint8_p),
                self._glyph_base,
                PR.GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV,
                PR.GL_TRANSLATE_X_NV, self._text_kerning.ctypes.data_as(c_float32_p))

            # Draw filled text interiors
            PR.glStencilFillPathInstancedNV(
                self._text_encoded.shape[0], PyGL.GL_UNSIGNED_BYTE, self._text_encoded.ctypes.data_as(c_uint8_p),
                self._glyph_base,
                PR.GL_PATH_FILL_MODE_NV, ~0,
                PR.GL_TRANSLATE_X_NV, self._text_kerning.ctypes.data_as(c_float32_p))
            PyGL.glColor3f(0,1,0)
            PR.glCoverFillPathInstancedNV(
                self._text_encoded.shape[0], PyGL.GL_UNSIGNED_BYTE, self._text_encoded.ctypes.data_as(c_uint8_p),
                self._glyph_base,
                PR.GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV,
                PR.GL_TRANSLATE_X_NV, self._text_kerning.ctypes.data_as(c_float32_p))

    def return_to_fixed_position(self, view):
        """Maintain position self.FIXED_POSITION_IN_VIEW relative to view's top left corner."""
        topleft = self.FIXED_POSITION_IN_VIEW
        if view.mapFromScene(self.pos()) != topleft:
            self.setPos(view.mapToScene(topleft))

    def _update_paths(self):
        if self._path is None:
            self._path = PR.glGenPathsNV(1)
            junk = numpy.zeros((10,), dtype=numpy.uint8)
            PR.glPathCommandsNV(self._path, 0, junk.ctypes.data_as(c_uint8_p), 0, PyGL.GL_FLOAT, junk.ctypes.data_as(c_float32_p))
            PR.glPathParameterfNV(self._path, PR.GL_PATH_STROKE_WIDTH_NV, 3.2)
            PR.glPathParameteriNV(self._path, PR.GL_PATH_JOIN_STYLE_NV, PR.GL_ROUND_NV)
        if self._glyph_base is None:
            glyph_base = PR.glGenPathsNV(256)
            for fontname in ("Liberation Mono", "Courier", "Terminal"):
                PR.glPathGlyphRangeNV(
                    glyph_base,
                    PR.GL_SYSTEM_FONT_NAME_NV, fontname, PR.GL_BOLD_BIT_NV,
                    0, 256,
                    PR.GL_SKIP_MISSING_GLYPH_NV, self._path, 32)
            PR.glPathGlyphRangeNV(
                    glyph_base,
                    PR.GL_SYSTEM_FONT_NAME_NV, fontname, PR.GL_BOLD_BIT_NV,
                    0, 256,
                    PR.GL_USE_MISSING_GLYPH_NV, self._path, 32)
            self._glyph_base = glyph_base
        if self._path_serial != self._text_serial:
            encoded_kern_text = numpy.array(list((self._text + '&').encode('ISO-8859-1')), dtype=numpy.uint8)
            self._text_kerning = numpy.zeros((len(self._text_encoded) + 1,), dtype=numpy.float32)
            self._text_kerning[0] = 0
            tko = self._text_kerning[1:]
            PR.glGetPathSpacingNV(
                PR.GL_ACCUM_ADJACENT_PAIRS_NV,
                encoded_kern_text.shape[0], PyGL.GL_UNSIGNED_BYTE, encoded_kern_text,#encoded_kern_text.ctypes.data_as(c_uint8_p),
                self._glyph_base,
                1.0, 1.0, PR.GL_TRANSLATE_X_NV,
                tko.ctypes.data_as(c_float32_p))
            self._path_serial = self._text_serial

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, v):
        #TODO: figure out how to deal with \n
        if v is not None:
            v = v.replace('\n', '\\n')
        if self._text != v:
            if v:
                encoded = v.encode('ISO-8859-1') # Only ISO-8859-1 is supported in this proof-of-concept implementation
                self._text_encoded = numpy.array(list(encoded), dtype=numpy.uint8)
                self.prepareGeometryChange()
            self._text = v
            self._text_serial += 1
            if self._text:
                self.show()
                self.update()
            else:
                self.hide()
            self.text_changed.emit()

#   @property
#   def font(self):
#       return self._font
#
#   @font.setter
#   def font(self, v):
#       assert isinstance(v, Qt.QFont)
#       self._font = v
#       self._picture = None
#       self.update()
#
#   @property
#   def pen(self):
#       """The pen used to draw text outline to provide contrast against any background.  If None,
#       outline is not drawn."""
#       return self._pen
#
#   @pen.setter
#   def pen(self, v):
#       assert isinstance(v, Qt.QPen) or v is None
#       self._pen = v
#       self._picture = None
#       self.update()
#
#   @property
#   def brush(self):
#       """The brush used to fill text.  If None, text is not filled."""
#       return self._brush
#
#   @brush.setter
#   def brush(self, v):
#       assert isinstance(v, Qt.QBrush) or v is None
#       self._brush = v
#       self._picture = None
#       self.update()
