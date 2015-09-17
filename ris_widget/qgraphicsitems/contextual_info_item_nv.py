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
        self._text_lines_encoded = None
        self._text_lines_kerning = None
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
        except:
            pass
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
            # TODO: determine why Y axis needs to be inverted for NV_path_drawing
            PyGL.glScale(1,-1,1)
            # TODO: If flipping does turn out to be necessary, translate down by total text height as computed via
            # font metrics from path rendering extension rather than by the fixed number -22
            LINE_OFFSET = -22

            for l, k in zip(self._text_lines_encoded, self._text_lines_kerning):
                PyGL.glTranslate(0,LINE_OFFSET,0)
                l_len = l.shape[0] - 1
                l_p = l.ctypes.data_as(c_uint8_p)
                k_p = k.ctypes.data_as(c_float32_p)

                # Draw text outline
                PR.glStencilStrokePathInstancedNV(
                    l_len, PyGL.GL_UNSIGNED_INT, l_p,
                    self._glyph_base,
                    1, ~0,
                    PR.GL_TRANSLATE_X_NV, k_p)
                PyGL.glColor3f(0,0,0)
                PR.glCoverStrokePathInstancedNV(
                    l_len, PyGL.GL_UNSIGNED_INT, l_p,
                    self._glyph_base,
                    PR.GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV,
                    PR.GL_TRANSLATE_X_NV, k_p)

                # Draw filled text interiors
                PR.glStencilFillPathInstancedNV(
                    l_len, PyGL.GL_UNSIGNED_INT, l_p,
                    self._glyph_base,
                    PR.GL_PATH_FILL_MODE_NV, ~0,
                    PR.GL_TRANSLATE_X_NV, k_p)
                PyGL.glColor3f(0,1,0)
                PR.glCoverFillPathInstancedNV(
                    l_len, PyGL.GL_UNSIGNED_INT, l_p,
                    self._glyph_base,
                    PR.GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV,
                    PR.GL_TRANSLATE_X_NV, k_p)

    def return_to_fixed_position(self, view):
        """Maintain position self.FIXED_POSITION_IN_VIEW relative to view's top left corner."""
        topleft = self.FIXED_POSITION_IN_VIEW
        if view.mapFromScene(self.pos()) != topleft:
            self.setPos(view.mapToScene(topleft))

    def _update_paths(self):
        if self._path is None:
            self._path = PR.glGenPathsNV(1)
            PR.glPathCommandsNV(self._path, 0, c_uint8_p(), 0, PyGL.GL_FLOAT, c_float32_p())
            PR.glPathParameterfNV(self._path, PR.GL_PATH_STROKE_WIDTH_NV, 3.0)
            PR.glPathParameteriNV(self._path, PR.GL_PATH_JOIN_STYLE_NV, PR.GL_ROUND_NV)
        if self._glyph_base is None:
            glyph_base = PR.glGenPathsNV(0x110000)
            for fontname in ("Liberation Mono", "Courier", "Terminal"):
                PR.glPathGlyphRangeNV(
                    glyph_base,
                    PR.GL_SYSTEM_FONT_NAME_NV, fontname, PR.GL_BOLD_BIT_NV,
                    0, 0x110000,
                    PR.GL_SKIP_MISSING_GLYPH_NV, self._path, 20)
            PR.glPathGlyphRangeNV(
                    glyph_base,
                    PR.GL_SYSTEM_FONT_NAME_NV, fontname, PR.GL_BOLD_BIT_NV,
                    0, 0x110000,
                    PR.GL_USE_MISSING_GLYPH_NV, self._path, 20)
            # TODO: determine why a) glGetPathMetricRangeNV gives NaN for offsets when called with
            # PR.GL_FONT_Y_MIN_BOUNDS_BIT_NV | PR.GL_FONT_Y_MAX_BOUNDS_BIT_NV b) whether individually computed
            # offsets vary with current projection and/or model and/or view matrices and if not, why results
            # are so different for histogram and general context info items
#           self._glyph_y_minmax_bounds = numpy.empty((2,), dtype=numpy.float32)
#           PR.glGetPathMetricRangeNV(
#               PR.GL_FONT_Y_MIN_BOUNDS_BIT_NV,
#               self._glyph_base, 1,
#               8,#self._glyph_y_minmax_bounds.nbytes,
#               self._glyph_y_minmax_bounds.ctypes.data_as(c_float32_p))
#           print('min', self._glyph_y_minmax_bounds)
#           PR.glGetPathMetricRangeNV(
#               PR.GL_FONT_Y_MAX_BOUNDS_BIT_NV,
#               self._glyph_base, 1,
#               8,#self._glyph_y_minmax_bounds.nbytes,
#               self._glyph_y_minmax_bounds.ctypes.data_as(c_float32_p))
#           print('max', self._glyph_y_minmax_bounds)
            self._glyph_base = glyph_base
        if self._path_serial != self._text_serial:
            self._text_lines_kerning = []
            for l in self._text_lines_encoded:
                tk = numpy.empty(l.shape, dtype=numpy.float32)
                tk[0] = 0
                tko = tk[1:]
                PR.glGetPathSpacingNV(
                    PR.GL_ACCUM_ADJACENT_PAIRS_NV,
                    l.shape[0], PyGL.GL_UNSIGNED_INT, l.ctypes.data_as(c_uint8_p),
                    self._glyph_base,
                    1.0, 1.0, PR.GL_TRANSLATE_X_NV,
                    tko.ctypes.data_as(c_float32_p))
                self._text_lines_kerning.append(tk)
            self._path_serial = self._text_serial

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, v):
        if self._text != v:
            if v:
                # The '&' is appended owing to a peculiarity of the glGetPathSpacingNV call, which wants a junk character
                # at the end of each line.  The '&' is not actually visible.
                lines_encoded = [numpy.array([ord(c) for c in l + '&'], dtype=numpy.uint32) for l in v.split('\n')]
                self.prepareGeometryChange()
                self._text_lines_encoded = lines_encoded
            else:
                self._text_lines_encoded = None
                self._text_lines_kerning = None
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
