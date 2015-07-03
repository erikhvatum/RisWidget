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
from pathlib import Path
import OpenGL
import OpenGL.GL.NV.path_rendering as PR
from PyQt5 import Qt
from ..shared_resources import UNIQUE_QGRAPHICSITEM_TYPE

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
            self._update_paths()
            estack.callback(qpainter.endNativePainting)

    def return_to_fixed_position(self, view):
        """Maintain position self.FIXED_POSITION_IN_VIEW relative to view's top left corner."""
        topleft = self.FIXED_POSITION_IN_VIEW
        if view.mapFromScene(self.pos()) != topleft:
            self.setPos(view.mapToScene(topleft))

    def _update_paths(self):
        if self._path_serial != self._text_serial:
            if self._path_serial is not None:
                PR.deletePathsNV(self._path, len(self.text))

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, v):
        if self._text != v:
            if v:
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
