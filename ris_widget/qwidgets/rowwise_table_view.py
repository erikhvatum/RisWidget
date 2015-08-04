# The MIT License (MIT)
#
# Copyright (c) 2015 WUSTL ZPLAB
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

from PyQt5 import Qt

class RowwiseTableViewMixin:
    def dragEnterEvent(self, event):
        if self.dragDropMode() == Qt.QAbstractItemView.InternalMove and \
           (event.source() is not self or not (event.possibleActions() & Qt.Qt.MoveAction)):
            return
        model = self.model()
        mime = event.mimeData()
        if event.dropAction() & model.supportedDropActions():
            for mime_type_name in model.mimeTypes():
                if mime.hasFormat(mime_type_name):
                    event.accept()
                    self.setState(Qt.QAbstractItemView.DraggingState)
                    return
        event.ignore()

    def dragMoveEvent(self, event):
        if self.dragDropMode() == Qt.QAbstractItemView.InternalMove and \
           (event.source() is not self or not (event.possibleActions() & Qt.Qt.MoveAction)):
            return
        event.ignore()

    def _should_auto_scroll(self, pos):
        if not self.hasAutoScroll():
            return False
        area = self.viewport().rect()
        asm = self.autoScrollMargin()
        return \
            pos.y() - area.top() < asm or \
            area.bottom() - pos.y() < asm or \
            pos.x() - area.left() < asm or \
            area.right() - pos.x() < asm

class RowwiseTableView(RowwiseTableViewMixin, Qt.QTableView):
    def __init__(self, parent=None):
        Qt.QTableView.__init__(self, parent)
        RowwiseTableViewMixin.__init__(self)
