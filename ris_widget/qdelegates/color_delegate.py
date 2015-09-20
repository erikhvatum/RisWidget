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

class ColorDelegate(Qt.QStyledItemDelegate):
    def createEditor(self, parent, option, midx):
        if midx.isValid():
            e = Qt.QColorDialog(parent)
            e.setOption(Qt.QColorDialog.ShowAlphaChannel)
            e.setOption(Qt.QColorDialog.DontUseNativeDialog)
            # Due to a modal event loop peculiarity, the .result() value for a modal dialog created by a delegate is not updated upon
            # dialog dismissal as it should be.  The following installs attempts to install workaround for this oddity, and if it
            # can not, OK/Cancel buttons are omitted from the color picker dialog.  In this case, the color picker dialog may be dismissed
            # by clicking anywhere in the application outside the dialog, and the selected color is always used.
            bb = e.findChild(Qt.QDialogButtonBox)
            if bb is None:
                e.setOption(Qt.QColorDialog.NoButtons)
            else:
                bb.accepted.connect(lambda: e.setResult(1))
            return e

    def setEditorData(self, e, midx):
        d = midx.data(Qt.Qt.DecorationRole)
        if isinstance(d, Qt.QVariant):
            d = d.value()
        e.setCurrentColor(d)

    def setModelData(self, e, model, midx):
        has_bb = not e.testOption(Qt.QColorDialog.NoButtons)
        if has_bb and e.result() or not has_bb:
            color = e.currentColor()
            model.setData(midx, (color.redF(), color.greenF(), color.blueF(), color.alphaF()))

    def paint(self, painter, option, midx):
        style = None
        if option.widget is not None:
            style = option.widget.style()
        if style is None:
            style = Qt.QApplication.style()
        # Fill cell background in the *exact same manner* as the default delegate.  This is the simplest way to get the correct
        # cell background in all circumstances, including while dragging a row.
        style.drawPrimitive(Qt.QStyle.PE_PanelItemViewItem, option, painter, option.widget)
        d = midx.data(Qt.Qt.DecorationRole)
        if isinstance(d, Qt.QVariant):
            d = d.value()
        swatch_rect = Qt.QStyle.alignedRect(
            option.direction,
            Qt.Qt.AlignCenter,
            option.decorationSize,
            option.rect)
        painter.save()
        try:
            painter.setPen(Qt.QPen(option.palette.color(Qt.QPalette.Normal, Qt.QPalette.Mid)))
            painter.setBrush(Qt.QBrush(d))
            painter.drawRect(swatch_rect)
        finally:
            painter.restore()
