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

from PyQt5 import Qt

class PropertyCheckboxDelegate(Qt.QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def paint(self, painter, option, midx):
        data = midx.model().data(midx, Qt.Qt.CheckStateRole).value()
        print('paint', data)

        s = Qt.QApplication.style()
        cbs = Qt.QStyleOptionButton()
        cbr = Qt.QRect(s.subElementRect(Qt.QStyle.SE_CheckBoxIndicator, cbs))

        cbs.rect = Qt.QRect(option.rect)
        cbs.rect.setLeft(option.rect.x() + (option.rect.width() - cbr.width()) / 2)

        if data:
            cbs.state = Qt.QStyle.State_On | Qt.QStyle.State_Enabled
        else:
            cbs.state = Qt.QStyle.State_Off | Qt.QStyle.State_Enabled

        s.drawControl(Qt.QStyle.CE_CheckBox, cbs, painter)

    def sizeHint(self, option, midx):
        return Qt.QApplication.style().subElementRect(Qt.QStyle.SE_CheckBoxIndicator, Qt.QStyleOptionButton()).size()

    def createEditor(self, parent, option, midx):
        return Qt.QCheckBox(parent)

    def setEditorData(self, editor, midx):
        # Load data from model into editor widget
        print(type(midx.data(Qt.Qt.CheckStateRole)))
#       editor.setChecked(midx.data(Qt.Qt.CheckStateRole))
        editor.setChecked(False)

    def setModelData(self, editor, model, midx):
        # Store data from editor widget in model
        print('setModelData', editor.isChecked(), model)
        model.setData(midx, editor.isChecked(), Qt.Qt.EditRole)

    def updateEditorGeometry(self, editor, option, midx):
        cbs = Qt.QStyleOptionButton()
        cbr = Qt.QRect(Qt.QApplication.style().subElementRect(Qt.QStyle.SE_CheckBoxIndicator, cbs))
        w, h = cbr.width(), cbr.height()
        l = option.rect.x() + (option.rect.width() - w) / 2
        t = option.rect.y() + (option.rect.height() - h) / 2
        editor.setGeometry(l, t, w, h)
