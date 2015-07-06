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
from ..qdelegates.property_checkbox_delegate import PropertyCheckboxDelegate
from ..signaling_list.signaling_list import SignalingList
from ..signaling_list.signaling_list_property_table_model import SignalingListPropertyTableModel

#TODO: make list items drop targets so that layer contents can be replaced by dropping file on associated item
class ImageStackTableView(Qt.QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.horizontalHeader().setSectionResizeMode(Qt.QHeaderView.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)
        self.property_checkbox_delegate = PropertyCheckboxDelegate(self)
        self.setItemDelegateForColumn(0, self.property_checkbox_delegate)
        self.setSelectionBehavior(Qt.QAbstractItemView.SelectRows)
        self.setSelectionMode(Qt.QAbstractItemView.SingleSelection)

class ImageStackTableModel(SignalingListPropertyTableModel):
    def __init__(self, signaling_list, parent):
        super().__init__(('visible', 'name', 'size', 'type', 'dtype'), signaling_list, parent)

    def flags(self, midx):
        flags = Qt.Qt.ItemIsEnabled | Qt.Qt.ItemIsSelectable | Qt.Qt.ItemNeverHasChildren
        column = midx.column()
        if column == 0:
            flags |= Qt.Qt.ItemIsUserCheckable
        elif column == 1:
            flags |= Qt.Qt.ItemIsEditable
        return flags

    def data(self, midx, role=Qt.Qt.DisplayRole):
        if midx.column() == 0:
            if role == Qt.Qt.CheckStateRole and midx.isValid():
                return Qt.QVariant(Qt.Qt.Checked if self.signaling_list[midx.row()].visible else Qt.Qt.Unchecked)
            return Qt.QVariant()
        return super().data(midx, role)

    def setData(self, midx, value, role=Qt.Qt.EditRole):
        if midx.isValid():
            column = midx.column()
            if column == 0:
                if role == Qt.Qt.CheckStateRole:
                    setattr(self.signaling_list[midx.row()], self._property_names[midx.column()], value.value())
                    return True
            elif column == 1:
                super().setData(midx, value, role)
        return False
