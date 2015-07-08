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
from ..image.image import Image
from ..qdelegates.dropdown_list_delegate import DropdownListDelegate
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
        self.blend_function_delegate = DropdownListDelegate(Image.BLEND_FUNCTIONS, self)
        self.setItemDelegateForColumn(2, self.blend_function_delegate)
        self.setSelectionBehavior(Qt.QAbstractItemView.SelectRows)
        self.setSelectionMode(Qt.QAbstractItemView.SingleSelection)
#       self.setEditTriggers(Qt.QAbstractItemView.EditKeyPressed | Qt.QAbstractItemView.SelectedClicked)

class ImageStackTableModel(SignalingListPropertyTableModel):
    def __init__(self, signaling_list, parent):
        super().__init__(('visible', 'name', 'blend_function', 'size', 'type', 'dtype'), signaling_list, parent)
        ngs = {
            'visible' : self.__getd_visible,
            'size' : self.__getd_size,
            'dtype' : self.__getd_dtype}
        self.__property_data_getters = {self.property_columns[n] : g for n, g in ngs.items()}

    def flags(self, midx):
        flags = Qt.Qt.ItemIsEnabled | Qt.Qt.ItemIsSelectable | Qt.Qt.ItemNeverHasChildren
        column = midx.column()
        if column == 0:
            flags |= Qt.Qt.ItemIsUserCheckable
        elif column in (1, 2):
            flags |= Qt.Qt.ItemIsEditable
        return flags

    def data(self, midx, role=Qt.Qt.DisplayRole):
        if midx.isValid():
            d = self.__property_data_getters.get(midx.column(), super().data)(midx, role)
            if isinstance(d, Qt.QVariant):
                return d
        return Qt.QVariant()

    def __getd_visible(self, midx, role):
        if role == Qt.Qt.CheckStateRole:
            return Qt.QVariant(Qt.Qt.Checked if self.signaling_list[midx.row()].visible else Qt.Qt.Unchecked)

    def __getd_size(self, midx, role):
        if role == Qt.Qt.DisplayRole:
            qsize = self.signaling_list[midx.row()].size
            return Qt.QVariant('{}x{}'.format(qsize.width(), qsize.height()))

    def __getd_dtype(self, midx, role):
        if role == Qt.Qt.DisplayRole:
            return Qt.QVariant(str(self.signaling_list[midx.row()].data.dtype))

    def setData(self, midx, value, role=Qt.Qt.EditRole):
        if midx.isValid():
            column = midx.column()
            if column == 0:
                if role == Qt.Qt.CheckStateRole:
                    setattr(self.signaling_list[midx.row()], self.property_names[midx.column()], value.value())
                    return True
            elif column in (1, 2):
                return super().setData(midx, value, role)
        return False
