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
from ..qdelegates.dropdown_list_delegate import DropdownListDelegate
from ..qdelegates.tint_delegate import TintDelegate
from ..qdelegates.property_checkbox_delegate import PropertyCheckboxDelegate
from ..signaling_list import SignalingList
from ..signaling_list_property_table_model import SignalingListPropertyTableModel

#TODO: make list items drop targets so that layer contents can be replaced by dropping file on associated item
class LayerStackTableView(Qt.QTableView):
    def __init__(self, layer_stack_table_model, parent=None):
        super().__init__(parent)
        self.horizontalHeader().setSectionResizeMode(Qt.QHeaderView.ResizeToContents)
#       self.horizontalHeader().setStretchLastSection(True)
        self.property_checkbox_delegate = PropertyCheckboxDelegate(self)
        self.setItemDelegateForColumn(layer_stack_table_model.property_columns['visible'], self.property_checkbox_delegate)
        self.setItemDelegateForColumn(layer_stack_table_model.property_columns['auto_min_max_enabled'], self.property_checkbox_delegate)
        self.blend_function_delegate = DropdownListDelegate(lambda midx: self.model().signaling_list[midx.row()].BLEND_FUNCTIONS, self)
        self.setItemDelegateForColumn(layer_stack_table_model.property_columns['blend_function'], self.blend_function_delegate)
        self.tint_delegate = TintDelegate(self)
        self.setItemDelegateForColumn(layer_stack_table_model.property_columns['tint'], self.tint_delegate)
        self.setSelectionBehavior(Qt.QAbstractItemView.SelectRows)
        self.setSelectionMode(Qt.QAbstractItemView.SingleSelection)
        self.setModel(layer_stack_table_model)
#       self.setEditTriggers(Qt.QAbstractItemView.EditKeyPressed | Qt.QAbstractItemView.SelectedClicked)

class LayerStackTableModel(SignalingListPropertyTableModel):
    # ImageStackTableModel accesses PROPERTIES strictly via self.PROPERTIES and never via ImageStackTableModel.PROPERTIES,
    # meaning that subclasses may safely add or remove columns by overridding PROPERTIES.  For example, adding a column for
    # a sublcassed Images having an "image_quality" property:
    #
    # class ImageStackTableModel_ImageQuality(ImageStackTableModel):
    #     PROPERTIES = ImageStackTableModel.PROPERTIES + ('image_quality',)
    #
    # And that's it, provided image_quality is always a plain string and should not be editable.  Making it editable
    # would require adding an entry to self._special_flag_getters.  Alternative .flags may be overridden to activate the
    # Qt.Qt.ItemIsEditable flag, as in this example:
    #
    # class ImageStackTableModel_ImageQuality(ImageStackTableModel):
    #     PROPERTIES = ImageStackTableModel.PROPERTIES + ('image_quality',)
    #     def flags(self, midx):
    #         if midx.column() == self.property_columns['image_quality']:
    #             return Qt.Qt.ItemIsEnabled | Qt.Qt.ItemIsSelectable | Qt.Qt.ItemNeverHasChildren | Qt.Qt.ItemIsEditable
    #         return super().flags(midx)

    PROPERTIES = (
        'visible',
#       'size',
#       'type',
#       'dtype',
        'blend_function',
        'auto_min_max_enabled',
        'tint',
        'getcolor_expression',
        'name',
        'transform_section',)

    def __init__(self, signaling_list, parent=None):
        super().__init__(self.PROPERTIES, signaling_list, parent)
        self._special_data_getters = {
            'visible' : self._getd_visible,
            'auto_min_max_enabled' : self._getd_auto_min_max_enabled,
            'tint' : self._getd_tint}
#           'size' : self._getd_size,
#           'dtype' : self._getd_dtype}
        self._special_flag_getters = {
            'visible' : self._getf__always_checkable,
            'auto_min_max_enabled' : self._getf__always_checkable,
            'tint' : self._getf__always_editable,
            'name' : self._getf__always_editable,
            'getcolor_expression' : self._getf__always_editable,
            'transform_section' : self._getf__always_editable,
            'blend_function' : self._getf__always_editable}
        self._special_data_setters = {
            'visible' : self._setd_visible,
            'auto_min_max_enabled' : self._setd_auto_min_max_enabled}

    # flags #

    def _getf_default(self, midx):
        return Qt.Qt.ItemIsEnabled | Qt.Qt.ItemIsSelectable | Qt.Qt.ItemNeverHasChildren

    def _getf__always_checkable(self, midx):
        return Qt.Qt.ItemIsEnabled | Qt.Qt.ItemIsSelectable | Qt.Qt.ItemNeverHasChildren | Qt.Qt.ItemIsUserCheckable

    def _getf__always_editable(self, midx):
        return Qt.Qt.ItemIsEnabled | Qt.Qt.ItemIsSelectable | Qt.Qt.ItemNeverHasChildren | Qt.Qt.ItemIsEditable

    def flags(self, midx):
        return self._special_flag_getters.get(self.property_names[midx.column()], self._getf_default)(midx)

    # data #

    def _getd__checkable(self, property_name, midx, role):
        if role == Qt.Qt.CheckStateRole:
            return Qt.QVariant(Qt.Qt.Checked if getattr(self.signaling_list[midx.row()], property_name) else Qt.Qt.Unchecked)

    def _getd_visible(self, midx, role):
        return self._getd__checkable('visible', midx, role)

    def _getd_auto_min_max_enabled(self, midx, role):
        return self._getd__checkable('auto_min_max_enabled', midx, role)

    def _getd_tint(self, midx, role):
        if role == Qt.Qt.DecorationRole:
            return Qt.QVariant(Qt.QColor(*(int(c*255) for c in self.signaling_list[midx.row()].tint)))
        elif role == Qt.Qt.DisplayRole:
            return Qt.QVariant(self.signaling_list[midx.row()].tint)

    def _getd_size(self, midx, role):
        if role == Qt.Qt.DisplayRole:
            qsize = self.signaling_list[midx.row()].size
            return Qt.QVariant('{}x{}'.format(qsize.width(), qsize.height()))

    def _getd_dtype(self, midx, role):
        if role == Qt.Qt.DisplayRole:
            return Qt.QVariant(str(self.signaling_list[midx.row()].data.dtype))

    def data(self, midx, role=Qt.Qt.DisplayRole):
        if midx.isValid():
            d = self._special_data_getters.get(self.property_names[midx.column()], super().data)(midx, role)
            if isinstance(d, Qt.QVariant):
                return d
        return Qt.QVariant()

    # setData #

    def _setd__checkable(self, property_name, midx, value, role):
        if role == Qt.Qt.CheckStateRole:
            setattr(self.signaling_list[midx.row()], property_name, value.value())
            return True
        return False

    def _setd_visible(self, midx, value, role):
        return self._setd__checkable('visible', midx, value, role)

    def _setd_auto_min_max_enabled(self, midx, value, role):
        return self._setd__checkable('auto_min_max_enabled', midx, value, role)

    def setData(self, midx, value, role=Qt.Qt.EditRole):
        if midx.isValid():
            return self._special_data_setters.get(self.property_names[midx.column()], super().setData)(midx, value, role)
        return False
