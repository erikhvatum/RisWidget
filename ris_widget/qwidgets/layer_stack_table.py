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
from .rowwise_table_view import RowwiseTableView
from ..qdelegates.dropdown_list_delegate import DropdownListDelegate
from ..qdelegates.slider_delegate import SliderDelegate
from ..qdelegates.color_delegate import ColorDelegate
from ..qdelegates.checkbox_delegate import CheckboxDelegate
from ..shared_resources import CHOICES_QITEMDATA_ROLE
from .. import om

#TODO: make list items drop targets so that layer contents can be replaced by dropping file on associated item
class LayerStackTableView(RowwiseTableView):
    def __init__(self, layer_stack_table_model, parent=None):
        super().__init__(parent)
        self.horizontalHeader().setSectionResizeMode(Qt.QHeaderView.ResizeToContents)
#       self.horizontalHeader().setStretchLastSection(True)
        self.checkbox_delegate = CheckboxDelegate(self)
#       self.setItemDelegateForColumn(layer_stack_table_model.property_columns['visible'], self.checkbox_delegate)
#       self.setItemDelegateForColumn(layer_stack_table_model.property_columns['auto_min_max_enabled'], self.checkbox_delegate)
        self.blend_function_delegate = DropdownListDelegate(self)
#       self.setItemDelegateForColumn(layer_stack_table_model.property_columns['blend_function'], self.blend_function_delegate)
        self.tint_delegate = ColorDelegate(self)
#       self.setItemDelegateForColumn(layer_stack_table_model.property_columns['tint'], self.tint_delegate)
        self.opacity_delegate = SliderDelegate(0.0, 1.0, self)
#       self.setItemDelegateForColumn(layer_stack_table_model.property_columns['opacity'], self.opacity_delegate)
        self.setSelectionBehavior(Qt.QAbstractItemView.SelectRows)
        self.setSelectionMode(Qt.QAbstractItemView.SingleSelection)
        self.setModel(layer_stack_table_model)
#       self.setEditTriggers(Qt.QAbstractItemView.EditKeyPressed | Qt.QAbstractItemView.SelectedClicked)
        self.delete_current_row_action = Qt.QAction(self)
        self.delete_current_row_action.setText('Delete current row')
        self.delete_current_row_action.triggered.connect(self._on_delete_current_row_action_triggered)
        self.delete_current_row_action.setShortcut(Qt.Qt.Key_Delete)
        self.delete_current_row_action.setShortcutContext(Qt.Qt.WidgetShortcut)
        self.addAction(self.delete_current_row_action)
#       self.verticalHeader().setSectionsMovable(True)
        self.setDragDropOverwriteMode(False)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(Qt.QAbstractItemView.InternalMove)
        self.setDropIndicatorShown(True)

    def _on_delete_current_row_action_triggered(self):
        sm = self.selectionModel()
        m = self.model()
        if None in (m, sm):
            return
        midx = sm.currentIndex()
        if midx.isValid():
            m.removeRow(midx.row())

    def dropEvent(self, event):
        super().dropEvent(event)

class InvertingProxyModel(Qt.QSortFilterProxyModel):
    # Making a full proxy model that reverses/inverts indexes from Qt.QAbstractProxyModel or Qt.QIdentityProxyModel turns
    # out to be tricky but would theoretically be more efficient than this implementation for large lists.  However,
    # a layer stack will never be large enough for the inefficiency to be a concern.
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sort(0, Qt.Qt.DescendingOrder)

    def lessThan(self, lhs, rhs):
        # We want the table upside-down and therefore will be sorting by index (aka row #)
        return lhs.row() < rhs.row()

class LayerStackTableModel(om.signaling_list.PropertyTableModel):
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
        'blend_function',
        'auto_min_max_enabled',
        'tint',
        'opacity',
        'getcolor_expression',
        'name',
#       'image.name'
#       'transform_section',
        )

    def __init__(self, signaling_list, LayerClass, blend_function_choice_to_value_mapping_pairs=None, parent=None):
        super().__init__(self.PROPERTIES, signaling_list, parent)
        if blend_function_choice_to_value_mapping_pairs is None:
            blend_function_choice_to_value_mapping_pairs = [
                ('screen (normal)', 'screen'),
                ('src-over (blend)', 'src-over')]
        else:
            blend_function_choice_to_value_mapping_pairs = list(blend_function_choice_to_value_mapping_pairs)

        # Tack less commonly used / advanced blend function names onto list of dropdown choices without duplicating
        # entries for values that have verbose choice names
        adv_blend_functions = set(LayerClass.BLEND_FUNCTIONS.keys())
        adv_blend_functions -= set(v for c, v in blend_function_choice_to_value_mapping_pairs)
        blend_function_choice_to_value_mapping_pairs += [(v + ' (advanced)', v) for v in sorted(adv_blend_functions)]

        self.blend_function_choices = tuple(c for c, v in blend_function_choice_to_value_mapping_pairs)
        self.blend_function_choice_to_value = dict(blend_function_choice_to_value_mapping_pairs)
        self.blend_function_value_to_choice = {v:c for c, v in blend_function_choice_to_value_mapping_pairs}
        assert \
            len(self.blend_function_choices) == \
            len(self.blend_function_choice_to_value) == \
            len(self.blend_function_value_to_choice),\
            'Duplicate or unmatched (value, the 2nd pair component, does not appear in LayerClass.BLEND_FUNCTIONS) '\
            'entry in blend_function_choice_to_value_mapping_pairs.'

        self._special_data_getters = {
            'visible' : self._getd_visible,
            'auto_min_max_enabled' : self._getd_auto_min_max_enabled,
            'tint' : self._getd_tint,
            'blend_function' : self._getd_blend_function}
        self._special_flag_getters = {
            'visible' : self._getf__always_checkable,
            'auto_min_max_enabled' : self._getf__always_checkable,
            'tint' : self._getf__always_editable,
            'name' : self._getf__always_editable,
            'getcolor_expression' : self._getf__always_editable,
#           'transform_section' : self._getf__always_editable,
            'blend_function' : self._getf__always_editable}
        self._special_data_setters = {
            'visible' : self._setd_visible,
            'auto_min_max_enabled' : self._setd_auto_min_max_enabled,
            'blend_function' : self._setd_blend_function}

    def supportedDropActions(self):
        return Qt.Qt.MoveAction# | Qt.Qt.TargetMoveAction

    def supportedDragActions(self):
        return Qt.Qt.MoveAction# | Qt.Qt.TargetMoveAction

    def canDropMimeData(self, mime_data, drop_action, row, column, parent):
        if column != -1:
            return False
        r = super().canDropMimeData(mime_data, drop_action, row, column, parent)
        print('canDropMimeData', mime_data, drop_action, row, column, parent, ':', r)
        return r

    def dropMimeData(self, mime_data, drop_action, row, column, parent):
        r = super().dropMimeData(mime_data, drop_action, row, column, parent)
        print('dropMimeData', mime_data, drop_action, row, column, parent, ':', r)
        return r

    # flags #

    def _getf_default(self, midx):
        return Qt.Qt.ItemIsEnabled | Qt.Qt.ItemIsSelectable | Qt.Qt.ItemNeverHasChildren

    def _getf__always_checkable(self, midx):
        return Qt.Qt.ItemIsEnabled | Qt.Qt.ItemIsSelectable | Qt.Qt.ItemNeverHasChildren | Qt.Qt.ItemIsUserCheckable

    def _getf__always_editable(self, midx):
        return Qt.Qt.ItemIsEnabled | Qt.Qt.ItemIsSelectable | Qt.Qt.ItemNeverHasChildren | Qt.Qt.ItemIsEditable

    def flags(self, midx):
        f = self._special_flag_getters.get(self.property_names[midx.column()], self._getf_default)(midx)
        # QAbstractItemView calls its model's flags method to determine if an item at a specific model index ("midx") can be
        # dragged about or dropped upon.  If midx is valid, flags(..) is being called for an existing row, and we respond that
        # it is draggable.  If midx is not valid, flags(..) is being called for an imaginary row between two existing rows or
        # at the top or bottom of the table, all of which are valid spots for a row to be inserted, so we respond that the
        # imaginary row in question is OK to drop upon.
        f |= Qt.Qt.ItemIsDragEnabled if midx.isValid() else Qt.Qt.ItemIsDropEnabled
        return f

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

    def _getd_blend_function(self, midx, role):
        if role == CHOICES_QITEMDATA_ROLE:
            return Qt.QVariant(self.blend_function_choices)
        elif role == Qt.Qt.DisplayRole:
            v = self.signaling_list[midx.row()].blend_function
            try:
                c = self.blend_function_value_to_choice[v]
                return Qt.QVariant(c)
            except KeyError:
                Qt.qDebug('No choice for blend function "{}".'.format(v))

    def data(self, midx, role=Qt.Qt.DisplayRole):
        if midx.isValid():
            d = self._special_data_getters.get(self.property_names[midx.column()], super().data)(midx, role)
            if isinstance(d, Qt.QVariant):
                return d
        return Qt.QVariant()

    # setData #

    def _setd__checkable(self, property_name, midx, value, role):
        if isinstance(value, Qt.QVariant):
            value = value.value()
        if role == Qt.Qt.CheckStateRole:
            setattr(self.signaling_list[midx.row()], property_name, value)
            return True
        return False

    def _setd_visible(self, midx, value, role):
        return self._setd__checkable('visible', midx, value, role)

    def _setd_auto_min_max_enabled(self, midx, value, role):
        return self._setd__checkable('auto_min_max_enabled', midx, value, role)

    def _setd_blend_function(self, midx, c, role):
        if isinstance(c, Qt.QVariant):
            c = c.value()
        if role == Qt.Qt.EditRole:
            try:
                v = self.blend_function_choice_to_value[c]
                self.signaling_list[midx.row()].blend_function = v
                return True
            except KeyError:
                Qt.qDebug('No blend function for choice "{}".'.format(c))
        return False

    def setData(self, midx, value, role=Qt.Qt.EditRole):
        if midx.isValid():
            return self._special_data_setters.get(self.property_names[midx.column()], super().setData)(midx, value, role)
        return False
