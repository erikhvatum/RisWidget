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
from ..qdelegates.slider_delegate import SliderDelegate
from ..qdelegates.color_delegate import ColorDelegate
from ..qdelegates.checkbox_delegate import CheckboxDelegate
from ..shared_resources import CHOICES_QITEMDATA_ROLE, FREEIMAGE
from .. import om

#TODO: make list items drop targets so that layer contents can be replaced by dropping file on associated item
class LayerStackTableView(Qt.QTableView):
    def __init__(self, layer_stack_table_model, parent=None):
        super().__init__(parent)
        self.horizontalHeader().setSectionResizeMode(Qt.QHeaderView.ResizeToContents)
#       self.horizontalHeader().setStretchLastSection(True)
        self.checkbox_delegate = CheckboxDelegate(parent=self)
        self.setItemDelegateForColumn(layer_stack_table_model.property_columns['visible'], self.checkbox_delegate)
        self.setItemDelegateForColumn(layer_stack_table_model.property_columns['auto_min_max_enabled'], self.checkbox_delegate)
        self.blend_function_delegate = DropdownListDelegate(self)
        self.setItemDelegateForColumn(layer_stack_table_model.property_columns['blend_function'], self.blend_function_delegate)
        self.tint_delegate = ColorDelegate(self)
        self.setItemDelegateForColumn(layer_stack_table_model.property_columns['tint'], self.tint_delegate)
        self.opacity_delegate = SliderDelegate(0.0, 1.0, self)
        self.setItemDelegateForColumn(layer_stack_table_model.property_columns['opacity'], self.opacity_delegate)
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
#       self.setDragDropOverwriteMode(False)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(Qt.QAbstractItemView.DragDrop)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(Qt.Qt.LinkAction)

    def _on_delete_current_row_action_triggered(self):
        sm = self.selectionModel()
        m = self.model()
        if None in (m, sm):
            return
        midx = sm.currentIndex()
        if midx.isValid():
            m.removeRow(midx.row())

#   def dropEvent(self, event):
#       super().dropEvent(event)

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

class LayerStackTableDragDropBehavior(om.signaling_list.DragDropModelBehavior):
    def handle_dropped_qimage(self, qimage, name, dst_row, dst_column, dst_parent):
        image = self.ImageClass.from_qimage(qimage=qimage, name=name)
        if image is not None:
            layer = self.LayerClass(image=image)
            self.signaling_list[dst_row:dst_row] = [layer]
            return True
        return False

    def handle_dropped_files(self, fpaths, dst_row, dst_column, dst_parent):
        # Note: if the URL is a "file://..." representing a local file, toLocalFile returns a string
        # appropriate for feeding to Python's open() function.  If the URL does not refer to a local file,
        # toLocalFile returns None.
        freeimage = FREEIMAGE(show_messagebox_on_error=True, error_messagebox_owner=None)
        if freeimage is None:
            return False
        # TODO: read images in background thread and display modal progress bar dialog with cancel button
        layers = [self.LayerClass(self.ImageClass(freeimage.read(fpath_str), name=fpath_str)) for fpath_str in (str(fpath) for fpath in fpaths)]
        self.signaling_list[dst_row:dst_row] = layers
        return True

class LayerStackTableModel(LayerStackTableDragDropBehavior, om.signaling_list.RecursivePropertyTableModel):
    # LayerStackTableModel accesses PROPERTIES strictly via self.PROPERTIES and never via LayerStackTableModel.PROPERTIES,
    # meaning that subclasses may safely add or remove columns by overridding PROPERTIES.  For example, adding a column for
    # a sublcassed Images having an "image_quality" property:
    #
    # class LayerStackTableModel_ImageQuality(LayerStackTableModel):
    #     PROPERTIES = ImageStackTableModel.PROPERTIES + ('image.image_quality',)
    #
    # And that's it, provided image_quality is always a plain string and should not be editable.  Making it editable
    # would require adding an entry to self._special_flag_getters.  Alternative .flags may be overridden to activate the
    # Qt.Qt.ItemIsEditable flag, as in this example:
    #
    # class LayerStackTableModel_ImageQuality(LayerStackTableModel):
    #     PROPERTIES = ImageStackTableModel.PROPERTIES + ('image.image_quality',)
    #     def flags(self, midx):
    #         if midx.column() == self.property_columns['image.image_quality']:
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
        'image.dtype',
        'image.size',
        'image.name'
#       'transform_section',
        )

    def __init__(
            self,
            override_enable_auto_min_max_action,
            examine_layer_mode_action,
            ImageClass,
            LayerClass,
            signaling_list=None,
            blend_function_choice_to_value_mapping_pairs=None,
            parent=None
        ):
        super().__init__(self.PROPERTIES, signaling_list, parent)
        self.ImageClass = ImageClass
        self.LayerClass = LayerClass
        self.override_enable_auto_min_max_action = override_enable_auto_min_max_action
        self.override_enable_auto_min_max_action.toggled.connect(self._on_override_enable_auto_min_max_toggled)
        self.examine_layer_mode_action = examine_layer_mode_action
        self.examine_layer_mode_action.toggled.connect(self._on_examine_layer_mode_toggled)
        self._current_row = -1
        if blend_function_choice_to_value_mapping_pairs is None:
            blend_function_choice_to_value_mapping_pairs = [
                ('screen', 'screen'),
                ('src-over (normal)', 'src-over')]
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
            'blend_function' : self._getd_blend_function,
            'image.size' : self._getd_image_size,
            'image.dtype' : self._getd_image_dtype}
        self._special_flag_getters = {
            'visible' : self._getf__always_checkable,
            'auto_min_max_enabled' : self._getf__always_checkable,
            'tint' : self._getf__always_editable,
            'name' : self._getf__always_editable,
            'image.name' : self._getf__always_editable,
            'getcolor_expression' : self._getf__always_editable,
#           'transform_section' : self._getf__always_editable,
            'blend_function' : self._getf__always_editable}
        self._special_data_setters = {
            'visible' : self._setd_visible,
            'auto_min_max_enabled' : self._setd__checkable,
            'blend_function' : self._setd_blend_function}

    # flags #

    def _getf_default(self, midx):
        return super().flags(midx)

    def _getf__always_checkable(self, midx):
        return self._getf_default(midx) | Qt.Qt.ItemIsUserCheckable

    def _getf__always_editable(self, midx):
        return self._getf_default(midx) | Qt.Qt.ItemIsEditable

    def flags(self, midx):
        if midx.isValid():
            return self._special_flag_getters.get(self.property_names[midx.column()], self._getf_default)(midx) | Qt.Qt.ItemIsDragEnabled
        else:
            return self._getf_default(midx) | Qt.Qt.ItemIsDropEnabled

    # data #

    def _getd__checkable(self, midx, role):
        if role == Qt.Qt.CheckStateRole:
            return Qt.QVariant(Qt.Qt.Checked if self.get_cell(midx.row(), midx.column()) else Qt.Qt.Unchecked)

    def _getd_visible(self, midx, role):
        if role == Qt.Qt.CheckStateRole:
            is_checked = self.get_cell(midx.row(), midx.column())
            if self.examine_layer_mode_action.isChecked():
                if self._current_row == midx.row():
                    if is_checked:
                        r = Qt.Qt.Checked
                    else:
                        r = Qt.Qt.PartiallyChecked
                else:
                    r = Qt.Qt.Unchecked
            else:
                if is_checked:
                    r = Qt.Qt.Checked
                else:
                    r = Qt.Qt.Unchecked
            return Qt.QVariant(r)

    def _getd_auto_min_max_enabled(self, midx, role):
        if role == Qt.Qt.CheckStateRole:
            if self.get_cell(midx.row(), midx.column()):
                r = Qt.Qt.Checked
            elif self.override_enable_auto_min_max_action.isChecked():
                r = Qt.Qt.PartiallyChecked
            else:
                r = Qt.Qt.Unchecked
            return Qt.QVariant(r)

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

    def _getd_image_size(self, midx, role):
        if role == Qt.Qt.DisplayRole:
            sz = self.get_cell(midx.row(), midx.column())
            if sz is not None:
                return Qt.QVariant('{}x{}'.format(sz.width(), sz.height()))

    def _getd_image_dtype(self, midx, role):
        if role == Qt.Qt.DisplayRole:
            image = self.signaling_list[midx.row()].image
            if image is not None:
                return Qt.QVariant(str(image.data.dtype))

    def data(self, midx, role=Qt.Qt.DisplayRole):
        if midx.isValid():
            d = self._special_data_getters.get(self.property_names[midx.column()], super().data)(midx, role)
            if isinstance(d, Qt.QVariant):
                return d
        return Qt.QVariant()

    # setData #

    def _setd__checkable(self, midx, value, role):
        if role == Qt.Qt.CheckStateRole:
            if isinstance(value, Qt.QVariant):
                value = value.value()
            return self.set_cell(midx.row(), midx.column(), value)
        return False

    def _setd_visible(self, midx, value, role):
        if role == Qt.Qt.CheckStateRole:
            if isinstance(value, Qt.QVariant):
                value = value.value()
            if value == Qt.Qt.Checked and self.examine_layer_mode_action.isChecked() and self._current_row != midx.row():
                # checkbox_delegate is telling us that, as a result of being hit, we should to check a visibility checkbox
                # that is shown as partially checked.  However, it is shown as partially checked because it is actually checked,
                # but the effect of its checkedness is being supressed because we are in "examine layer" mode and the layer
                # containing the visibility checkbox in question is not the current layer in the layer table.  It is nominally
                # checked, and so toggling it actually means unchecking it.  This is the only instance where an override 
                # causes something checked to appear partially checked, rather than causing something unchecked to appear
                # partially checked.  And, so, in this one instance, we must special case *setting* of an overridable checkbox
                # property.
                value = Qt.Qt.Unchecked
            return self.set_cell(midx.row(), midx.column(), value)
        return False

    def _setd_blend_function(self, midx, c, role):
        if role == Qt.Qt.EditRole:
            if isinstance(c, Qt.QVariant):
                c = c.value()
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

    def _refresh_column(self, column):
        self.dataChanged.emit(self.createIndex(0, column), self.createIndex(len(self.signaling_list)-1, column))

    def _on_override_enable_auto_min_max_toggled(self):
        self._refresh_column(self.property_columns['auto_min_max_enabled'])

    def _on_examine_layer_mode_toggled(self):
        self._refresh_column(self.property_columns['visible'])

    def on_view_current_row_changed(self, row):
        self._current_row = row
        self._on_examine_layer_mode_toggled()
