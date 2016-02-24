# The MIT License (MIT)
#
# Copyright (c) 2016 WUSTL ZPLAB
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

from .. import om
from .default_table import DefaultTable
from PyQt5 import Qt

class PointListPickerTableModel(om.signaling_list.DragDropModelBehavior, om.signaling_list.PropertyTableModel):
    pass

class PointListPickerTable(DefaultTable):
    def __init__(self, point_list_picker, model=None, parent=None):
        super().__init__(model, parent)
        self.setWindowTitle('Point Picker')
        self.point_list_picker = point_list_picker
        if model is None:
            self.setModel(PointListPickerTableModel(('x', 'y'), self.point_list_picker.points))
        self.point_list_picker.point_list_replaced.connect(self._on_point_list_replaced)
        self.ignore_selection_change = False
        self.ignore_became_focused = False
        self.point_list_picker.point_is_selected_changed.connect(self._on_point_selection_changed)
        self.selectionModel().selectionChanged.connect(self._on_table_selection_changed)
        self.point_list_picker.point_focused.connect(self._on_point_became_focused)
        self.selectionModel().currentChanged.connect(self._on_table_focus_changed)

    def _on_point_list_replaced(self):
        self.model().signaling_list = self.point_list_picker.points

    def _on_table_selection_changed(self, selected, deselected):
        if not self.ignore_selection_change:
            self.ignore_selection_change = True
            try:
                for midx in selected.indexes():
                    if midx.isValid():
                        self.point_list_picker.point_items[self.point_list_picker.points[midx.row()]].setSelected(True)
                for midx in deselected.indexes():
                    if midx.isValid():
                        self.point_list_picker.point_items[self.point_list_picker.points[midx.row()]].setSelected(False)
            finally:
                self.ignore_selection_change = False

    def _on_point_selection_changed(self, point, isSelected):
        if not self.ignore_selection_change:
            self.ignore_selection_change = True
            try:
                m = self.model()
                sm = self.selectionModel()
                idx = self.point_list_picker.points.index(point)
                flags = Qt.QItemSelectionModel.Rows
                sm.select(m.index(idx, 0), flags | Qt.QItemSelectionModel.Select if isSelected else flags | Qt.QItemSelectionModel.Deselect)
            finally:
                self.ignore_selection_change = False

    def _on_table_focus_changed(self, focused_midx, unfocused_midx):
        if focused_midx.isValid():
            if not self.ignore_became_focused:
                self.ignore_became_focused = True
                try:
                    self.point_list_picker.point_items[self.point_list_picker.points[focused_midx.row()]].setFocus()
                finally:
                    self.ignore_became_focused = False

    def _on_point_became_focused(self, point):
        if not self.ignore_became_focused:
            self.ignore_became_focused = True
            try:
                m = self.model()
                sm = self.selectionModel()
                idx = self.point_list_picker.points.index(point)
                sm.setCurrentIndex(m.index(idx, 0), Qt.QItemSelectionModel.Current)
            finally:
                self.ignore_became_focused = False