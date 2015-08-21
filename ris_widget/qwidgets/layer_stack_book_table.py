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
from .. import om

# LayerStackBook: a SignalingList of LayerStacks (conceptual - there is no LayerStackBook class as such)
# LayerStack: a SignalingList of Layers (conceptual - there is no LayerStack class as such)
# Layer: An object with properties describing its appearance when rendered by a LayerStackItem.  Layer.image
#        is one such property, and may be either None or an instance of Image.
# Image: An object wrapping a numpy array representing a rectangular image and including various additional
#        bits of metadata needed to correctly interpret the elements of that numpy array as pixel components
#        in addition to statistical information computed from it.

class LayerStackBookView(Qt.QTableView):
    def __init__(self, layer_stack_book_table_model, displayed_page_properties=('name', 'type'), parent=None):
        assert pages is None or isinstance(pages, SignalingList)
        super().__init__(parent)
        self.setModel(self.layer_stack_book_table_model)
#       self.pages_view.selectionModel().currentRowChanged.connect(self._on_pages_model_current_row_changed)
#       self.pages_view.horizontalHeader().setSectionResizeMode(Qt.QHeaderView.ResizeToContents)
#       self.pages_view.setSelectionBehavior(Qt.QAbstractItemView.SelectRows)
#       self.pages_view.setSelectionMode(Qt.QAbstractItemView.SingleSelection)
#       vlayout.addWidget(self.pages_view)
#       self.delete_current_row_action = Qt.QAction(self)
#       self.delete_current_row_action.setText('Delete current row')
#       self.delete_current_row_action.triggered.connect(self._on_delete_current_row_action_triggered)
#       self.delete_current_row_action.setShortcut(Qt.Qt.Key_Delete)
#       self.delete_current_row_action.setShortcutContext(Qt.Qt.WidgetShortcut)
#       self.addAction(self.delete_current_row_action)
#
#   def _on_delete_current_row_action_triggered(self):
#       sm = self.selectionModel()
#       m = self.model()
#       if None in (m, sm):
#           return
#       midx = sm.currentIndex()
#       if midx.isValid():
#           m.removeRow(midx.row())
#
#   @property
#   def pages(self):
#       return self.pages_model.signaling_list
#
#   @pages.setter
#   def pages(self, pages):
#       assert isinstance(pages, SignalingList)
#       self.pages_model.signaling_list = pages
#
#   def _on_pages_model_current_row_changed(self, midx, old_midx):
#       if midx.isValid():
#           row = midx.row()
#           self.current_page_changed.emit(row, self.pages[row])
#       else:
#           self.current_page_changed.emit(-1, None)
