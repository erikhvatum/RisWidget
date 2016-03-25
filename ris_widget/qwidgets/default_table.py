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

from PyQt5 import Qt

class DefaultTable(Qt.QTableView):
    """A QTableView with some sensible defaults set, including row-wise selection and a delete key shortcut
    that efficiently removes the selected rows by submitting spans of adjacent rows as slice deletion operations
    where adjacency occurs (and single element deletions otherwise)."""
    def __init__(self, model=None, parent=None):
        super().__init__(parent)
        if model is not None:
            self.setModel(model)
        self.setDragDropOverwriteMode(False)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(Qt.QAbstractItemView.DragDrop)
        self.setSelectionBehavior(Qt.QAbstractItemView.SelectRows)
        self.setSelectionMode(Qt.QAbstractItemView.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(Qt.QAbstractItemView.DragDrop)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(Qt.Qt.LinkAction)
        self.delete_selected_rows_action = Qt.QAction(self)
        self.delete_selected_rows_action.setText('Delete selected rows')
        self.delete_selected_rows_action.triggered.connect(self._on_delete_selected_rows_action_triggered)
        self.delete_selected_rows_action.setShortcut(Qt.Qt.Key_Delete)
        self.delete_selected_rows_action.setShortcutContext(Qt.Qt.WidgetShortcut)
        self.addAction(self.delete_selected_rows_action)

    def _on_delete_selected_rows_action_triggered(self):
        sm = self.selectionModel()
        m = self.model()
        if None in (m, sm):
            return
        midxs = sorted(sm.selectedRows(), key=lambda midx: midx.row())
        # "run" as in consecutive indexes specified as range rather than individually
        runs = []
        run_start_idx = None
        run_end_idx = None
        for midx in midxs:
            if midx.isValid():
                idx = midx.row()
                if run_start_idx is None:
                    run_end_idx = run_start_idx = idx
                elif idx - run_end_idx == 1:
                    run_end_idx = idx
                else:
                    runs.append((run_start_idx, run_end_idx))
                    run_end_idx = run_start_idx = idx
        if run_start_idx is not None:
            runs.append((run_start_idx, run_end_idx))
        for run_start_idx, run_end_idx in reversed(runs):
            m.removeRows(run_start_idx, run_end_idx - run_start_idx + 1)