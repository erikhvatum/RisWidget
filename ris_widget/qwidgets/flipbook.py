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
from ..signaling_list.signaling_list import SignalingList
from ..signaling_list.signaling_list_property_table_model import SignalingListPropertyTableModel

class Flipbook(Qt.QWidget):
    """Flipbook: a widget containing a list box showing the name property values of the elements of its pages property.
    Changing which row is selected in the list box causes the current_page_changed signal to be emitted with the newly
    selected page's index and the page itself as parameters.

    The pages property of Flipbook is an SignalingList, a container with a list interface, containing a sequence 
    of elements.  The pages property should be manipulated via the standard list interface, which it implements
    completely.  So, for example, if you have a Flipbook of Images and wish to add an Image to the end of that Flipbook:
    
    image_flipbook.pages.append(Image(numpy.zeros((400,400,3), dtype=numpy.uint8)))

    Signals:
    current_page_changed(index of new page in .pages, the new page)"""
    current_page_changed = Qt.pyqtSignal(int, object)

    def __init__(self, pages=None, displayed_page_properties=('name', 'type'), parent=None):
        assert pages is None or isinstance(pages, SignalingList)
        super().__init__(parent)
        self.setAttribute(Qt.Qt.WA_DeleteOnClose)
        vlayout = Qt.QVBoxLayout()
        self.setLayout(vlayout)
        self.drag_and_drop_enabled_checkbox = Qt.QCheckBox('Enable page drag and drop')
        self.drag_and_drop_enabled_checkbox.setChecked(False)
        self.drag_and_drop_enabled_checkbox.stateChanged.connect(self._on_drag_and_drop_checkbox_toggled)
        vlayout.addWidget(self.drag_and_drop_enabled_checkbox)
        self.pages_view = Qt.QTableView()
        self.pages_model = _TableModel(displayed_page_properties, pages, self, self.pages_view)
        self.pages_view.setModel(self.pages_model)
        self.pages_view.selectionModel().currentRowChanged.connect(self._on_pages_model_current_row_changed)
        self.pages_view.setDragDropMode(Qt.QAbstractItemView.InternalMove)
        self.pages_view.horizontalHeader().setSectionResizeMode(Qt.QHeaderView.ResizeToContents)
        self.pages_view.setSelectionBehavior(Qt.QAbstractItemView.SelectRows)
        self.pages_view.setSelectionMode(Qt.QAbstractItemView.SingleSelection)
        vlayout.addWidget(self.pages_view)

    @property
    def pages(self):
        return self.pages_model.signaling_list

    @pages.setter
    def pages(self, pages):
        assert isinstance(pages, SignalingList)
        self.pages_model.signaling_list = pages

    def _list_view_item_flags(self, midx):
        flags = Qt.Qt.ItemIsEnabled | Qt.Qt.ItemIsSelectable | Qt.Qt.ItemNeverHasChildren
        if self._drag_and_drop_enabled:
            flags |= Qt.Qt.ItemIsDropEnabled
            if midx.isValid():
                flags |= Qt.Qt.ItemIsDragEnabled
        return flags

    def _on_drag_and_drop_checkbox_toggled(self, state):
        self._drag_and_drop_enabled = state == Qt.Qt.Checked
        pv = self.pages_view
        if self._drag_and_drop_enabled:
            pv.setDragEnabled(True)
            pv.setAcceptDrops(True)
            pv.setDropIndicatorShown(True)
            pv.setSelectionMode(Qt.QAbstractItemView.ExtendedSelection)
            pv.supported_drop_actions = Qt.Qt.MoveAction
        else:
            pv.setDragEnabled(False)
            pv.setAcceptDrops(False)
            pv.setDropIndicatorShown(False)
            pv.setSelectionMode(Qt.QAbstractItemView.SingleSelection)
            pv.supported_drop_actions = 0

    @property
    def drag_and_drop_enabled(self):
        return self.drag_and_drop_enabled_checkbox.isChecked()

    @drag_and_drop_enabled.setter
    def drag_and_drop_enabled(self, v):
        self.drag_and_drop_enabled_checkbox.setChecked(v)

    def _on_pages_model_current_row_changed(self, old_midx, midx):
        if midx.isValid():
            row = midx.row()
            self.current_page_changed.emit(row, self.pages[row])
        else:
            self.current_page_changed.emit(-1, None)

# TODO: connect to Image.image_changed; emit changes across row as in commented _on_image_changed in image_stack_widget.py
class _TableModel(SignalingListPropertyTableModel):
    def __init__(self, property_names, signaling_list, flipbook, parent):
        super().__init__(property_names, signaling_list, parent)
        self.flipbook = flipbook

    def flags(self, midx):
        flags = Qt.Qt.ItemIsEnabled | Qt.Qt.ItemIsSelectable | Qt.Qt.ItemNeverHasChildren
        if self.flipbook.drag_and_drop_enabled:
            flags |= Qt.Qt.ItemIsDropEnabled
            if midx.isValid():
                flags |= Qt.Qt.ItemIsDragEnabled
        return flags
