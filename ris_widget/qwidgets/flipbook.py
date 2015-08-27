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
from .. import om

class Flipbook(Qt.QListView):
    """Flipbook: a widget containing a list box showing the name property values of the elements of its pages property.
    Changing which row is selected in the list box causes the current_page_changed signal to be emitted with the newly
    selected page's index and the page itself as parameters.

    The pages property of Flipbook is an SignalingList, a container with a list interface, containing a sequence 
    of elements.  The pages property should be manipulated via the standard list interface, which it implements
    completely.  So, for example, if you have a Flipbook of Images and wish to add an Image to the end of that Flipbook:
    
    image_flipbook.pages.append(Image(numpy.zeros((400,400,3), dtype=numpy.uint8)))

    Signals:
    current_page_changed(index of new page in .pages, the new page)"""
    pass
    current_page_changed = Qt.pyqtSignal(int, object)

    def __init__(self, pages=None, displayed_page_properties=('name', 'type'), parent=None):
        assert pages is None or isinstance(pages, SignalingList)
        super().__init__(parent)
        self.setAttribute(Qt.Qt.WA_DeleteOnClose)
        vlayout = Qt.QVBoxLayout()
        self.setLayout(vlayout)
        self.pages_view = Qt.QTableView()
        self.pages_model = SignalingListPropertyTableModel(displayed_page_properties, pages, self.pages_view)
        self.pages_view.setModel(self.pages_model)
        self.pages_view.selectionModel().currentRowChanged.connect(self._on_pages_model_current_row_changed)
        self.pages_view.horizontalHeader().setSectionResizeMode(Qt.QHeaderView.ResizeToContents)
        self.pages_view.setSelectionBehavior(Qt.QAbstractItemView.SelectRows)
        self.pages_view.setSelectionMode(Qt.QAbstractItemView.SingleSelection)
        vlayout.addWidget(self.pages_view)
        self.delete_current_row_action = Qt.QAction(self)
        self.delete_current_row_action.setText('Delete current row')
        self.delete_current_row_action.triggered.connect(self._on_delete_current_row_action_triggered)
        self.delete_current_row_action.setShortcut(Qt.Qt.Key_Delete)
        self.delete_current_row_action.setShortcutContext(Qt.Qt.WidgetShortcut)
        self.addAction(self.delete_current_row_action)

    def _on_delete_current_row_action_triggered(self):
        sm = self.selectionModel()
        m = self.model()
        if None in (m, sm):
            return
        midx = sm.currentIndex()
        if midx.isValid():
            m.removeRow(midx.row())

    @property
    def pages(self):
        return self.pages_model.signaling_list

    @pages.setter
    def pages(self, pages):
        assert isinstance(pages, SignalingList)
        self.pages_model.signaling_list = pages

    def _on_pages_model_current_row_changed(self, midx, old_midx):
        if midx.isValid():
            row = midx.row()
            self.current_page_changed.emit(row, self.pages[row])
        else:
            self.current_page_changed.emit(-1, None)

#class FlipbookModel():
