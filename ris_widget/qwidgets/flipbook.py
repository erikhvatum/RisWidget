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
import sys
from .signaling_list_names_view import SignalingListNamesView
from ..signaling_list import SignalingList
from ..shared_resources import UNIQUE_QLISTWIDGETITEM_TYPE

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

    def __init__(self, pages=None, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.Qt.WA_DeleteOnClose)
        if pages is None:
            pages = SignalingList(parent=self)
        self._drag_and_drop_enabled = False
        self._init_widgets(pages)

    def _init_widgets(self, pages):
        vlayout = Qt.QVBoxLayout()
        self.setLayout(vlayout)
        self.drag_and_drop_enabled_checkbox = Qt.QCheckBox('Enable page drag and drop')
        self.drag_and_drop_enabled_checkbox.setChecked(self._drag_and_drop_enabled)
        self.drag_and_drop_enabled_checkbox.stateChanged.connect(self._on_drag_and_drop_checkbox_toggled)
        vlayout.addWidget(self.drag_and_drop_enabled_checkbox)
        self.list_view_selection_model = None
        self.list_view = SignalingListNamesView(self, pages, self._list_view_item_flags)
        self.list_view.current_row_changed.connect(self.current_page_changed)
        self.list_view.setDragEnabled(True)
        self.list_view.setDragDropMode(Qt.QAbstractItemView.InternalMove)
        vlayout.addWidget(self.list_view)

    @property
    def pages(self):
        return self.list_view.signaling_list

    @pages.setter
    def pages(self, pages):
        self.list_view.signaling_list = pages

    def _list_view_item_flags(self, midx):
        flags = Qt.Qt.ItemIsEnabled | Qt.Qt.ItemIsSelectable | Qt.Qt.ItemNeverHasChildren
        if self._drag_and_drop_enabled:
            flags |= Qt.Qt.ItemIsDropEnabled
            if midx.isValid():
                flags |= Qt.Qt.ItemIsDragEnabled
        return flags

    def _on_drag_and_drop_checkbox_toggled(self, state):
        self._drag_and_drop_enabled = state == Qt.Qt.Checked

    @property
    def drag_and_drop_enabled(self):
        return self._drag_and_drop_enabled

    @drag_and_drop_enabled.setter
    def drag_and_drop_enabled(self, v):
        v = bool(v)
        self._drag_and_drop_enabled = v
        self.drag_and_drop_enabled_checkbox.setChecked(v)
