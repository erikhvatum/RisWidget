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
    
    image_flipbook.pages.append(Image(numpy.zeros((400,400,3), dtype=numpy.uint8)))"""
    current_page_changed = Qt.pyqtSignal(int, object)

    def __init__(self, pages=None, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.Qt.WA_DeleteOnClose)
        if pages is None:
            pages = SignalingList(parent=self)
        self._pages = None
        self._page_name_changed_signal_mapper = Qt.QSignalMapper(self)
        self._init_widgets()
        self.pages = pages

    def _init_widgets(self):
        layout = Qt.QVBoxLayout()
        self.setLayout(layout)
        drag_setting_layout = Qt.QHBoxLayout()
        layout.addLayout(drag_setting_layout)
        self._list_view = Qt.QListView(self)
#       self._list_widget.setDragEnabled(True)
#       self._list_widget.setDragDropMode(Qt.QAbstractItemView.InternalMove)
#       self._list_widget.currentRowChanged.connect(self._on_list_widget_current_row_changed)
        layout.addWidget(self._list_widget)

    @property
    def pages(self):
        return self._pages

    @pages.setter
    def pages(self, pages):
        assert isinstance(pages, SignalingList)
        if pages is not self._pages:
            old_models = self.model(), self.selectionModel()
        if isinstance(pages, SignalingList):
            if pages is self._pages:
                return
        else if :
            pages = SignalingList(pages)


    def _on_list_widget_current_row_changed(self, row):
        self.current_page_changed.emit(row, self.pages[row])
