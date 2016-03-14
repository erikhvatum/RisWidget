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
from .. import om

class FlipbookPageAnnotator(Qt.QWidget):
    def __init__(self, flipbook, page_metadata_attribute_name, fields, parent=None):
        super().__init__(parent)
        self.flipbook = flipbook
        self.page_metadata_attribute_name = page_metadata_attribute_name
        flipbook.pages_list_replaced.connect(self._on_flipbook_pages_list_replaced())
        self._attach_pages_list(flipbook.pages)

    def _attach_pages_list(self):
        self._attached_pages_list = self.flipbook.pages

    def _detach_pages_list(self):
        if self._attached_pages_list is not None:
            self._attached_pages_list.inserted.disconnect(self._on_flipbook_pages_inserted)
            self._attached_pages_list.removed.disconnect(self._on_flipbook_pages_removed)
            self._attached_pages_list = None

    def _on_flipbook_pages_list_replaced(self):
        self._detach_pages_list()
        self._attach_pages_list()

    def _on_flipbook_pages_inserted(self, idx, pages):


    def _on_flipbook_pages_removed(self, idxs, pages):
        pass