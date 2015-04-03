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

from .image import Image
from .shared_resources import UNIQUE_QLISTWIDGETITEM_TYPE
from PyQt5 import Qt
import sys

class Flipbook(Qt.QWidget):
    name_changed = Qt.pyqtSignal(object, str, str)
#   image_idx_changed = Qt.pyqtSignal(int)

    def __init__(self, uniqueify_name_func, rw_image_setter_func, images=None, name=None, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.Qt.WA_DeleteOnClose)
        self._uniqueify_name_func = uniqueify_name_func
        self._rw_image_setter_func = rw_image_setter_func
        if name is None:
            name = 'Flipbook'
        self._name = self._uniqueify_name_func(name)
        self.setObjectName(self._name)
        self._init_widgets()
        if images is not None:
            self.append_images(images)

    def _init_widgets(self):
        layout = Qt.QVBoxLayout()
        self.setLayout(layout)
        self._list_widget = Qt.QListWidget(self)
        self._list_widget.currentItemChanged.connect(self._on_list_widget_current_item_changed)
        layout.addWidget(self._list_widget)

    def append_images(self, images):
        self.insert_images(self._list_widget.count(), images)

    def insert_images(self, idx, images):
        """Insert images before table widget item at position idx (counting from zero)."""
        for image in images:
            if not issubclass(type(image), Image):
                image = Image(image)
            self._list_widget.insertItem(idx, _ListWidgetImageItem(image))
            idx += 1
        if self._list_widget.currentRow() == -1:
            self._list_widget.setCurrentRow(0)

    @Qt.pyqtProperty(str, notify=name_changed)
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name is None:
            name = 'Flipbook'
        unique_name = self._uniqueify_name_func(name)
        if unique_name != name:
            print('There is already a Flipbook named "{}" associated with this RisWidget.  Using "{}" instead.'.format(name, unique_name), file=sys.stderr)
        old_name = self._name
        self._name = unique_name
        self.setWindowTitle(unique_name)
        self.setObjectName(unique_name)
        self.parent().setWindowTitle(unique_name)
        self.name_changed.emit(self, old_name, unique_name)

#   @Qt.pyqtProperty(int, notify=image_idx_changed)
#   def image_idx(self):
#       return self._image_idx
#
#   @image_idx.setter
#   def image_idx(self, image_idx):
#       if image_idx != self._image_idx:
#           self.image_idx_changed.emit(image_idx)

    def _on_qobject_name_changed(self, name):
        if name != self._name:
            raise RuntimeError('Flipbook.objectName must match Flipbook.name.')

    def _on_list_widget_current_item_changed(self, new_item, old_item):
        self._rw_image_setter_func(new_item.image)

class _ListWidgetImageItem(Qt.QListWidgetItem):
    QLISTWIDGETITEM_TYPE = UNIQUE_QLISTWIDGETITEM_TYPE()

    def __init__(self, image):
        super().__init__(image.name, type=_ListWidgetImageItem.QLISTWIDGETITEM_TYPE)
        self.image = image
