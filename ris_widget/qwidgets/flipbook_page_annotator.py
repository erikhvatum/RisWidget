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

class _BaseField(Qt.QWidget):
    widget_value_changed = Qt.pyqtSignal(object)

    def __init__(self, field_tuple, parent):
        super().__init__(parent)
        self.name = field_tuple[0]
        self.type = field_tuple[1]
        self.default = self.type(field_tuple[2])
        self.field_tuple = field_tuple
        self._init_widget()

    def _init_widget(self):
        pass

    def update(self, value):
        self.widget.setValue(value)

    def _on_widget_change(self):
        self.widget_value_changed.emit(self.widget.value())

class _StringField(_BaseField):
    def _init_widget(self):

class _IntField(_BaseField):
    def _init_widget(self):
        self.min = self.field_tuple[3] if len(self.field_tuple) >= 4 else None
        self.max = self.field_tuple[4] if len(self.field_tuple) >= 5 else None
        self.widget = Qt.QSpinBox(self)
        if self.min is not None:
            self.widget.setMinimum(self.min)
            if self.max is not None:
                self.widget.setMaximum(self.max)
        self.widget.valueChanged.connect(self._on_widget_change())



class FlipbookPageAnnotator(Qt.QWidget):
    def __init__(self, flipbook, page_metadata_attribute_name, fields, parent=None):
        super().__init__(parent)
        self.flipbook = flipbook
        self.page_metadata_attribute_name = page_metadata_attribute_name
        self.fields = fields
