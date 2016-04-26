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

class ContextualInfoItemBase(Qt.QGraphicsObject):
    def __init__(self, parent_item):
        super().__init__(parent_item)
        self.contextual_info = None

    def set_contextual_info(self, contextual_info):
        if contextual_info is not self.contextual_info:
            if self.contextual_info is not None:
                self.contextual_info.value_changed.disconnect(self._handle_contextual_info_change)
            contextual_info.value_changed.connect(self._handle_contextual_info_change)
            self.contextual_info = contextual_info
            self._handle_contextual_info_change()

    def clear_contextual_info(self, source):
        if self.contextual_info is not None and source is self.contextual_info.source:
            self.contextual_info.value_changed.disconnect(self._handle_contextual_info_change)
            self.contextual_info = None
            self._handle_contextual_info_change()

    def _handle_contextual_info_change(self):
        raise NotImplementedError()