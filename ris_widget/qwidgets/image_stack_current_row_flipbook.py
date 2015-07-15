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
import numpy
from .flipbook import Flipbook

class ImageStackCurrentRowFlipbook(Flipbook):
    def __init__(self, image_stack, image_stack_selection_model, pages=None, displayed_page_properties=('name', 'type'), parent=None):
        super().__init__(pages, displayed_page_properties, parent)
        self.image_stack = image_stack
        self.image_stack_selection_model = image_stack_selection_model
        self.page_change_behavior_button_group = Qt.QButtonGroup(self)
        self.page_change_image_radio = Qt.QRadioButton('Replace Image')
        self.page_change_data_radio = Qt.QRadioButton('Replace Data')
        self.page_change_data_radio.setChecked(True)
        self.button_to_behavior = {
            self.page_change_image_radio : 'replace image',
            self.page_change_data_radio : 'replace data'}
        self.behavior_to_button = {v : k for k, v in self.button_to_behavior.items()}
        self.page_change_behavior_button_group.addButton(self.page_change_image_radio, 0)
        self.page_change_behavior_button_group.addButton(self.page_change_data_radio, 1)
        self.page_change_behavior_hlayout = Qt.QHBoxLayout()
        self.page_change_behavior_hlayout.addWidget(self.page_change_image_radio)
        self.page_change_behavior_hlayout.addWidget(self.page_change_data_radio)
        self.layout().insertLayout(0, self.page_change_behavior_hlayout)
        self.behavior_to_page_changed_handler = {
            'replace image' : self._replace_image,
            'replace data' : self._replace_data}
        self.current_page_changed.connect(self._on_current_page_changed)

    @property
    def page_change_behavior(self):
        # None is returned if, somehow, neither radio button is checked
        return self.button_to_behavior.get(self.page_change_behavior_button_group.checkedButton())

    @page_change_behavior.setter
    def page_change_behavior(self, v):
        try:
            self.behavior_to_button[v].setChecked(True)
        except KeyError:
            raise KeyError('The value assigned to the .page_change_behavior must be one of the following:\n    ' + '\n    '.join(sorted(self.behavior_to_button.keys())))

    def _on_current_page_changed(self, fb_idx, page):
        if fb_idx < 0:
            return
        page_change_behavior = self.page_change_behavior
        if page_change_behavior is None:
            return
        stack_current_midx = self.image_stack_selection_model.currentIndex()
        if not stack_current_midx.isValid():
            return
        self.behavior_to_page_changed_handler[page_change_behavior](stack_current_midx.row(), page)

    def _replace_image(self, stack_idx, page):
        self.image_stack[stack_idx] = page

    def _replace_data(self, stack_idx, page):
        if page.dtype is numpy.float32:
            self.image_stack[stack_idx].set_data(page.data, float_range=page.range)
        elif page.dtype is numpy.uint16:
            self.image_stack[stack_idx].set_data(page.data, is_twelve_bit=page.is_twelve_bit)
        else:
            self.image_stack[stack_idx].set_data(page.data)
