# The MIT License (MIT)
#
# Copyright (c) 2015 WUSTL ZPLAB
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
from ..shared_resources import CHOICES_QITEMDATA_ROLE

class DropdownListDelegate(Qt.QStyledItemDelegate):
    def editorEvent(self, event, model, option, midx):
        if not midx.isValid() or option.widget is None:
            return False
        flags = midx.flags()
        item_is_enabled = flags | Qt.Qt.ItemIsEnabled
        item_is_editable = flags | Qt.Qt.ItemIsEditable
        if not item_is_enabled or not item_is_editable:
            return False
        menu = Qt.QMenu(option.widget)
        menu.setAttribute(Qt.Qt.WA_DeleteOnClose)
        choices = midx.data(CHOICES_QITEMDATA_ROLE)
        choice_actions = [menu.addAction(choice) for choice in choices]
        try:
            current_choice = midx.data()
            if isinstance(current_choice, Qt.QVariant):
                current_choice = current_choice.value()
            current_choice_action = choice_actions[choices.index(current_choice)]
            menu.setActiveAction(current_choice_action)
        except ValueError:
            current_choice_action = None
        cell_rect = option.widget.visualRect(midx)
        menu_pos = option.widget.mapToGlobal(Qt.QPoint(cell_rect.left(), (cell_rect.top() + cell_rect.bottom())/2))
        pmidx = Qt.QPersistentModelIndex(midx)
        def on_entry_selected(action):
            if pmidx.isValid():
                model.setData(model.index(pmidx.row(), pmidx.column()), action.text())
        menu.triggered.connect(on_entry_selected)
        menu.popup(menu_pos, current_choice_action)
        return False
