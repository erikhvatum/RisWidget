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

from contextlib import ExitStack
from PyQt5 import Qt
from ..shared_resources import ICONS

_ENABLED_CHECKBOX_STATE_ICONS = {
    Qt.Qt.Unchecked : ICONS()['unchecked_box_icon'],
    Qt.Qt.Checked : ICONS()['checked_box_icon'],
    Qt.Qt.PartiallyChecked : ICONS()['pseudo_checked_box_icon'],
    None : ICONS()['wrong_type_checked_box_icon']}

_DISABLED_CHECKBOX_STATE_ICONS = {
    Qt.Qt.Unchecked : ICONS()['disabled_unchecked_box_icon'],
    Qt.Qt.Checked : ICONS()['disabled_checked_box_icon'],
    Qt.Qt.PartiallyChecked : ICONS()['disabled_pseudo_checked_box_icon'],
    None : ICONS()['disabled_wrong_type_checked_box_icon']}

class CheckboxDelegate(Qt.QStyledItemDelegate):
    """CheckboxDelegate: A light way of showing item-model-view cells containing boolean or tri-state-check values as a checkbox
    that is optionally interactive, well centered, and that does not have weird and deceptive appearances when contained within
    non-focused widgets.

    A subtle point regarding CheckboxDelegate: the ItemIsUserCheckable flag* for an item (a "cell", if our model is a table) controls
    how CheckboxDelegate gets, sets, and interprets that item's data.

    If ItemIsUserCheckable is set for an item, then CheckStateRole is specified when retrieving that item's data.  If the return
    value of this query is not Qt.Qt.Unchecked, Qt.Qt.Checked, or Qt.Qt.PartiallyChecked, the "wrong type" icon is displayed. 
    CheckStateRole is used when setting the item's data, and the value supplied is either Qt.Qt.Unchecked or Qt.Qt.Checked.  This
    closely matches the default delegate behavior.

    If ItemIsUserCheckable is not set for an item, the default role, DisplayRole, is used when retrieving the item's data, and
    True, False, Qt.Qt.Unchecked, Qt.Qt.Checked, and Qt.Qt.PartiallyChecked are all acceptable return values for the midx.data() query.
    Any other value results in a "wrong type" icon.  EditRole is used when setting the item's data, and if the item's data query
    call returned a bool, the value supplied is either True or False, whereas if the data query call returned Qt.Qt.Unchecked,
    Qt.Qt.Checked, or Qt.Qt.PartiallyChecked, then the value supplied is either Qt.Qt.Unchecked or Qt.Qt.Checked.

    ____________________________________________________
    * The ItemIsUserCheckable flag is said to be set for midx, a Qt.QModelIndex instance, if "midx.flags() & Qt.Qt.ItemIsUserCheckable"
    evaluates to True."""
    def __init__(self, parent=None, margin=10):
        super().__init__(parent)
        self.margin = margin

    def paint(self, painter, option, midx):
        assert isinstance(painter, Qt.QPainter)
        style = option.widget.style()
        if style is None:
            style = Qt.QApplication.style()
        # Fill cell background in the *exact same manner* as the default delegate.  This is the simplest way to get the correct
        # cell background in all circumstances, including while dragging a row.
        style.drawPrimitive(Qt.QStyle.PE_PanelItemViewItem, option, painter, option.widget)
        if midx.isValid():
            flags = midx.flags()
            if flags & Qt.Qt.ItemIsEnabled and flags & (Qt.Qt.ItemIsUserCheckable | Qt.Qt.ItemIsEditable):
                cs_icons = _ENABLED_CHECKBOX_STATE_ICONS
            else:
                cs_icons = _DISABLED_CHECKBOX_STATE_ICONS
            cs = self._get_checkstate_for_midx(midx, flags)
            icon_rect = self._compute_icon_rect(option, midx)
            painter.drawPixmap(icon_rect, cs_icons[cs].pixmap(icon_rect.size() * option.widget.devicePixelRatio()))

    def _get_checkstate_for_midx(self, midx, flags, from_bool=None):
        if flags & Qt.Qt.ItemIsUserCheckable:
            v = midx.data(Qt.Qt.CheckStateRole)
            if isinstance(v, Qt.QVariant):
                v = v.value()
        else:
            v = midx.data()
            if isinstance(v, Qt.QVariant):
                v = v.value()
            if isinstance(v, bool):
                v = Qt.Qt.Checked if v else Qt.Qt.Unchecked
                if from_bool is not None:
                    from_bool[0] = True
        if v in _ENABLED_CHECKBOX_STATE_ICONS:
            return v

    def sizeHint(self, option, midx):
        if midx.isValid():
            s = 50 + self.margin
            return Qt.QSize(s, s)
        return super().sizeHint(option, midx)

    def createEditor(self, parent, option, midx):
        # We don't make use of "edit mode".  Returning None here prevents double click, enter keypress, etc, from
        # engaging the default delegate behavior of dropping us into string edit mode, wherein a blinking text cursor
        # is displayed in the cell.
        return None

    def editorEvent(self, event, model, option, midx):
        if event.type() == Qt.QEvent.MouseButtonRelease:
            if not self._compute_icon_rect(option, midx).contains(event.pos()):
                return False
        elif event.type() == Qt.QEvent.KeyPress:
            if event.key() not in (Qt.Qt.Key_Space, Qt.Qt.Key_Select):
                return False
        else:
            return False
        flags = model.flags(midx)
        if not flags & Qt.Qt.ItemIsEnabled or not flags & (Qt.Qt.ItemIsUserCheckable | Qt.Qt.ItemIsEditable):
            return False
        from_bool = [False]
        cs = self._get_checkstate_for_midx(midx, flags, from_bool)
        nv = Qt.Qt.Unchecked if cs == Qt.Qt.Checked else Qt.Qt.Checked
        if from_bool[0]:
            nv = nv == Qt.Qt.Checked
        return model.setData(midx, Qt.QVariant(nv), Qt.Qt.CheckStateRole if flags & Qt.Qt.ItemIsUserCheckable else Qt.Qt.EditRole)

    def _compute_icon_rect(self, option, midx):
        cell_rect = option.widget.visualRect(midx)
        min_constraint = min(cell_rect.width(), cell_rect.height()) - self.margin
        return Qt.QStyle.alignedRect(
            option.direction,
            Qt.Qt.AlignCenter,
            Qt.QSize(min_constraint, min_constraint),
            cell_rect)
