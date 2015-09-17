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
    def paint(self, painter, option, midx):
        assert isinstance(painter, Qt.QPainter)
        if midx.isValid():
            cell_rect = self._compute_checkbox_rect(option)
            min_constraint = min(cell_rect.width(), cell_rect.height())
            icon_rect = Qt.QStyle.alignedRect(
                option.direction,
                Qt.Qt.AlignCenter,
                Qt.QSize(min_constraint, min_constraint),
                cell_rect)
            flags = midx.flags()
            item_is_enabled = flags | Qt.Qt.ItemIsEnabled
            item_is_checkable = flags | Qt.Qt.ItemIsUserCheckable
            cs_icons = _ENABLED_CHECKBOX_STATE_ICONS if item_is_enabled and item_is_checkable else _DISABLED_CHECKBOX_STATE_ICONS
            v = midx.data(Qt.Qt.CheckStateRole)
            if isinstance(v, Qt.QVariant):
                v = v.value()
            if v not in cs_icons:
                v = None
            dpi_ratio = Qt.QApplication.instance().desktop().devicePixelRatio()
            painter.drawPixmap(icon_rect, cs_icons[v].pixmap(icon_rect.width() * dpi_ratio, icon_rect.height() * dpi_ratio))

    def sizeHint(self, option, midx):
        if midx.isValid():
            margin = Qt.QApplication.style().pixelMetric(Qt.QStyle.PM_FocusFrameHMargin) + 1
            return Qt.QSize(50 + margin, 50 + margin)
        return super().sizeHint(option, midx)

    def editorEvent(self, event, model, option, midx):
        flags = model.flags(midx)
        if not flags & Qt.Qt.ItemIsUserCheckable or not flags & Qt.Qt.ItemIsEnabled:
            return False
        value = midx.data(Qt.Qt.CheckStateRole)
        if isinstance(value, Qt.QVariant):
            if value.isValid():
                value = value.value()
            else:
                return False
        if event.type() == Qt.QEvent.MouseButtonRelease:
            if not self._compute_checkbox_rect(option).contains(event.pos()):
                return False
        elif event.type() == Qt.QEvent.KeyPress:
            if event.key() not in (Qt.Qt.Key_Space, Qt.Qt.Key_Select):
                return False
        else:
            return False
        return model.setData(midx, Qt.QVariant(Qt.Qt.Unchecked if value == Qt.Qt.Checked else Qt.Qt.Checked), Qt.Qt.CheckStateRole)

    def _compute_checkbox_rect(self, option):
        text_margin = Qt.QApplication.style().pixelMetric(Qt.QStyle.PM_FocusFrameHMargin) + 1
        return Qt.QStyle.alignedRect(
            option.direction,
            Qt.Qt.AlignCenter,
            Qt.QSize(
                option.decorationSize.width() + 5,
                option.decorationSize.height()),
            Qt.QRect(
                option.rect.x() + text_margin,
                option.rect.y(),
                option.rect.width() - text_margin - text_margin,
                option.rect.height()))
