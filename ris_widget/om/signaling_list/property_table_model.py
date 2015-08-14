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

import ctypes
from PyQt5 import Qt

# Packed ROW_DRAG_MIME_TYPE data consists of a uint64 that is the sending model's Python id() followed by zero or
# more uint64s that are Python id()s of the objects represented by the dropped rows.
ROW_DRAG_MIME_TYPE = 'application/x-row_objects_drag_ZL1AO5IENFKSK9BQP0'

class PropertyTableModel(Qt.QAbstractTableModel):
    def __init__(self, property_names, signaling_list=None, parent=None):
        super().__init__(parent)
        self._signaling_list = None
        self.property_names = list(property_names)
        self.property_columns = {pn : idx for idx, pn in enumerate(self.property_names)}
        assert all(map(lambda p: isinstance(p, str) and len(p) > 0, self.property_names)), 'property_names must be a non-empty iterable of non-empty strings.'
        if len(self.property_names) != len(set(self.property_names)):
            raise ValueError('The property_names argument contains at least one duplicate.')
        self._property_changed_slots = [lambda element, pn=pn: self._on_property_changed(element, pn) for pn in self.property_names]
        self._instance_counts = {}
        self.signaling_list = signaling_list

    def rowCount(self, _=None):
        sl = self.signaling_list
        return 0 if sl is None else len(sl)

    def columnCount(self, _=None):
        return len(self.property_names)

    def data(self, midx, role=Qt.Qt.DisplayRole):
        if midx.isValid() and role in (Qt.Qt.DisplayRole, Qt.Qt.EditRole):
            return Qt.QVariant(getattr(self.signaling_list[midx.row()], self.property_names[midx.column()]))
        return Qt.QVariant()

    def setData(self, midx, value, role=Qt.Qt.EditRole):
        if midx.isValid() and role == Qt.Qt.EditRole:
            setattr(self.signaling_list[midx.row()], self.property_names[midx.column()], value)
            return True
        return False

    def headerData(self, section, orientation, role=Qt.Qt.DisplayRole):
        if orientation == Qt.Qt.Vertical:
            if role == Qt.Qt.DisplayRole and 0 <= section < self.rowCount():
                return Qt.QVariant(section)
        elif orientation == Qt.Qt.Horizontal:
            if role == Qt.Qt.DisplayRole and 0 <= section < self.columnCount():
                return Qt.QVariant(self.property_names[section])
        return Qt.QVariant()

    def removeRows(self, row, count, parent=Qt.QModelIndex()):
#       print('removeRows', row, count, parent)
        try:
            del self.signaling_list[row:row+count]
            return True
        except IndexError:
            return False

    def supportedDropActions(self):
        return Qt.Qt.LinkAction | Qt.Qt.CopyAction | Qt.Qt.MoveAction | Qt.Qt.TargetMoveAction

    def supportedDragActions(self):
        return Qt.Qt.LinkAction | Qt.Qt.CopyAction | Qt.Qt.MoveAction | Qt.Qt.TargetMoveAction

    def canDropMimeData(self, mime_data, drop_action, row, column, parent):
        return super().canDropMimeData(mime_data, drop_action, row, column, parent)
        r = super().canDropMimeData(mime_data, drop_action, row, column, parent)
#       print('canDropMimeData', mime_data, drop_action, row, column, parent, ':', r)
        return r

    def dropMimeData(self, mime_data, drop_action, row, column, parent):
#       print('dropMimeData', row, mime_data.data(ROW_DRAG_MIME_TYPE))
        row_drag = self._decode_row_drag_mime(mime_data)
        if row_drag is None:
            return False
        if row == -1:
            # Qt supplies a value of -1 for row to indicate the drop occurred on the widget background somewhere and not
            # on or adjacent to a row.  We append such a drop to the end of our .signaling_list.
            row = len(self.signaling_list)
        self.signaling_list[row:row] = row_drag[1:]
#       print('dropMimeData', drop_action, row, column, parent, row_drag)
        return True

    def _decode_row_drag_mime(self, mime_data):
        """If mime_data contains packed ROW_DRAG_MIME_TYPE mime data, it is unpacked into a list containing the source model
        and objects dragged, and this tuple is returned.  None is returned otherwise."""
        if mime_data.hasFormat(ROW_DRAG_MIME_TYPE):
            d = mime_data.data(ROW_DRAG_MIME_TYPE)
            ds = Qt.QDataStream(d, Qt.QIODevice.ReadOnly)
            ret = []
            while not ds.atEnd():
                ptr = ds.readUInt64()
#               print('<-{}'.format(ptr))
                ret.append(None if ptr == 0 else ctypes.cast(ptr, ctypes.py_object).value)
            if ret:
                return ret

    def mimeTypes(self):
        return 'application/x-qabstractitemmodeldatalist', ROW_DRAG_MIME_TYPE

    def mimeData(self, midxs):
        mime_data = super().mimeData(midxs)
        if mime_data is None:
            mime_data = Qt.QMimeData()
        d = Qt.QByteArray()
        ds = Qt.QDataStream(d, Qt.QIODevice.WriteOnly)
        ds.writeUInt64(id(self))
        # There is an midx for every column of the dragged row(s), but our ROW_DRAG_MIME_TYPE data should contain only one entry per row
        packed_rows = set()
        for midx in midxs:
            assert midx.isValid()
            row = midx.row()
            if row in packed_rows:
                continue
            packed_rows.add(row)
            ptr = id(self.signaling_list[row])
#           print('->{}'.format(ptr))
            ds.writeUInt64(ptr)
        mime_data.setData(ROW_DRAG_MIME_TYPE, d)
        return mime_data

    @property
    def signaling_list(self):
        return self._signaling_list

    @signaling_list.setter
    def signaling_list(self, v):
        if v is not self._signaling_list:
            if self._signaling_list is not None or v is not None:
                self.beginResetModel()
                try:
                    if self._signaling_list is not None:
                        self._signaling_list.inserting.disconnect(self._on_inserting)
                        self._signaling_list.inserted.disconnect(self._on_inserted)
                        self._signaling_list.replaced.disconnect(self._on_replaced)
                        self._signaling_list.removing.disconnect(self._on_removing)
                        self._signaling_list.removed.disconnect(self._on_removed)
                        self._detach_elements(self._signaling_list)
                    assert len(self._instance_counts) == 0
                    self._signaling_list = v
                    if v is not None:
                        v.inserting.connect(self._on_inserting)
                        v.inserted.connect(self._on_inserted)
                        v.replaced.connect(self._on_replaced)
                        v.removing.connect(self._on_removing)
                        v.removed.connect(self._on_removed)
                        self._attach_elements(v)
                finally:
                    self.endResetModel()

    def _attach_elements(self, elements):
        for element in elements:
            instance_count = self._instance_counts.get(element, 0) + 1
            assert instance_count > 0
            self._instance_counts[element] = instance_count
            if instance_count == 1:
                for property_name, changed_slot in zip(self.property_names, self._property_changed_slots):
                    try:
                        changed_signal = getattr(element, property_name + '_changed')
                        changed_signal.connect(changed_slot)
                    except AttributeError:
                        pass

    def _detach_elements(self, elements):
        for element in elements:
            instance_count = self._instance_counts[element] - 1
            assert instance_count >= 0
            if instance_count == 0:
                for property_name, changed_slot in zip(self.property_names, self._property_changed_slots):
                    try:
                        changed_signal = getattr(element, property_name + '_changed')
                        changed_signal.disconnect(changed_slot)
                    except AttributeError:
                        pass
                del self._instance_counts[element]
            else:
                self._instance_counts[element] = instance_count

    def _on_property_changed(self, element, property_name):
        column = self.property_columns[property_name]
        signaling_list = self.signaling_list
        next_idx = 0
        instance_count = self._instance_counts[element]
        assert instance_count > 0
        for _ in range(instance_count):
            row = signaling_list.index(element, next_idx)
            next_idx = row + 1
            self.dataChanged.emit(self.createIndex(row, column), self.createIndex(row, column))

    def _on_inserting(self, idx, elements):
        self.beginInsertRows(Qt.QModelIndex(), idx, idx+len(elements)-1)

    def _on_inserted(self, idx, elements):
        self.endInsertRows()
        self._attach_elements(elements)

    def _on_replaced(self, idxs, replaced_elements, elements):
        self._detach_elements(replaced_elements)
        self._attach_elements(elements)
        self.dataChanged.emit(self.createIndex(min(idxs), 0), self.createIndex(max(idxs), len(self.property_names) - 1))

    def _on_removing(self, idxs, elements):
        self.beginRemoveRows(Qt.QModelIndex(), min(idxs), max(idxs))

    def _on_removed(self, idxs, elements):
        self.endRemoveRows()
        self._detach_elements(elements)
