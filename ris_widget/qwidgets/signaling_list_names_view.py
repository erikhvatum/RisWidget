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
from ..signaling_list import SignalingList

class SignalingListNamesView(Qt.QListView):
    def __init__(self, parent=None, signaling_list=None):
        super().__init__(parent)
        self._signaling_list = None
        self.signaling_list = signaling_list

    @property
    def signaling_list(self):
        return self._signaling_list

    @signaling_list.setter
    def signaling_list(self, sl):
        if sl is not self._signaling_list:
            old_models = self.model(), self.selectionModel()
            self.setModel(None if sl is None else _SignalingListNamesModel(self, sl))
            self._signaling_list = sl
            for old_model in old_models:
                if old_model is not None:
                    old_model.deleteLater()

class _SignalingListNamesModel(Qt.QAbstractListModel):
    def __init__(self, parent, signaling_list):
        assert isinstance(signaling_list, SignalingList)
        super().__init__(parent)
        signaling_list.inserting.connect(self._on_inserting)
        signaling_list.inserted.connect(self._on_inserted)
        signaling_list.removing.connect(self._on_removing)
        signaling_list.removed.connect(self._on_removed)
        self._element_name_changed_signal_mapper = Qt.QSignalMapper(self)
        self._element_name_changed_signal_mapper.mapped[Qt.QObject].connect(self._on_element_name_changed)
        for element in signaling_list:
            self._element_name_changed_signal_mapper.setMapping(element, element)
            element.name_changed.connect(self._element_name_changed_signal_mapper.map)
        self.signaling_list = signaling_list

    def rowCount(self, _=None):
        return len(self.signaling_list)

    def data(self, midx, role=Qt.Qt.DisplayRole):
        if role == Qt.Qt.DisplayRole and midx.column() == 0:
            return Qt.QVariant(self.signaling_list[midx.row()].name)
        return Qt.QVariant()

    def flags(self, midx):
        return Qt.Qt.ItemIsEnabled | Qt.Qt.ItemIsSelectable | Qt.Qt.ItemNeverHasChildren #| Qt.Qt.ItemIsDragEnabled

    def _on_inserting(self, idx, element):
        self.beginInsertRows(Qt.QModelIndex(), idx, idx)

    def _on_inserted(self, idx, element):
        self.endInsertRows()
        self._element_name_changed_signal_mapper.setMapping(element, element)
        element.name_changed.connect(self._element_name_changed_signal_mapper.map)

    def _on_removing(self, idx, element):
        self.beginRemoveRows(Qt.QModelIndex(), idx, idx)

    def _on_removed(self, idx, element):
        self.endRemoveRows()
        element.name_changed.disconnect(self._element_name_changed_signal_mapper.map)
        self._element_name_changed_signal_mapper.removeMappings(element)

    def _on_element_name_changed(self, element):
        idx = self.signaling_list.index(element)
        self.dataChanged.emit(self.createIndex(idx), self.createIndex(idx), (Qt.Qt.DisplayRole,))
