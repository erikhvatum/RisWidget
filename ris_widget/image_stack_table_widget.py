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

#TODO: make list items drop targets so that layer contents can be replaced by dropping file on associated item
class ImageStackTableWidget(Qt.QTableView):
    def __init__(self, parent=None, image_stack=None):
        super().__init__(parent)
        self._image_stack = None
        self.image_stack = image_stack
        self.horizontalHeader().setSectionResizeMode(Qt.QHeaderView.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)

    @property
    def image_stack(self):
        return self.image_stack

    @image_stack.setter
    def image_stack(self, i_s):
        if i_s is not self._image_stack:
            old_models = self.model(), self.selectionModel()
            self.setModel(None if i_s is None else _ImageStackTableModel(self, i_s))
            self._image_stack = i_s
            for old_model in old_models:
                if old_model is not None:
                    old_model.deleteLater()

class _ImageStackTableModel(Qt.QAbstractTableModel):
    def __init__(self, parent, image_stack):
        super().__init__(parent)
        self.headers = 'idx', 'muted', 'name', 'size', 'type', 'dtype'
        image_stack.inserting.connect(self._on_inserting)
        image_stack.inserted.connect(self._on_inserted)
        image_stack.removing.connect(self._on_removing)
        image_stack.removed.connect(self._on_removed)
        self._image_changed_signal_mapper = Qt.QSignalMapper(self)
        self._image_changed_signal_mapper.mapped[Qt.QObject].connect(self._on_image_changed)
        for image in image_stack:
            self._image_changed_signal_mapper.setMapping(image, image)
            image.changed.connect(self._image_changed_signal_mapper.map)
        self.image_stack = image_stack

    def rowCount(self, _=None):
        return len(self.image_stack)

    def columnCount(self, _=None):
        return 6

    def data(self, midx, role=Qt.Qt.DisplayRole):
        if role != Qt.Qt.DisplayRole\
          or not 0 <= midx.row() < self.rowCount()\
          or not 0 <= midx.column() < self.columnCount():
            return Qt.QVariant()
        if midx.column() == 0:
            return Qt.QVariant(midx.row())
        if midx.column() == 1:
            return Qt.QVariant(self.image_stack[midx.row()].mute_enabled)
        if midx.column() == 2:
            return Qt.QVariant(self.image_stack[midx.row()].name)
        if midx.column() == 3:
            return Qt.QVariant(self.image_stack[midx.row()].size)
        if midx.column() == 4:
            return Qt.QVariant(self.image_stack[midx.row()].type)
        if midx.column() == 5:
            return Qt.QVariant(str(self.image_stack[midx.row()].dtype))

    def headerData(self, section, orientation, role=Qt.Qt.DisplayRole):
        if role != Qt.Qt.DisplayRole\
          or orientation != Qt.Qt.Horizontal\
          or not 0 <= section < len(self.headers):
            return Qt.QVariant()
        return Qt.QVariant(self.headers[section])

    def _on_inserting(self, idx, image):
        self.beginInsertRows(Qt.QModelIndex(), idx, idx)

    def _on_inserted(self, idx, image):
        self.endInsertRows()
        self._image_changed_signal_mapper.setMapping(image, image)
        image.changed.connect(self._image_changed_signal_mapper.map)

    def _on_removing(self, idx, image):
        self.beginRemoveRows(Qt.QModelIndex(), idx, idx)

    def _on_removed(self, idx, image):
        self.endRemoveRows()
        image.changed.disconnect(self._image_changed_signal_mapper.map)
        self._image_changed_signal_mapper.removeMappings(image)

    def _on_image_changed(self, image):
        idx = self.image_stack.index(image)
        self.dataChanged.emit(self.createIndex(idx, 0), self.createIndex(idx, self.columnCount()))
