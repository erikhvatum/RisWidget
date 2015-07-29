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

class PropertyDescrTreeNode:
    __slots__ = ('model', 'parent', 'name', 'full_name', 'children')

    def __init__(self, model, parent, name):
        self.model = model
        self.parent = parent
        self.name = name
        self.children = {}
        def getfullpath(node):
            if node.parent is None:
                # We have reached the root node.  Stop recursion and do not include the root node's name.
                return []
            else:
                return getfullpath(node.parent) + [node.name]
        self.full_name = '.'.join(getfullpath(self))

    def __str__(self):
        if self.is_leaf:
            o = '<{}>'.format(self.name)
        else:
            o = '({}'.format(self.name)
            o += ' : '
            o += ', '.join(str(child) for child in self.children.values())
            o+= ')'
        return o

    @property
    def is_leaf(self):
        return len(self.children) == 0

class PropertyInstTreeNode(Qt.QObject):
    def __init__(self, parent, desc_tree_node):
        self.parent = parent
        self.desc_tree_node = desc_tree_node
        self.children = {}

    def on_property_changed(self, element):
        dtn = self.desc_tree_node
        is_leaf = dtn.is_leaf
        if is_leaf:
            #
        else:

    @property
    def trunk_branch_element(self):
        # In order to emit a dataChanged signal for a leaf, it is necessary to determine which rows
        # in the table contain the leaf value in question.  Row # corresponds to signaling_list index,
        # and signaling_list actually contains branches from the trunk.



class RecursivePropertyTableModel(Qt.QAbstractTableModel):
    def __init__(self, property_names, signaling_list=None, parent=None):
        super().__init__(parent)
        self._signaling_list = None
        property_names = list(property_names)
        if len(property_names) == 0 or \
           any(not isinstance(pn, str) or len(pn) == 0 for pn in property_names) or \
           len(set(property_names)) != len(property_names):
            raise ValueError('The property_names must be a non-empty iterable of unique, non-empty strings.')
        self.property_columns = {pn : idx for idx, pn in enumerate(property_names)}
        # Having a null property description tree root node allows property paths with common intermediate components to share
        # nodes up to the point of divergence.  EG, if the property_names argument is ('foo.bar.biff.baz', 'foo.bar.biff.zap'),
        # there will be one PropertyDescrTreeNode for each of foo, foo.bar, and foo.bar.biff.  foo.bar.biff's node will have 
        # two children: foo.bar.biff.baz and foo.bar.biff.zap.
        self.property_descr_tree_root = PropertyDescrTreeNode(self, None, '*ROOT*')
        for pn in property_names:
            self._rec_init_desc_tree(self.property_descr_tree_root, pn.split('.'))
        self.property_inst_tree_root = PropertyInstTreeNode(None, self.property_descr_tree_root)
        self.signaling_list = signaling_list

    def _rec_init_desc_tree(self, parent, path):
        assert len(path) > 0
        name = path[0]
        try:
            node = parent.children[name]
        except KeyError:
            node = parent.children[name] = PropertyDescrTreeNode(self, parent, name)
        # Even if parent already has a node for this name, there must be a divergence away from the existing branch or this
        # function would not be called in the first place (entirely duplicate paths are not allowed).  So, we may only
        # skip the next line if we have already arrived at the leaf.
        if len(path) > 1:
            self._rec_init_desc_tree(node, path[1:])

    def rowCount(self, _=None):
        sl = self.signaling_list
        return 0 if sl is None else len(sl)

    def columnCount(self, _=None):
        return len(self.property_names)

    def _get_rec_prop_val(self, pv, pp):
        if pv is not None:
            if len(pp) == 1:
                return getattr(pv, pp[0])
            return self._get_rec_prop_val(getattr(pv, pp[0]), pp[1:])

    def data(self, midx, role=Qt.Qt.DisplayRole):
        if midx.isValid() and role in (Qt.Qt.DisplayRole, Qt.Qt.EditRole):
            # NB: Qt.QVariant(None) is equivalent to Qt.QVariant(), so the case where self._get_rect_prop_val(..) returns None does
            # not require special handling
            return Qt.QVariant(self._get_rec_prop_val(self.signaling_list[midx.row()], self.property_namepaths[midx.column()].path))
        return Qt.QVariant()

    def _set_rec_prop_val(self, pv, pp, v):
        if pv is not None:
            if len(pp) == 1:
                setattr(pv, pp[0], v)
                return True
            return self._set_rec_prop_val(getattr(pv, pp[0]), pp[1:], v)
        return False

    def setData(self, midx, value, role=Qt.Qt.EditRole):
        if midx.isValid() and role == Qt.Qt.EditRole:
            return self._set_rec_prop_val(self.signaling_list[midx.row()], self.property_names[midx.column()].path, value)
        return False

    def headerData(self, section, orientation, role=Qt.Qt.DisplayRole):
        if orientation == Qt.Qt.Vertical:
            if role == Qt.Qt.DisplayRole and 0 <= section < self.rowCount():
                return Qt.QVariant(section)
        elif orientation == Qt.Qt.Horizontal:
            if role == Qt.Qt.DisplayRole and 0 <= section < self.columnCount():
                return Qt.QVariant(self.property_namepaths[section].name)
        return Qt.QVariant()

    def removeRows(self, row, count, parent=Qt.QModelIndex()):
        try:
            del self.signaling_list[row:row+count]
            return True
        except IndexError:
            return False

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

    def _on_leaf_property_changed(self, ):

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
