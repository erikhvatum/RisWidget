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
    __slots__ = ('model', 'parent', 'name', 'full_name', 'children', '__weakref__')
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
            o += ')'
        return o

    @property
    def is_leaf(self):
        return len(self.children) == 0

class PropertyInstTreeBaseNode:
    __slots__ = ('parent', 'children', 'desc_tree_node', '__weakref__')
    def __init__(self, parent, desc_tree_node):
        self.parent = parent
        self.desc_tree_node = desc_tree_node
        # The .children of the root PropertyInstTreeBaseNode instance maps signaling list element -> PropertyInstTreeElementNode instance,
        # and that PropertyInstTreeElementNode's .value is element.
        #
        # The .children of a PropertyInstTreeElementNode instance maps signaling list element attribute name -> PropertyInstTreeLeafPropNode
        # instance if the attribute is top-level (contains no dots) and PropertyInstTreeIntermediatePropNode otherwise.
        # 
        # The .children of a PropertyInstTreeIntermediatePropNode maps .value attribute name (ie getattr(self.value, list(child.keys())[0])
        # would give child's .value if child is also a PropertyInstTreeIntermediatePropNode or the desired property value if child is an
        # PropertyInstTreeLeafPropNode).
        #
        # PropertyInstTreeLeafPropNode does not make use of its .children attribute.
        self.children = {}

    def __str__(self):
        name = '*ROOT*' if self.desc_tree_node is None else self.desc_tree_node.name
        if len(self.children) == 0:
            o = '<{}>'.format(name)
        else:
            o = '({}'.format(name)
            o += ' : '
            o += ', '.join(str(child) for child in self.children.values())
            o += ')'
        return o

    def get_rec_value(self, pp):
        pv = self.value
#       try:
#           return (True, getattr)
        assert isinstance(pv, (PropertyInstTreeIntermediatePropNode, PropertyInstTreeLeafPropNode))
        return pv.get_rec_value(pp)

    def set_rec_value(self, pp, v):
        pv = self.value
        assert isinstance(pv, (PropertyInstTreeIntermediatePropNode, PropertyInstTreeLeafPropNode))
        return pv.set_rec_value(pp, v)

class PropertyInstTreeElementNode(PropertyInstTreeBaseNode):
    __slots__ = ('value', 'instance_count')
    def __init__(self, parent, element, desc_tree_root):
        super().__init__(parent, desc_tree_root)
        self.value = element
        self.instance_count = 0

    def attach(self):
        assert self.instance_count >= 0
        self.instance_count += 1
        if self.instance_count == 1:
            name = self.desc_tree_node.name
            assert name not in self.parent.children
            self.parent.children[name] = self
            for cname, cdtn in self.desc_tree_node.children.items():
                assert cname == cdtn.name
                assert cname not in self.children
                if cdtn.is_leaf:
                    citn = PropertyInstTreeLeafPropNode(self, cdtn)
                else:
                    citn = PropertyInstTreeIntermediatePropNode(self, cdtn)
                citn.attach()

    def detach(self):
        assert self.instance_count > 0
        self.instance_count -= 1
        if self.instance_count == 0:
            for citn in self.children.values():
                citn.detach()
            assert len(self.children) == 0
            del self.parent.children[self]

    @property
    def inst_tree_element_node(self):
        return self

class PropertyInstTreeIntermediatePropNode(PropertyInstTreeBaseNode):
    __slots__ = ('value')
    def __init__(self, parent, descr_tree_node):
        super().__init__(parent, descr_tree_node)
        self.value = None

    def on_changed(self, obj):
        assert self.parent.value is obj
        self.detach()
        self.attach()

    def attach(self):
        dtn = self.desc_tree_node
        name = dtn.name
        assert name not in self.parent.children
        self.parent.children[name] = self
        try:
            value = self.value = getattr(self.parent.value, name)
        except AttributeError:
            return
        if value is not None:
            for cdtn in dtn.children.values():
                if cdtn.name not in self.children:
                    if cdtn.is_leaf:
                        citn = PropertyInstTreeLeafPropNode(self, cdtn)
                    else:
                        citn = PropertyInstTreeIntermediatePropNode(self, cdtn)
                    citn.attach()
        try:
            changed_signal = getattr(self.parent.value, name + '_changed')
            changed_signal.connect(self.on_changed)
        except AttributeError:
            pass

    def detach(self):
        name = self.desc_tree_node.name
        for citn in self.children.values():
            citn.detach()
        assert len(self.children) == 0
        del self.parent.children[name]
        try:
            changed_signal = getattr(self.parent.value, name + '_changed')
            changed_signal.disconnect(self.on_changed)
        except AttributeError:
            pass

    @property
    def inst_tree_element_node(self):
        return self.parent.inst_tree_element_node

    def get_rec_value(self, pp):
        pv = self.value
        if pv is None:
            return (False,)
        assert isinstance(pv, (PropertyInstTreeIntermediatePropNode, PropertyInstTreeLeafPropNode))
        return pv.get_rec_value(pp)

class PropertyInstTreeLeafPropNode(PropertyInstTreeBaseNode):
    __slots__ = tuple()
    def attach(self):
        name = self.desc_tree_node.name
        assert name not in self.parent.children
        self.parent.children[name] = self
        try:
            changed_signal = getattr(self.parent.value, name + '_changed')
            changed_signal.connect(self.on_changed)
        except AttributeError:
            pass
        self.on_changed(self.parent.value)

    def detach(self):
        name = self.desc_tree_node.name
        del self.parent.children[name]
        try:
            changed_signal = getattr(self.parent.value, name + '_changed')
            changed_signal.disconnect(self.on_changed)
        except AttributeError:
            pass

    def on_changed(self, obj):
        assert self.parent.value is obj
        dtn = self.desc_tree_node
        model = dtn.model
        column = model._property_columns[dtn.full_name]
        iten = self.inst_tree_element_node
        element = iten.value
        signaling_list = model.signaling_list
        next_idx = 0
        instance_count = iten.instance_count
        assert instance_count > 0
        for _ in range(instance_count):
            row = signaling_list.index(element, next_idx)
            next_idx = row + 1
            model.dataChanged.emit(model.createIndex(row, column), model.createIndex(row, column))

    @property
    def inst_tree_element_node(self):
        return self.parent.inst_tree_element_node

class RecursivePropertyTableModel(Qt.QAbstractTableModel):
    def __init__(self, property_names, signaling_list=None, parent=None):
        super().__init__(parent)
        self._signaling_list = None
        self.property_names = property_names = tuple(property_names)
        if len(property_names) == 0 or \
           any(not isinstance(pn, str) or len(pn) == 0 for pn in property_names) or \
           len(set(property_names)) != len(property_names):
            raise ValueError('property_names must be a non-empty iterable of unique, non-empty strings.')
        self._property_paths = [pn.split('.') for pn in property_names]
        self._property_columns = {pn : idx for idx, pn in enumerate(property_names)}
        # Having a null property description tree root node allows property paths with common intermediate components to share
        # nodes up to the point of divergence.  EG, if the property_names argument is ('foo.bar.biff.baz', 'foo.bar.biff.zap'),
        # there will be one PropertyDescrTreeNode for each of foo, foo.bar, and foo.bar.biff.  foo.bar.biff's node will have 
        # two children: foo.bar.biff.baz and foo.bar.biff.zap.
        self._property_descr_tree_root = PropertyDescrTreeNode(self, None, '*ROOT*')
        for pn in property_names:
            self._rec_init_descr_tree(self._property_descr_tree_root, pn.split('.'))
        self._property_inst_tree_root = PropertyInstTreeBaseNode(None, None)
        self.signaling_list = signaling_list

    def _rec_init_descr_tree(self, parent, path):
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
            self._rec_init_descr_tree(node, path[1:])

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
            return Qt.QVariant(self._get_rec_prop_val(self.signaling_list[midx.row()], self._property_paths[midx.column()]))
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
            return self._set_rec_prop_val(self.signaling_list[midx.row()], self._property_paths[midx.column()], value)
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
                    assert len(self._property_inst_tree_root.children) == 0
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
        property_inst_tree_root = self._property_inst_tree_root
        for element in elements:
            try:
                element_inst_node = property_inst_tree_root.children[element]
            except KeyError:
                element_inst_node = PropertyInstTreeElementNode(property_inst_tree_root, element, self._property_descr_tree_root)
            element_inst_node.attach()

    def _detach_elements(self, elements):
        for element in elements:
            element_inst_node = self._property_inst_tree_root.children[element]
            element_inst_node.detach()

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
