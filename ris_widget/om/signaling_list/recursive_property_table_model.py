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
    __slots__ = ('model', 'parent', 'name', 'full_name', 'children', 'is_seen', '__weakref__')
    def __init__(self, model, parent, name):
        self.model = model
        self.parent = parent
        self.name = name
        self.children = {}
        def getfullpath(node):
            if node.parent is None:
                return []
            else:
                return getfullpath(node.parent) + [node.name]
        self.full_name = '.'.join(getfullpath(self))
        self.is_seen = False

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

    @property
    def dot_graph(self):
        def mrec(n):
            r = 'n{};\n'.format(id(n))
            r+= 'n{}'.format(id(n)) + ' [label="{}"];\n'.format(n.name)
            for c in sorted(n.children.values(), key=lambda cdtn: cdtn.name):
                r+= mrec(c)
                r+= 'n{}'.format(id(n)) + ' -> ' + 'n{};\n'.format(id(c))
            return r
        return 'digraph "DescTree" {\nrankdir="LR";\n' + mrec(self) + '}'

class PropertyInstTreeBaseNode:
    __slots__ = ('parent', 'children', 'desc_tree_node', '__weakref__')
    def __init__(self, parent, desc_tree_node):
        self.parent = parent
        self.desc_tree_node = desc_tree_node
        # PropertyInstTreeRootNode_inst.children is a dict mapping signaling list element -> PropertyInstTreeElementNode instance (whose
        # .value is element).  In other words, for RecursivePropertyTableModel instance r,
        # set(r.signaling_list) == set(r._property_inst_tree_root.children)
        #
        # PropertyInstTreeElementNode_inst.children maps signaling list element attribute name -> PropertyInstTreeLeafPropNode instance where
        # child.desc_tree_node.is_leaf is True and PropertyInstTreeIntermediatePropNode instance otherwise.
        # 
        # PropertyInstTreeIntermediatePropNode_inst.children maps .value attribute name such that getattr(self.value, list(child.keys())[0])
        # would give child's .value if child is also a PropertyInstTreeIntermediatePropNode (itself either an Intermediate or Leaf) or the
        # end property value if child is an PropertyInstTreeLeafPropNode.
        #
        # A PropertyInstTreeLeafPropNode instance does not make use of its .children attribute.
        self.children = {}

    def __str__(self):
        name = self.name
        if len(self.children) == 0:
            o = '<{}>'.format(name)
        else:
            o = '({}'.format(name)
            o += ' : '
            o += ', '.join(str(child) for child in self.children.values())
            o += ')'
        return o

    def path_exists(self, pp):
        try:
            citn = self.children[pp[0]]
        except KeyError:
            return False
        return citn.path_exists(pp[1:])

    @property
    def name(self):
        return self.get_name()

    @property
    def dot_graph(self):
        def mrec(n):
            r = 'n{};\n'.format(id(n))
            r+= 'n{}'.format(id(n)) + ' [label="{}"];\n'.format(n.get_dot_graph_node_label())
            for c in sorted(n.children.values(), key=lambda citn: citn.get_dot_graph_node_label()):
                r+= mrec(c)
                r+= 'n{}'.format(id(n)) + ' -> ' + 'n{};\n'.format(id(c))
            return r
        return 'digraph "InstTree" {\nrankdir="LR";\n' + mrec(self) + '}'

    @property
    def dot_graph_node_label(self):
        return self.get_dot_graph_node_label()

    def get_dot_graph_node_label(self):
        return self.name

    def rec_get(self, pp):
        try:
            citn = self.children[pp[0]]
        except KeyError:
            return
        return citn.rec_get(pp[1:])

    def rec_set(self, pp, v):
        try:
            citn = self.children[pp[0]]
        except KeyError:
            return False
        return citn.rec_set(pp[1:], v)

class PropertyInstTreeRootNode(PropertyInstTreeBaseNode):
    def __init__(self):
        super().__init__(None, None)

    def get_name(self):
        return '*INST ROOT*'

    def get_dot_graph_node_label(self):
        return 'signaling_list[..]'

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
            name = self.name
            assert name == '*ROOT*' or name not in self.parent.children
            self.parent.children[name] = self
            for cname, cdtn in self.desc_tree_node.children.items():
                assert cname == cdtn.name
                assert cname not in self.children
                if hasattr(self.value, cname):
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

    def get_name(self):
        return self.value

    def get_dot_graph_node_label(self):
        instance_count = self.instance_count
        assert instance_count > 0
        model = self.desc_tree_node.model
        signaling_list = model.signaling_list
        element = self.value
        next_idx = 0
        idxs = []
        for _ in range(instance_count):
            row = signaling_list.index(element, next_idx)
            next_idx = row + 1
            idxs.append(str(row))
        return ', '.join(idxs)

# "Named" in PropertyInstTreeNamedNode indicates an instance represents a
# RecursivePropertyTableModel.signaling_list[n].named.property.component.
class PropertyInstTreeNamedNode(PropertyInstTreeBaseNode):
    def on_seen_value_changed(self):
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

class PropertyInstTreeIntermediatePropNode(PropertyInstTreeNamedNode):
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
        name = self.name
        assert name not in self.parent.children
        self.parent.children[name] = self
        try:
            value = self.value = getattr(self.parent.value, name)
        except AttributeError:
            return
        if value is not None:
            for cdtn in dtn.children.values():
                cname = cdtn.name
                if hasattr(value, cname) and cname not in self.children:
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
        if dtn.is_seen:
            self.on_seen_value_changed()

    def detach(self):
        name = self.name
        for citn in list(self.children.values()):
            citn.detach()
        assert len(self.children) == 0
        try:
            del self.parent.children[name]
        except KeyError:
            return
        try:
            changed_signal = getattr(self.parent.value, name + '_changed')
            changed_signal.disconnect(self.on_changed)
        except AttributeError:
            pass
        if self.desc_tree_node.is_seen:
            self.on_seen_value_changed()

    def path_exists(self, pp):
        if len(pp) == 0:
            return True
        return super().path_exists(pp)

    @property
    def inst_tree_element_node(self):
        return self.parent.inst_tree_element_node

    def get_name(self):
        return self.desc_tree_node.name

    def rec_get(self, pp):
        if len(pp) == 0:
            return self.value
        return super().rec_get(pp)

    def rec_set(self, pp, v):
        if len(pp) == 0:
            setattr(self.parent.value, self.name, v)
            return True
        return super().rec_set(pp, v)

class PropertyInstTreeLeafPropNode(PropertyInstTreeNamedNode):
    __slots__ = tuple()
    def __str__(self):
        return '<{} : {}>'.format(self.name, self.value)

    def on_changed(self, obj):
        assert self.parent.value is obj
        self.on_seen_value_changed()

    def attach(self):
        name = self.name
        assert name not in self.parent.children
        self.parent.children[name] = self
        try:
            changed_signal = getattr(self.parent.value, name + '_changed')
            changed_signal.connect(self.on_changed)
        except AttributeError:
            pass
        self.on_seen_value_changed()

    def detach(self):
        name = self.name
        del self.parent.children[name]
        try:
            changed_signal = getattr(self.parent.value, name + '_changed')
            changed_signal.disconnect(self.on_changed)
        except AttributeError:
            pass
        self.on_seen_value_changed()

    def path_exists(self, pp):
        assert len(pp) == 0
        return True

    @property
    def inst_tree_element_node(self):
        return self.parent.inst_tree_element_node

    def get_name(self):
        return self.desc_tree_node.name

    def get_dot_graph_node_label(self):
        return self.name + '\\n{}'.format(self.value)

    @property
    def value(self):
        return getattr(self.parent.value, self.name)

    def rec_get(self, pp):
        assert len(pp) == 0
        return self.value

    def rec_set(self, pp, v):
        assert len(pp) == 0
        setattr(self.parent.value, self.name, v)
        return True

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
        self._property_descr_tree_root = PropertyDescrTreeNode(self, None, '*DESCR ROOT*')
        for pn in property_names:
            self._rec_init_descr_tree(self._property_descr_tree_root, pn.split('.'))
        self._property_inst_tree_root = PropertyInstTreeRootNode()
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
        else:
            node.is_seen = True

    def rowCount(self, _=None):
        sl = self.signaling_list
        return 0 if sl is None else len(sl)

    def columnCount(self, _=None):
        return len(self.property_names)

    def flags(self, midx):
        f = Qt.Qt.ItemIsSelectable | Qt.Qt.ItemNeverHasChildren
        if midx.isValid():
            f |= Qt.Qt.ItemIsDragEnabled
            if self._property_inst_tree_root.children[self.signaling_list[midx.row()]].path_exists(self._property_paths[midx.column()]):
                f |= Qt.Qt.ItemIsEnabled
        else:
            f |= Qt.Qt.ItemIsDropEnabled
        return f

    def data(self, midx, role=Qt.Qt.DisplayRole):
        if midx.isValid() and role in (Qt.Qt.DisplayRole, Qt.Qt.EditRole):
            # NB: Qt.QVariant(None) is equivalent to Qt.QVariant(), so the case where eitn.rec_get returns None does not require
            # special handling
            eitn = self._property_inst_tree_root.children[self.signaling_list[midx.row()]]
            return Qt.QVariant(eitn.rec_get(self._property_paths[midx.column()]))
        return Qt.QVariant()

    def setData(self, midx, value, role=Qt.Qt.EditRole):
        if midx.isValid() and role == Qt.Qt.EditRole:
            eitn = self._property_inst_tree_root.children[self.signaling_list[midx.row()]]
            return eitn.rec_set(self._property_paths[midx.column()], value)
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
