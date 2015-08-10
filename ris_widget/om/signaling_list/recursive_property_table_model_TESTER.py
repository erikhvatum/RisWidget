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
from .recursive_property_table_model import RecursivePropertyTableModel
from .signaling_list import SignalingList
from ..property import Property

class A(Qt.QObject):
    changed = Qt.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        for property in self.properties:
            property.instantiate(self)

    properties = []

    a = Property(properties, 'a', default_value_callback=lambda _: 'default a')
    b = Property(properties, 'b', default_value_callback=lambda _: 'default b')
    c = Property(properties, 'c', default_value_callback=lambda _: 'default c')
    d = Property(properties, 'd', default_value_callback=lambda _: 'default d')
    e = Property(properties, 'e', default_value_callback=lambda _: 'default e')

    for property in properties:
        exec(property.changed_signal_name + ' = Qt.pyqtSignal(object)')
    del property

class TestModel(RecursivePropertyTableModel):
    def flags(self, midx):
        f = Qt.Qt.ItemIsEnabled | Qt.Qt.ItemIsSelectable | Qt.Qt.ItemNeverHasChildren | Qt.Qt.ItemIsEditable
        f |= Qt.Qt.ItemIsDragEnabled if midx.isValid() else Qt.Qt.ItemIsDropEnabled
        return f

class TestWidget(Qt.QWidget):
    def __init__(self):
        super().__init__()
        vl = Qt.QVBoxLayout()
        self.setLayout(vl)
        self.table = Qt.QTableView()
        self.table.horizontalHeader().setSectionResizeMode(Qt.QHeaderView.ResizeToContents)
        vl.addWidget(self.table)
        hl = Qt.QHBoxLayout()
        vl.addLayout(hl)
        for test_index in range(99999):
            test_fn_name = '_test_{}'.format(test_index)
            if hasattr(self, test_fn_name):
                btn = Qt.QPushButton(str(test_index))
                btn.clicked.connect(getattr(self, test_fn_name))
                hl.addWidget(btn)
            else:
                break
        hl = Qt.QHBoxLayout()
        self.desc_graph_button = Qt.QPushButton('desc tree')
        self.desc_graph_button.clicked.connect(self._on_desc_graph_button_clicked)
        hl.addWidget(self.desc_graph_button)
        self.inst_graph_button = Qt.QPushButton('inst tree')
        self.inst_graph_button.clicked.connect(self._on_inst_graph_button_clicked)
        hl.addWidget(self.inst_graph_button)
        vl.addLayout(hl)
        self.signaling_list = SignalingList()
        self.model = TestModel(
            (
                #'a','a.a','a.a.a','a.b','a.b.c.d.e'
                #'b','b.a','b.a.a','b.b','b.b.b','b.c.d.e',
                #'c','c.a','c.a.a','c.b','c.c.c','c.c.d.e',
                #'d','d.a','d.a.a','d.b','d.d.d','d.c.d.e',
                #'e','e.a','e.a.a','e.b','e.e.e','e.c.d.e'
                'a.a',
                'a.b',
                'a.c',
                'a.d',
                'a.e',
                'b.a',
                'b.b',
                'b.c',
                'b.d',
                'b.e',
                'c.a',
                'c.b',
                'c.c',
                'c.d',
                'c.e',
                'd.a',
                'd.b',
                'd.c',
                'd.d',
                'd.e',
                'e.a',
                'e.b',
                'e.c',
                'e.d',
                'e.e'
            ),
            self.signaling_list)
        self.table.setModel(self.model)

    def _test_0(self):
        self.signaling_list.append(A())
#       self.signaling_list[-1].a = A()
#       self.signaling_list[-1].a.a = 'stuff'

    def _test_1(self):
        self.signaling_list.append(A())
        self.signaling_list[-1].a = A()
        self.signaling_list[-1].b = A()
        self.signaling_list[-1].c = A()
        self.signaling_list[-1].d = A()
        self.signaling_list[-1].e = A()
        self.signaling_list.append(self.signaling_list[-1])
        self.signaling_list.append(A())
        self.signaling_list[-1].a = A()
        self.signaling_list[-1].b = A()
        self.signaling_list[-1].c = A()
        self.signaling_list[-1].d = A()
        self.signaling_list[-1].e = A()
#       self.signaling_list[-1].a.a = A()
#       self.signaling_list[-1].a.a.a = A()
#       self.signaling_list[-1].a.b = A()
#       self.signaling_list[-1].a.b.c = A()
#       self.signaling_list[-1].a.b.c.d = A()
#       self.signaling_list[-1].a.b.c.d.e = A()

    def _on_desc_graph_button_clicked(self):
        self._show_dot_graph(self.model._property_descr_tree_root.dot_graph)

    def _on_inst_graph_button_clicked(self):
        self._show_dot_graph(self.model._property_inst_tree_root.dot_graph)

    def _show_dot_graph(self, dot):
        import io
        import matplotlib.pyplot as plt
        import pygraphviz
        plt.imshow(plt.imread(io.BytesIO(pygraphviz.AGraph(string=dot, directed=True).draw(format='png', prog='dot'))))

if __name__ == '__main__':
    import sys
    app = Qt.QApplication(sys.argv)
    test_widget = TestWidget()
    test_widget.show()
    app.exec_()
