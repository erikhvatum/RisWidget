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
from .. import om

class BasicSignalingListView(Qt.QListView):
    '''BasicSignalingListView is a Qt.QListView with a constructor that provides a few conveniences for the common use case 
    where a list whose elements (or specified element attribute/property values) are instances of simple types (ex: int, 
    float, str) needs to be presented as an editable list view. 

    If a SignalingList instance is supplied for the list_ argument to BasicSignalingListView.__init__(self, 
    property_name=None, list_=None, model=None, parent=None) and the model argument is None, BasicSignalingListView creates 
    a BasicSignalingListModel and binds it to list_, pass the value of the property_name argument on to 
    BasicSignalingListModel's constructor.  Subsequent edits to that list_ or, equivalently, 
    BasicSignalingListView.signaling_list, will cause its BasicSignalingListView to update (and vice versa). If list_ is 
    some other kind of iterable that does not offer the various change signals required by ListModel, its elements are 
    copied to a new SignalingList.  In both cases, a new BasicSignalingListModel is created, bound to the SignalingList, and 
    set as BasicSignalingListView's model. 

    If BasicSignalingListView.__init__'s model argument is not None, BasicSignalingListView attempts to use that model
    rather than creating a new one, and the values of the list_ and property_name arguments are ignored.

    It is possible to modify BasicSignalingListView's behavior by assigning a custom delegate to some or all of 
    BasicSignalingListView's rows, and futher customization is may be achieved by subclassing BasicSignalingListView. 
    However, it may be cleaner to compose your own ListModel from DragDropBehavior and ListModel and to work with a plain 
    Qt.QListView instance or instance of your own Qt.QListView subclass.

    Usage example:

    from PyQt5 import Qt
    from ris_widget.qwidgets.basic_signaling_list import BasicSignalingListView
    list_view = BasicSignalingListView(list_=[1,2,3,4,'a'])
    list_view.setWindowTitle('Example: BasicSignalingListView + BasicSignalingListModel')
    list_view.show()
    list_view.signaling_list.extend(list('hello world'))
    list_view.signaling_list = sorted(list_view.signaling_list, key=lambda e: str(e))
    def selection_changed(view, row):
        print('newly selected row:', row)
    list_view.current_row_changed.connect(selection_changed)

    Usage example demonstrating horizontal resizing options:

    from PyQt5 import Qt
    from ris_widget.qwidgets.basic_signaling_list import BasicSignalingListView
    class Example(Qt.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle('Example: BasicSignalingListView + BasicSignalingListModel + horizontal scrollbar vs eliding')
            l = Qt.QVBoxLayout()
            self.setLayout(l)
            self.sort_button = Qt.QPushButton('lexicographically sort contents')
            self.sort_button.clicked.connect(self.sort_list_contents)
            l.addWidget(self.sort_button)
            self.scroll_mode_button = Qt.QPushButton('scroll if too wide')
            self.scroll_mode_button.clicked.connect(self.enter_scroll_mode)
            l.addWidget(self.scroll_mode_button)
            self.elide_mode_button = Qt.QPushButton('elide if too wide')
            self.elide_mode_button.clicked.connect(self.enter_elide_mode)
            l.addWidget(self.elide_mode_button)
            self.current_row_num_label = Qt.QLabel()
            l.addWidget(self.current_row_num_label)
            self.list_view = BasicSignalingListView(list_=[1,2,3,4,'a'])
            p = self.list_view.sizePolicy()
            p.setHorizontalPolicy(Qt.QSizePolicy.Ignored)
            self.list_view.setSizePolicy(p)
            self.list_view.setTextElideMode(Qt.Qt.ElideNone)
            self.signaling_list.append('a long string for provoking the need to scroll or elide')
            self.signaling_list.extend(list('hello world'))
            self.list_view.current_row_changed.connect(self.on_current_row_changed)
            self.on_current_row_changed(self.list_view, self.list_view.selectionModel().currentIndex().row())
            l.addWidget(self.list_view)
        @property
        def signaling_list(self):
            return self.list_view.signaling_list
        @signaling_list.setter
        def signaling_list(self, v):
            self.list_view.signaling_list = v
        def sort_list_contents(self):
            self.signaling_list = sorted(self.signaling_list, key=lambda e: str(e))
        def enter_scroll_mode(self):
            self.list_view.setTextElideMode(Qt.Qt.ElideNone)
            self.list_view.setHorizontalScrollBarPolicy(Qt.Qt.ScrollBarAsNeeded)
            self.list_view.scheduleDelayedItemsLayout()
        def enter_elide_mode(self):
            self.list_view.setTextElideMode(Qt.Qt.ElideMiddle)
            self.list_view.setHorizontalScrollBarPolicy(Qt.Qt.ScrollBarAlwaysOff)
            self.list_view.scheduleDelayedItemsLayout()
        def on_current_row_changed(self, view, row):
            self.list_view.setTextElideMode(Qt.Qt.ElideNone)
            self.current_row_num_label.setText('row #: {}'.format(row))
            self.list_view.scheduleDelayedItemsLayout()
    example = Example()
    example.show()

    Signals:
    * current_row_changed(BasicSignalingListView instance, row #): If no row is focused, row # is -1.'''

    current_row_changed = Qt.pyqtSignal(object, int)

    def __init__(self, property_name=None, list_=None, model=None, parent=None):
        super().__init__(parent)
        if model is None:
            if list_ is None:
                list_ = om.SignalingList()
            elif not isinstance(list_, om.SignalingList) and any(not hasattr(list_, signal) for signal in ('inserted', 'removed', 'replaced', 'name_changed')):
                list_ = om.SignalingList(list_)
            model = BasicSignalingListModel(property_name, list_, parent)
        self.setModel(model)
        self.setDragDropMode(Qt.QAbstractItemView.DragDrop)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(Qt.Qt.LinkAction)
        self.setSelectionMode(Qt.QAbstractItemView.SingleSelection)
        self.delete_current_row_action = Qt.QAction(self)
        self.delete_current_row_action.setText('Delete current row')
        self.delete_current_row_action.setShortcut(Qt.Qt.Key_Delete)
        self.delete_current_row_action.setShortcutContext(Qt.Qt.WidgetShortcut)
        self.delete_current_row_action.triggered.connect(self._on_delete_current_row_action)
        self.addAction(self.delete_current_row_action)
    __init__.__doc__ = __doc__

    def _on_delete_current_row_action(self):
        sm = self.selectionModel()
        m = self.model()
        if None in (m, sm):
            return
        smidx = sm.currentIndex()
        if smidx.isValid():
            m.removeRow(smidx.row())

    def _on_selection_model_current_row_changed(self, new_midx, old_midx):
        self.current_row_changed.emit(self, new_midx.row())

    @property
    def signaling_list(self):
        return self.model().signaling_list

    @signaling_list.setter
    def signaling_list(self, list_):
        if not isinstance(list_, om.SignalingList) and any(not hasattr(list_, signal) for signal in ('inserted', 'removed', 'replaced', 'name_changed')):
            list_ = om.SignalingList(list_)
        self.model().signaling_list = list_

    def setSelectionModel(self, sm):
        osm = self.selectionModel()
        if osm is not None:
            try:
                osm.currentRowChanged.disconnect(self._on_selection_model_current_row_changed)
            except TypeError:
                pass
        super().setSelectionModel(sm)
        if sm is not None:
            sm.currentRowChanged.connect(self._on_selection_model_current_row_changed)

class BasicSignalingListModel(om.signaling_list.DragDropModelBehavior, om.signaling_list.ListModel):
    '''BasicSignalingListModel is a composition of DragDropModelBehavior and ListModel that does not
    override any of the behaviors of its components.  This is often adequate; there is built-in
    support for displaying and editing basic data types, and rows containing a supported data type
    are editable by default.'''
    pass
