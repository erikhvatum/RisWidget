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

    from ris_widget.qwidgets.basic_signaling_list import BasicSignalingListView
    bslv = BasicSignalingListView(list_=[1,2,3,4,'a'])
    bslv.show()
    bslv.signaling_list.extend(list('hello world'))
    bslv.signaling_list = sorted(bslv.signaling_list, key=lambda e: str(e))'''
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

    @property
    def signaling_list(self):
        return self.model().signaling_list

    @signaling_list.setter
    def signaling_list(self, list_):
        if not isinstance(list_, om.SignalingList) and any(not hasattr(list_, signal) for signal in ('inserted', 'removed', 'replaced', 'name_changed')):
            list_ = om.SignalingList(list_)
        self.model().signaling_list = list_

class BasicSignalingListModel(om.signaling_list.DragDropModelBehavior, om.signaling_list.ListModel):
    '''BasicSignalingListModel is a composition of DragDropModelBehavior and ListModel that does not
    override any of the behaviors of its components.  This is often adequate; there is built-in
    support for displaying and editing basic data types, and rows containing a supported data type
    are editable by default.'''
    pass
