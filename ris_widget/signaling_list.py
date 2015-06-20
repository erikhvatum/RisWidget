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

from abc import ABCMeta
from collections.abc import MutableSequence
from PyQt5 import Qt

class _QtAbcMeta(Qt.pyqtWrapperType, ABCMeta):
    pass

class SignalingList(Qt.QObject, MutableSequence, metaclass=_QtAbcMeta):
    """SignalingList: a list-like container representing an ordered collection of unique object objects that
    emits change signals when list contents change.

    Pre-change signals (handy for things like virtual table views that must know of certain changes
    before they occur in order to maintain a consistent state):
    * inserting(index where object will be inserted, the object that will be inserted)
    * removing(index of object to be removed, the object that will be removed)

    Post-change signals:
    * inserted(index of inserted object, the object inserted)
    * removed(index of removed object, the object removed)
    * replaced(index of the replaced object, the object that was replaced, the object that replaced it)

    No signals are emitted for objects with indexes that change as a result of inserting or removing
    a preceeding object.  If you do need to maintain a mapping of object -> index for objects in a
    SignalingList, an object -> index dict may be made to shadow the SignalingList by updating that dict
    in response to the SignalingList's inserted, removed, and replaced signals."""

    inserting = Qt.pyqtSignal(int, object)
    removing = Qt.pyqtSignal(int, object)

    inserted = Qt.pyqtSignal(int, object)
    removed = Qt.pyqtSignal(int, object)
    replaced = Qt.pyqtSignal(int, object, object)

    def __init__(self, iterable=None, parent=None):
        Qt.QObject.__init__(self, parent)
        if iterable is None:
            self._list = list()
            self._set = set()
        else:
            self._list = list(iterable)
            self._set = set(iterable)
        assert len(self._list) == len(self._set), 'The iterable argument contains duplicate objects.'

    def __repr__(self):
        r = super().__repr__()[:-1]
        if self._list:
            r += '\n[\n    ' + ',\n    '.join(obj.__repr__() for obj in self._list) + '\n]'
        return r + '>'

    def __iter__(self):
        return iter(self._list)

    def __contains__(self, obj):
        return obj in self._set

    def __len__(self):
        return len(self._list)

    def __getitem__(self, idx):
        return self._list[idx]

    def __setitem__(self, idx, obj):
        assert idx < len(self._list)
        assert obj not in self._set
        replaced_obj = self._list[idx]
        self._list[idx] = obj
        self.replaced.emit(idx, replaced_obj, obj)

    def insert(self, idx, obj):
        assert idx <= len(self._list)
        assert obj not in self._set
        self.inserting.emit(idx, obj)
        self._list.insert(idx, obj)
        self._set.add(obj)
        self.inserted.emit(idx, obj)

    def __delitem__(self, idx):
        obj = self._list[idx]
        self.removing.emit(idx, obj)
        del self._list[idx]
        self._set.remove(obj)
        self.removed.emit(idx, obj)
