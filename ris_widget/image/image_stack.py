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
from .image import Image

class _QtAbcMeta(Qt.pyqtWrapperType, ABCMeta):
    pass

class ImageStack(Qt.QObject, MutableSequence, metaclass=_QtAbcMeta):
    """ImageStack: a list-like container representing an ordered collection of Image objects to be
    blended together sequentially by ImageStackItem.

    Pre-change signals (handy for things like virtual table views that must know of certain changes
    before they occur in order to maintain a consistent state):
    inserting(index where Image will be inserted, the Image that will be inserted)
    removing(index of Image to be removed, the Image that will be removed)

    Post-change signals:
    inserted(index of inserted Image, the Image inserted)
    removed(index of removed Image, the Image removed)
    replaced(index of the replaced Image, the Image that was replaced, the Image that replaced it)

    No signals are emitted for Images with indexes that change as a result of inserting or removing
    a preceeding Image.  If you require this, consider making a sequential container shadowing
    the ImageStack by updating that shadow in response to the inserted, removed, and replaced signals."""

    inserting = Qt.pyqtSignal(int, Image)
    removing = Qt.pyqtSignal(int, Image)

    inserted = Qt.pyqtSignal(int, Image)
    removed = Qt.pyqtSignal(int, Image)
    replaced = Qt.pyqtSignal(int, Image, Image)

    def __init__(self, images_iterable=tuple(), parent_qobject=None):
        Qt.QObject.__init__(self, parent_qobject)
        self._list = list(images_iterable)
        self._set = set(images_iterable)
        assert len(self._list) == len(self._set), 'argument images_iterable contains duplicate images.'

    def __repr__(self):
        r = super().__repr__()[:-1]
        if self._list:
            r += '\n[\n    ' + ',\n    '.join(image.__repr__() for image in self._list) + '\n]'
        return r + '>'

    def __iter__(self):
        return iter(self._list)

    def __contains__(self, image):
        return image in self._set

    def __len__(self):
        return len(self._list)

    def __getitem__(self, idx):
        return self._list[idx]

    def __setitem__(self, idx, image):
        assert isinstance(image, Image)
        assert idx < len(self._list)
        assert image not in self._set
        replaced_image = self._list[idx]
        self._list[idx] = image
        self.replaced.emit(idx, replaced_image, image)

    def insert(self, idx, image):
        assert isinstance(image, Image)
        assert idx <= len(self._list)
        assert image not in self._set
        self.inserting.emit(idx, image)
        self._list.insert(idx, image)
        self._set.add(image)
        self.inserted.emit(idx, image)

    def __delitem__(self, idx):
        image = self._list[idx]
        self.removing.emit(idx, image)
        del self._list[idx]
        self._set.remove(image)
        self.removed.emit(idx, image)
