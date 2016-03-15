# The MIT License (MIT)
#
# Copyright (c) 2016 WUSTL ZPLAB
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
from .property import Property

def define_struct(properties):

    class Struct(Qt.QObject):
        changed = Qt.pyqtSignal(object)

        def __init__(self, parent=None, **kw):
            super().__init__(parent)
            for property in self.properties:
                property.instantiate(self)
            for kw_n, kw_v in kw.items():
                if not hasattr(self, kw_n):
                    raise NameError('Constructor argument {0} supplied, but this struct has no property named {0}.'.format(kw_n))


        properties = []

        for property in properties:
            exec(property.changed_signal_name + ' = Qt.pyqtSignal(object)')
        del property

        @property
        def dict(self):
            return {p.name : getattr(self, p.name) for p in self.properties}

        @dict.setter
        def dict(self, dict_):
            unset = {p.name for p in self.properties}
            for pn, pv in dict_.items():
                setattr(self, pn, pv)
                unset.remove(pn)
            for pn in unset:
                