# The MIT License (MIT)
#
# Copyright (c) 2014-2015 WUSTL ZPLAB
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

"""Stuff for Qt development and debugging.  RisWidget does not depend on this file;
it is provided for developer use.  Feel free to delete it."""

import numpy
from PyQt5 import Qt

_QEVENT_TYPE_ENUM_VALUE_TO_STRING = None

def qevent_type_value_enum_string(qevent):
    global _QEVENT_TYPE_ENUM_VALUE_TO_STRING
    if _QEVENT_TYPE_ENUM_VALUE_TO_STRING is None:
        _QEVENT_TYPE_ENUM_VALUE_TO_STRING = {}
        for name, value in Qt.QEvent.__dict__.items():
            if type(value) is Qt.QEvent.Type:
                _QEVENT_TYPE_ENUM_VALUE_TO_STRING[value] = name
    try:
        return _QEVENT_TYPE_ENUM_VALUE_TO_STRING[qevent.type()]
    except KeyError:
        return 'UNKNOWN'

def qtransform_to_numpy(t):
    return numpy.array(((t.m11(),t.m12(),t.m13()),(t.m21(),t.m22(),t.m23()),(t.m31(),t.m32(),t.m33())))

def print_qtransform(t):
    print(qtransform_to_numpy(t))
