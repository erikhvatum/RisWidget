# The MIT License (MIT)
#
# Copyright (c) 2014-2016 WUSTL ZPLAB
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

import numpy
from PyQt5 import Qt
import sys
import unittest

if Qt.qVersion() == '5.7.0': # TODO: remove this once the Qt 5.7 dev tree code stops leaving the GL context with a pending, useless error
    import OpenGL
    OpenGL.ERROR_CHECKING = False

app = Qt.QApplication(sys.argv)
Qt.QApplication.setAttribute(Qt.Qt.AA_ShareOpenGLContexts)

from .ris_widget import RisWidget

class RisWidgetTestCase(unittest.TestCase):
    def setUp(self):
        self.rw = RisWidget()

    def test_float32(self):
        self.rw.image = numpy.linspace(1, 100, 10000, dtype=numpy.float32).reshape((100,100), order='F')
        Qt.QApplication.processEvents()