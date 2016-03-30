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

import numpy
from pathlib import Path
from PyQt5 import Qt
from ..qwidgets.qml_item_widget import QmlItemWidget

class MandelbrotWidget(QmlItemWidget):
    _QML_REGISTERED = False

    def __init__(self, image, width=1024, height=1024, parent=None):
        self.mandelbrot = Mandelbrot(image, width, height)
        super().__init__(Path(__file__).parent / 'main_thread_mandelbrot.qml', parent)

    def on_loaded(self):
        super().on_loaded()
        if not MandelbrotWidget._QML_REGISTERED:
            Qt.qmlRegisterType(Mandelbrot, 'MandelbrotImport', 1, 0, 'Mandelbrot')
            MandelbrotWidget._QML_REGISTERED = True
        self.rootObject().setProperty('mandelbrot', self.mandelbrot)

class Mandelbrot(Qt.QObject):
    isRunningChanged = Qt.pyqtSignal()
    iterationCountChanged = Qt.pyqtSignal()
    currentIterationChanged = Qt.pyqtSignal()

    def __init__(self, image, width, height, parent=None):
        super().__init__(parent)
        self._isRunning = False
        self._image = image
        self._width, self._height = width, height
        self._iterationCount = 60
        self._init_lut()

    def _init_lut(self):
        xr = numpy.linspace(0, 2*numpy.pi, 65536, True)
        xg = xr + 2*numpy.pi/3
        xb = xr + 4*numpy.pi/3
        self._lut = ((numpy.dstack(list(map(numpy.sin, (xr, xg, xb)))) + 1) / 2).astype(numpy.float32)

    @Qt.pyqtSlot()
    def update_gui(self):
        Qt.QApplication.processEvents()

    @Qt.pyqtSlot()
    def run(self):
        self._stopRequested = False
        # Begin Mandelbrot drawing code adapted from https://scipy.github.io/old-wiki/pages/Tentative_NumPy_Tutorial/Mandelbrot_Set_Example.html
        y, x = numpy.ogrid[-1.4:1.4:self._height * 1j, -2:0.8:self._width * 1j]
        c = x + y * 1j
        z = c
        # End
        data = numpy.zeros(z.shape, dtype=numpy.float32) + numpy.float32(self._iterationCount)
        # self._image.set(self._lut[data.astype(numpy.uint16)], data_shape_is_width_height=False)
        self._isRunning = True
        self.isRunningChanged.emit()
        self._currentIteration = 0
        self.currentIterationChanged.emit()
        self.update_gui()
        try:
            for current_iteration in range(1, self._iterationCount+1):
                if self._stopRequested:
                    break
                # Begin Mandelbrot drawing code adapted from https://scipy.github.io/old-wiki/pages/Tentative_NumPy_Tutorial/Mandelbrot_Set_Example.html
                z = z ** 2 + c
                diverge = z * numpy.conj(z) > 2 ** 2  # who is diverging
                div_now = diverge & (data == self._iterationCount)  # who is diverging now
                data[div_now] = current_iteration  # note when
                z[diverge] = 2  # avoid diverging too much
                # End
                self._image.set(
                    numpy.take(self._lut, (65535*data/data.max()).astype(numpy.uint16), axis=1)[0], # Pretty cool...
                    data_shape_is_width_height=False)
                self._currentIteration = current_iteration
                self.currentIterationChanged.emit()
                self.update_gui()
        finally:
            self._isRunning = False
            self.isRunningChanged.emit()

    def setIsRunning(self, v):
        if v != self._isRunning:
            if v:
                self.run()
            else:
                self._stopRequested = True

    @Qt.pyqtProperty(bool, fset=setIsRunning, notify=isRunningChanged)
    def isRunning(self):
        return self._isRunning

    def setIterationCount(self, v):
        v = int(v)
        if self._iterationCount != v:
            self._iterationCount = v
            self.iterationCountChanged.emit()

    @Qt.pyqtProperty(int, fset=setIterationCount, notify=iterationCountChanged)
    def iterationCount(self):
        return self._iterationCount

    @Qt.pyqtProperty(int, notify=currentIterationChanged)
    def currentIteration(self):
        return self._currentIteration