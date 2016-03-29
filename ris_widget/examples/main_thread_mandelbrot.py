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
from PyQt5 import Qt

class Mandelbrot(Qt.QWidget):
    def __init__(self, image, width=2048, height=2048, parent=None):
        super().__init__(parent)
        self._running = False
        self.image = image
        self.width, self.height = width, height
        self._init_lut()
        l = Qt.QVBoxLayout()
        self.setLayout(l)
        ll = Qt.QHBoxLayout()
        self.total_iterations_label = Qt.QLabel('Iterations: ')
        ll.addWidget(self.total_iterations_label)
        self.total_iterations_spinner = Qt.QSpinBox()
        self.total_iterations_spinner.setRange(2,65535)
        ll.addWidget(self.total_iterations_spinner)
        l.addLayout(ll)
        ll = Qt.QHBoxLayout()
        self.current_iteration_label = Qt.QLabel('Current iteration: ')
        ll.addWidget(self.current_iteration_label)
        self.current_iteration_counter = Qt.QLCDNumber(5)
        self.current_iteration_counter.display('')
        ll.addWidget(self.current_iteration_counter)
        l.addLayout(ll)
        self.run_button = Qt.QPushButton('Run')
        self.run_button.setCheckable(True)
        self.run_button.setChecked(False)
        self.run_button.setSizePolicy(Qt.QSizePolicy.MinimumExpanding)
        l.addWidget(self.run_button)
        self.run_button.toggled.connect(self._on_run_toggled)

    def _init_lut(self):
        xr = numpy.linspace(0, 2*numpy.pi, 65536, True)
        xg = xr + 2*numpy.pi/3
        xb = xr + 4*numpy.pi/3
        self._lut = ((numpy.dstack(list(map(numpy.sin, (xr, xg, xb)))) + 1) / 2).astype(numpy.float32)

    def _on_run_toggled(self, run):
        if run == self._running:
            return
        self.total_iterations_label.setEnabled(not run)
        self.total_iterations_spinner.setEnabled(not run)
        self.current_iteration_label.setEnabled(run)
        self.current_iteration_counter.setEnabled(run)
        if run:
            self._exec()
        else:
            self._running = False

    def _exec(self):
        y, x = numpy.ogrid[-1.4:1.4:self.height * 1j, -2:0.8:self.width * 1j]
        c = x + y * 1j
        z = c
        max_iterations = self.total_iterations_spinner.value()
        data = numpy.zeros(z.shape, dtype=numpy.float32) + numpy.float32(max_iterations)
        self.current_iteration_counter.display(0)
        self.image.set(self._lut[data.astype(numpy.uint16)], data_shape_is_width_height=False)
        self._running = True
        for current_iteration in range(1, max_iterations+1):
            if not self._running:
                break
            self.current_iteration_counter.display(current_iteration)
        self.current_iteration_counter.display('')
        self.run_button.setChecked(False) # Causes self._on_run_toggled(False) call

def mandelbrot( h,w, maxit=20 ):
    '''Returns an image of the Mandelbrot fractal of size (h,w).
    Adapted from https://scipy.github.io/old-wiki/pages/Tentative_NumPy_Tutorial/Mandelbrot_Set_Example.html
    '''

    y,x = numpy.ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]
    c = x+y*1j
    z = c
    divtime = maxit + numpy.zeros(z.shape, dtype=numpy.float32)

    for i in range(maxit):
        z  = z**2 + c
        diverge = z*numpy.conj(z) > 2**2            # who is diverging
        div_now = diverge & (divtime==maxit)  # who is diverging now
        divtime[div_now] = i                  # note when
        z[diverge] = 2                        # avoid diverging too much

    return divtime