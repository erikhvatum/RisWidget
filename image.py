# The MIT License (MIT)
#
# Copyright (c) 2014 WUSTL ZPLAB
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
from pyagg import fast_hist
from PyQt5 import Qt

class Image:
    def __init__(self, image_data, name):
        self._name = name

        self._data = numpy.asarray(image_data, order='c')
        if self._data.dtype not in (numpy.uint8, numpy.uint16, numpy.float32):
            self._data = self._data.astype(numpy.float32)

        if self._data.ndim == 2:
            self._type = 'g'
            self._histogram, self._min_max = fast_hist.histogram_and_min_max(self._data)
        elif self._data.ndim == 3:
            self._type = {2: 'ga', 3: 'rgb', 4: 'rgba'}.get(self._data.shape[2])
            if self._type is None:
                e = '3D iterable supplied for image_data argument must be either MxNx2 (grayscale with alpha), '
                e+= 'MxNx3 (rgb), or MxNx4 (rgba).'
                raise ValueError(e)
            hmms = numpy.array([fast_hist.histogram_and_min_max(self._data[...,channel_idx]) for channel_idx in range(self._data.shape[2])])
            self._histogram = numpy.vstack(hmms[:,0])
            self._min_max = numpy.vstack(hmms[:,1])
        else:
            raise ValueError('image_data argument must be a 2D (grayscale) or 3D (grayscale with alpha, rgb, or rgba) iterable.')
        self._size = Qt.QSize(self._data.shape[1], self._data.shape[0])
        self._is_grayscale = self._type in ('g', 'ga')

    @property
    def type(self):
        return self._type

    @property
    def dtype(self):
        return self._data.dtype.type

    @property
    def data(self):
        return self._data

    @property
    def name(self):
        return self._name

    @property
    def histogram(self):
        return self._histogram

    @property
    def min_max(self):
        return self._min_max

    @property
    def size(self):
        return self._size

    @property
    def is_grayscale(self):
        return self._is_grayscale
