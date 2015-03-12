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

from .ndimage_statistics import compute_ndimage_statistics
import numpy
from PyQt5 import Qt

class Image:
    def __init__(self, data, name=None, is_twelve_bit=False, float_range=None):
        self._data = numpy.asarray(data)
        self._is_twelve_bit = is_twelve_bit
        dt = self._data.dtype.type
        if dt not in (numpy.uint8, numpy.uint16, numpy.float32):
            self._data = self._data.astype(numpy.float32)
            dt = numpy.float32

        if self._data.ndim == 2:
            self._data = numpy.asfortranarray(self._data)
            self._type = 'g'
            stats = compute_ndimage_statistics(self._data, self._is_twelve_bit)
            self._histogram = stats.histogram
            self._max_histogram_bin = stats.max_bin
        elif self._data.ndim == 3:
            self._type = {2: 'ga', 3: 'rgb', 4: 'rgba'}.get(self._data.shape[2])
            if self._type is None:
                e = '3D iterable supplied for image_data argument must be either MxNx2 (grayscale with alpha), '
                e+= 'MxNx3 (rgb), or MxNx4 (rgba).'
                raise ValueError(e)
            bpe = self._data.itemsize
            desired_strides = (self._data.shape[2]*bpe, self._data.shape[0]*self._data.shape[2]*bpe, bpe)
            if desired_strides != self._data.strides:
                d = self._data
                self._data = numpy.ndarray(d.shape, strides=desired_strides, dtype=d.dtype.type)
                self._data.flat = d.flat
            statses = [compute_ndimage_statistics(self._data[...,channel_idx], is_twelve_bit) for channel_idx in range(self._data.shape[2])]
            self._histogram = numpy.vstack((stats.histogram for stats in statses))
            self._min_max = numpy.vstack(((stats.min_intensity, stats.max_intensity) for stats in statses))
            self._max_histogram_bin = numpy.hstack((stats.max_bin for stats in statses))
        else:
            raise ValueError('image_data argument must be a 2D (grayscale) or 3D (grayscale with alpha, rgb, or rgba) iterable.')
        self._size = Qt.QSize(self._data.shape[0], self._data.shape[1])
        self._is_grayscale = self._type in ('g', 'ga')

        if dt == numpy.float32:
            if float_range is None:
                self._range = tuple(self._min_max)
            else:
                self._range = float_range
        else:
            if float_range is not None:
                raise ValueError('float_range must not be specified for uint8 or uint16 images.')
            if dt == numpy.uint8:
                self._range = (0, 255)
            elif dt == numpy.uint16:
                if self._is_twelve_bit:
                    self._range = (0, 4095)
                else:
                    self._range = (0, 65535)
            else:
                raise NotImplementedError('Support for another numpy dtype was added without implementing self._range calculation for it...')

    @property
    def type(self):
        return self._type

    @property
    def dtype(self):
        return self._data.dtype.type

    @property
    def strides(self):
        return self._data.strides

    @property
    def data(self):
        return self._data

    @property
    def histogram(self):
        return self._histogram

    @property
    def max_histogram_bin(self):
        return self._max_histogram_bin

    @property
    def min_max(self):
        return self._min_max

    @property
    def range(self):
        """The range of valid values that may be assigned to any channel of any pixel.  For 8-bit-per-channel integer images,
        this is always [0,255], for 12-bit-per-channel integer images, [0,4095], for 16-bit-per-channel integer images, [0,65535].
        For floating point images, this is min/max values for all channels of all pixels, unless specified with the float_range
        argument to our __init__ function."""
        return self._range

    @property
    def size(self):
        return self._size

    @property
    def is_grayscale(self):
        return self._is_grayscale

    @property
    def is_twelve_bit(self):
        return self._is_twelve_bit
