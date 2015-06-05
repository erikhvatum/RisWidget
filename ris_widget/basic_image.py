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

from .ndimage_statistics import compute_ndimage_statistics, compute_multichannel_ndimage_statistics
import numpy
from PyQt5 import Qt

class BasicImage:
    """An instance of the BasicImage class is a wrapper around a Numpy ndarray representing a single image.  BasicImage's properties
    are all either computed from that ndarray, provide views into that ndarray's data (in the case of .data and .data_T), or, in the
    special cases of .is_twelve_bit for uint16 images and .range for floating-point images, represent unenforced constraints limiting
    the domain of valid values expected to be assumed by elements of the ndarray.

    If an ndarray of supported dtype and striding is supplied as the data argument to BasicImage's constructor or set_data function,
    a reference to that ndarray is kept rather than a copy of it.  In such cases, if the wrapped data is subsequently modified, 
    care must be taken to call the BasicImage's .refresh method before querying its .histogram property or any subclass's properties
    derived from the data, as changes to the wrapped data are not automatically detected."""

    def __init__(self, data, is_twelve_bit=False, float_range=None, shape_is_width_height=True):
        """RisWidget defaults to the convention that the first element of the shape vector of a Numpy
        array represents width.  If you are supplying image data that does not follow this convention,
        specify the argument shape_is_width_height=False, and your image will be displayed correctly
        rather than mirrored over the X/Y axis."""
        self.set_data(data, is_twelve_bit, float_range, shape_is_width_height, False)

    def __repr__(self):
        num_channels = self.num_channels
        return '{}; {}x{}, {} channel{} ({}){}>'.format(
            super().__repr__()[:-1],
            self._size.width(),
            self._size.height(),
            num_channels,
            '' if num_channels == 1 else 's',
            self._type,
            ' (per-channel binary)' if self.is_binary else '')

    def refresh(self):
        if self._is_grayscale:
            self.stats_future = compute_ndimage_statistics(self._data, self._is_twelve_bit, return_future=True)
        else:
            self.stats_future = compute_multichannel_ndimage_statistics(self._data, self._is_twelve_bit, return_future=True)

    def set_data(self, data, is_twelve_bit=False, float_range=None, shape_is_width_height=True):
        self._data = numpy.asarray(data)
        self._is_twelve_bit = is_twelve_bit
        dt = self._data.dtype.type
        if dt not in (numpy.bool8, numpy.uint8, numpy.uint16, numpy.float32):
            self._data = self._data.astype(numpy.float32)
            dt = numpy.float32

        if self._data.ndim == 2:
            if not shape_is_width_height:
                self._data = self._data.transpose(1, 0)
            self._type = 'g'
            bpe = self._data.itemsize
            desired_strides = (bpe, self._data.shape[0]*bpe)
            if desired_strides != self._data.strides:
                d = self._data
                self._data = numpy.ndarray(d.shape, strides=desired_strides, dtype=d.dtype.type)
                self._data.flat = d.flat
            self.stats_future = compute_ndimage_statistics(self._data, self._is_twelve_bit, return_future=True)
        elif self._data.ndim == 3:
            if not shape_is_width_height:
                self._data = self._data.transpose(1, 0, 2)
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
            self.stats_future = compute_multichannel_ndimage_statistics(self._data, self._is_twelve_bit, return_future=True)
        else:
            raise ValueError('data argument must be a 2D (grayscale) or 3D (grayscale with alpha, rgb, or rgba) iterable.')
        self._size = Qt.QSize(self._data.shape[0], self._data.shape[1])
        self._is_grayscale = self._type in ('g', 'ga')

        if dt is numpy.float32:
            if float_range is None:
                # We end up waiting for our futures, now, in this case.  If displaying float images with unspecified range turns out
                # to be common enough that this slowdown is unacceptable, future-ify range computation.
                if self._data.ndim == 2:
                    self._range = tuple(self.min_max)
                else:
                    self._range = self.min_max[0,...].min(), self.min_max[1,...].max()
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

    set_data.__doc__ = __init__.__doc__

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
        """Image data as numpy array in shape = (width, height, [channels]) convention."""
        return self._data

    @property
    def data_T(self):
        """Image data as numpy array in shape = (height, width, [channels]) convention."""
        return self._data.transpose(*(1,0,2)[:self._data.ndim])

    @property
    def histogram(self):
        return self.stats_future.result().histogram

    @property
    def max_histogram_bin(self):
        return self.stats_future.result().max_bin

    @property
    def extremae(self):
        """The actual per-channel minimum and maximum intensity values.  The max intensity value of 4095 for 12-bit-per-channel images
        and the range optionally supplied with floating-point images are not enforced, so it is possible for min and/or max to fall
        outside of the interval represented by the value of the .range property."""
        return self.stats_future.result().min_max_intensity

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

    @property
    def is_binary(self):
        return self.dtype is numpy.bool8

    @property
    def has_alpha_channel(self):
        return self._type[-1] == 'a'

    @property
    def num_channels(self):
        return len(self._type)
