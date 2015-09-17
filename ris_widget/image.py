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

from .ndimage_statistics import ndimage_statistics
import ctypes
import numpy
from PyQt5 import Qt
import warnings

class Image(Qt.QObject):
    """An instance of the Image class is a wrapper around a Numpy ndarray representing a single image, plus some related attributes and
    a .name.

    If an ndarray of supported dtype, shape, and striding is supplied as the data argument to Image's constructor or set_data function,
    a reference to that ndarray is kept rather than a copy of it.  In such cases, if the wrapped data is subsequently modified, 
    care must be taken to call the Image's .refresh method before querying, for example, its .histogram property as changes to the
    data are not automatically detected.

    The attributes maintained by an Image instance fall into the following categories:
        * Properties that represent aspects of the ._data ndarray or its contents: .data, .data_T, .dtype, .strides, .histogram, .max_histogram_bin,
        .extremae, .range.  These may not be assigned to directly; .set_data(..) is intended for replacing .data and causing attendant properties
        to update, while .refresh() may be used in the case where ._data is a reference or view to an ndarray whose contents have been modified.
        The .data_changed signal is emitted to indicate that the value of any or all of these properties has changed.
        * Plain instance attributes computed by .refresh() and .set_data(..): .size, .is_grayscale, .num_channels, .has_alpha_channel, .is_binary,
        .is_twelve_bit.  Although nothing prevents assigning over these attributes, doing so is not advised.  The .data_changed signal is emitted
        to indicate that the value of any or all of these attributes has changed.
        * Properties with individual change signals: .name.  It is safe to assign None in addition anything else that str(..) accepts as its argument
        to .name.  When the value of .name is modified, .name_changed is emitted.

    Additionally, emission of .data_changed or .name_changed causes emission of .changed."""
    changed = Qt.pyqtSignal(object)
    data_changed = Qt.pyqtSignal(object)
    name_changed = Qt.pyqtSignal(object)

    def __init__(self, data, parent=None, is_twelve_bit=False, float_range=None, shape_is_width_height=True, name=None):
        """RisWidget defaults to the convention that the first element of the shape vector of a Numpy
        array represents width.  If you are supplying image data that does not follow this convention,
        specify the argument shape_is_width_height=False, and your image will be displayed correctly
        rather than mirrored over the X/Y axis."""
        super().__init__(parent)
        self.data_changed.connect(self.changed)
        self.objectNameChanged.connect(lambda: self.name_changed.emit(self))
        self.name_changed.connect(self.changed)
        self.set_data(data, is_twelve_bit, float_range, shape_is_width_height, False, name)

    @classmethod
    def from_qimage(cls, qimage, parent=None, is_twelve_bit=False, name=None):
        if not qimage.isNull() and qimage.format() != Qt.QImage.Format_Invalid:
            if qimage.hasAlphaChannel():
                desired_format = Qt.QImage.Format_RGBA8888
                channel_count = 4
            else:
                desired_format = Qt.QImage.Format_RGB888
                channel_count = 3
            if qimage.format() != desired_format:
                qimage = qimage.convertToFormat(desired_format)
            if channel_count == 3:
                # 24-bit RGB QImage rows are padded to 32-bit chunks, which we must match
                row_stride = qimage.width() * 3
                row_stride += 4 - (row_stride % 4)
                padded = numpy.ctypeslib.as_array(ctypes.cast(int(qimage.bits()), ctypes.POINTER(ctypes.c_uint8)), shape=(qimage.height(), row_stride))
                padded = padded[:, qimage.width() * 3].reshape((qimage.height(), qimage.width(), 3))
                npyimage = numpy.empty((qimage.height(), qimage.width(), 3), dtype=numpy.uint8)
                npyimage.flat = padded.flat
            else:
                npyimage = numpy.ctypeslib.as_array(
                    ctypes.cast(int(qimage.bits()), ctypes.POINTER(ctypes.c_uint8)),
                    shape=(qimage.height(), qimage.width(), channel_count))
            if qimage.isGrayscale():
                # Note: Qt does not support grayscale with alpha channels, so we don't need to worry about that case
                npyimage=npyimage[...,0]
            return cls(data=npyimage.copy(), parent=parent, is_twelve_bit=is_twelve_bit, shape_is_width_height=False, name=name)

    def __repr__(self):
        num_channels = self.num_channels
        name = self.name
        return '{}; {}, {}x{}, {} channel{} ({}){}>'.format(
            super().__repr__()[:-1],
            'with name "{}"'.format(name) if name else 'unnamed',
            self.size.width(),
            self.size.height(),
            num_channels,
            '' if num_channels == 1 else 's',
            self.type,
            ' (per-channel binary)' if self.is_binary else '')

    def refresh(self):
        # Assumption: only contents of ._data may have changed, not its size, shape, striding, or dtype.
        if self.is_grayscale:
            self.stats_future = ndimage_statistics.compute_ndimage_statistics(self._data, self.is_twelve_bit, return_future=True)
        else:
            self.stats_future = ndimage_statistics.compute_multichannel_ndimage_statistics(self._data, self.is_twelve_bit, return_future=True)
        self.data_changed.emit(self)

    def set_data(self, data, is_twelve_bit=False, float_range=None, shape_is_width_height=True, keep_name=True, name=None):
        """If keep_name is True, the existing name is not changed, and the value supplied for the name argument is ignored.
        If keep_name is False, the existing name is replaced with the supplied name or is cleared if supplied name is None
        or an empty string."""
        self._data = numpy.asarray(data)
        self.is_twelve_bit = is_twelve_bit
        dt = self._data.dtype.type
        if dt not in (numpy.bool8, numpy.uint8, numpy.uint16, numpy.float32):
            raise ValueError('The "data" argument must produce a numpy ndarray of dtype bool8, uint8, uint16, or float32 when '
                             'passed through numpy.asarray(data).  So, if data is, itself, an ndarray, then data.dtype must be '
                             'one of bool8, uint8, uint16, or float32.')
        if self._data.ndim == 2:
            if not shape_is_width_height:
                self._data = self._data.transpose(1, 0)
            self.type = 'G'
            bpe = self._data.itemsize
            desired_strides = (bpe, self._data.shape[0]*bpe)
            if desired_strides != self._data.strides:
                d = self._data
                self._data = numpy.ndarray(d.shape, strides=desired_strides, dtype=d.dtype.type)
                self._data.flat = d.flat
            if dt is not numpy.float32:
                self.stats_future = ndimage_statistics.compute_ndimage_statistics(self._data, twelve_bit=is_twelve_bit, return_future=True)
        elif self._data.ndim == 3:
            if not shape_is_width_height:
                self._data = self._data.transpose(1, 0, 2)
            self.type = {2: 'Ga', 3: 'rgb', 4: 'rgba'}.get(self._data.shape[2])
            if self.type is None:
                e = '3D iterable supplied for image_data argument must be either MxNx2 (grayscale with alpha), '
                e+= 'MxNx3 (rgb), or MxNx4 (rgba).'
                raise ValueError(e)
            bpe = self._data.itemsize
            desired_strides = (self._data.shape[2]*bpe, self._data.shape[0]*self._data.shape[2]*bpe, bpe)
            if desired_strides != self._data.strides:
                d = self._data
                self._data = numpy.ndarray(d.shape, strides=desired_strides, dtype=d.dtype.type)
                self._data.flat = d.flat
            if dt is not numpy.float32:
                self.stats_future = ndimage_statistics.compute_multichannel_ndimage_statistics(self._data, twelve_bit=self.is_twelve_bit, return_future=True)
        else:
            raise ValueError('data argument must be a 2D (grayscale) or 3D (grayscale with alpha, rgb, or rgba) iterable.')

        self.size = Qt.QSize(self._data.shape[0], self._data.shape[1])
        self.is_grayscale = self.type in ('G', 'Ga')
        self.num_channels = len(self.type)
        self.has_alpha_channel = self.type[-1] == 'a'
        self.is_binary = self.dtype is numpy.bool8

        if dt is numpy.float32:
            if float_range is None:
                extremae = ndimage_statistics.find_min_max(self._data)
                if self._data.ndim == 2:
                    self._range = float(extremae[0]), float(extremae[1])
                else:
                    self._range = float(extremae[0,...].min()), float(extremae[1,...].max())
                if self._range[0] == self._range[1]:
                    if self._range[0] == 0:
                        self._range = 0.0, 1.0
                    else:
                        self._range = self._range[0], self.range[0]*2
            else:
                assert float_range[1] > float_range[0]
                self._range = float_range
            self.stats_future = ndimage_statistics.compute_ranged_histogram(
                self._data,
                self._range if self._data.ndim == 2 else numpy.vstack((self._range,) * self.num_channels),
                258,
                with_overflow_bins=True,
                return_future=True,
                make_ndimage_statistics_tuple=True)
        else:
            if float_range is not None:
                raise ValueError('float_range must not be specified for uint8 or uint16 images.')
            if dt == numpy.uint8:
                self._range = 0, 255
            elif dt == numpy.uint16:
                if self.is_twelve_bit:
                    self._range = 0, 4095
                else:
                    self._range = 0, 65535
            else:
                raise NotImplementedError('Support for another numpy dtype was added without implementing self._range calculation for it...')

        if not keep_name:
            self.name = name
        self.data_changed.emit(self)

    set_data.__doc__ = __init__.__doc__

    def generate_contextual_info_for_pos(self, x, y, include_image_name=True):
        sz = self.size
        if 0 <= x < sz.width() and 0 <= y < sz.height():
            type_ = self.type
            num_channels = self.num_channels
            mst = ''
            if include_image_name:
                name = self.name
                if name:
                    mst += '"' + name + '" '
            mst+= 'x:{} y:{} '.format(x, y)
            vt = '(' + ' '.join((c + ':{}' for c in self.type)) + ')'
            if num_channels == 1:
                vt = vt.format(self.data[x, y])
            else:
                vt = vt.format(*self.data[x, y])
            return mst+vt

    name = property(
        Qt.QObject.objectName,
        lambda self, name: self.setObjectName('' if name is None else name),
        doc='Property proxy for QObject::objectName Qt property, which is directly accessible via the objectName getter and '
            'setObjectName setter.  Upon change, objectNameChanged is emitted.')

    @property
    def data(self):
        """Image data as numpy array in shape = (width, height, [channels]) convention."""
        return self._data

    @property
    def data_T(self):
        """Image data as numpy array in shape = (height, width, [channels]) convention."""
        return self._data.transpose(*(1,0,2)[:self._data.ndim])

    @property
    def dtype(self):
        return self._data.dtype.type

    @property
    def strides(self):
        return self._data.strides

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
        argument to our set_data function."""
        return self._range
