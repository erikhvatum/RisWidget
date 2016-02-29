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

import ctypes
import numpy
from PyQt5 import Qt
from .ndimage_statistics import ndimage_statistics
from . import om

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
        * mask: None or a 2D bool or uint8 ndarray with neither dimension smaller than the corresponding dimension of im.  If mask is not None, only image
        pixels with non-zero mask counterparts contribute to the histogram.  Mask pixels outside of im have no impact.  If mask is None, all image pixels
        are included.
        * Plain instance attributes computed by .refresh() and .set(..): .size, .is_grayscale, .num_channels, .has_alpha_channel, .is_binary,
        .is_twelve_bit.  Although nothing prevents assigning over these attributes, doing so is not advised.  The .data_changed signal is emitted
        to indicate that the value of any or all of these attributes has changed.
        * Properties with individual change signals: .name.  It is safe to assign None in addition anything else that str(..) accepts as its argument
        to .name.  When the value of .name is modified, .name_changed is emitted.

    Additionally, emission of .data_changed, mask_changed, or .name_changed causes emission of .changed."""
    changed = Qt.pyqtSignal(object)
    data_changed = Qt.pyqtSignal(object)
    mask_changed = Qt.pyqtSignal(object)
    name_changed = Qt.pyqtSignal(object)

    def __init__(
            self,
            data,
            mask=None,
            parent=None,
            is_twelve_bit=False,
            specified_float_range=None,
            shape_is_width_height=True,
            mask_shape_is_width_height=True,
            name=None):
        """RisWidget defaults to the convention that the first element of the shape vector of a Numpy
        array represents width.  If you are supplying image data that does not follow this convention,
        specify the argument shape_is_width_height=False, and your image will be displayed correctly
        rather than mirrored over the X/Y axis."""
        super().__init__(parent)
        if data is None:
            raise ValueError('The "data" argument supplied to Image.__init__(..) must not be None.')
        self.data_changed.connect(self.changed)
        self.objectNameChanged.connect(self._onObjectNameChanged)
        self.name_changed.connect(self.changed)
        self.update(data, mask, is_twelve_bit, specified_float_range, shape_is_width_height, mask_shape_is_width_height, name, _called_by_init=True)

    def _onObjectNameChanged(self):
        self.name_changed.emit(self)

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
        return '{}; {}, {}x{}, {}{} channel{} ({}){}>'.format(
            super().__repr__()[:-1],
            'with name "{}"'.format(name) if name else 'unnamed',
            self.size.width(),
            self.size.height(),
            '' if self._mask is None else 'masked, ',
            num_channels,
            '' if num_channels == 1 else 's',
            self.type,
            ' (per-channel binary)' if self.is_binary else '')

    def refresh(self, data_changed=False, mask_changed=False, is_twelve_bit_changed=False, float_range_override_changed=False):
        """The .refresh method should be called after modifying the contents of .data, .mask, and/or after replacing .is_twelve_bit or .stat_range
        by assignment, with True supplied for the respective _changed argument.  It is assumed that only those changes may have occurred, and that
        the shape, strides, and dtype of .data and/or .mask have not been changed (except by the .set method).

        The .refresh method is primarily useful to cause a user interface to update in response to data changes caused by manipulation of .data.data or
        another numpy view of the same memory.  The .set method is probably what you're looking for."""
        if not any((data_changed, mask_changed, is_twelve_bit_changed, stat_range_changed)):
            return
        if self.dtype is numpy.float32:
            if self.specified_float_range is None and data_changed:
                self._range = ndimage_statistics.min_max(self.data)
        if self.is_grayscale:
            if self.dtype is numpy.float32:
                self.stats_future = ndimage_statistics.compute_ranged_histogram(
                    self._data,
                    self._range,
                    256,
                    with_overflow_bins=False,
                    return_future=True,
                    make_ndimage_statistics_tuple=True)
            else:
                self.stats_future = ndimage_statistics.compute_ndimage_statistics(self._data, self.is_twelve_bit, return_future=True)
        else:
            if self.dtype is numpy.float32:

            self.stats_future = ndimage_statistics.compute_multichannel_ndimage_statistics(self._data, self.is_twelve_bit, return_future=True)
        self.data_changed.emit(self)



    def set(
            self,
            data=...,
            mask=...,
            is_twelve_bit=...,
            float_range_override=...,
            shape_is_width_height=True,
            mask_shape_is_width_height=True,
            name=...):
        """In addition to the values __init__ accepts for the data, mask, name, is_twelve_bit, and float_range_override arguments, set accepts ...
        (Ellipses) for these arguments to indicate No Change.  That is, the contents of i.data, i.name, i.mask, i.is_twelve_bit, and
        i.specified_float_range are left unchanged by i.set_data(..) if their corresponding arguments are Ellipses (as they are by default)."""
        if data is ...:

        data_changed = data is not ...
        data =
        mask_changed = mask is not ...
        is_twelve_bit_changed = is_twelve_bit is not ...
        float_range_override_changed = float_range_override is not ...

        if is_twelve_bit is not ... and data
        if data_changed:
            if data is None:
                raise ValueError('data argument must not be None')
            self._data = numpy.asarray(data)
        if mask_changed:
            if mask is None:
                self._mask = None
            else:
                mask = numpy.array(mask, dtype=numpy.bool, copy=False, order=
        self.is_twelve_bit = is_twelve_bit
        dt = self._data.dtype.type

        self.refresh(data_changed, mask_changed, is_twelve_bit_changed, specified_float_range_changed)



        if dt not in (numpy.bool8, numpy.uint8, numpy.uint16, numpy.float32):
            raise ValueError(
                'The "data" argument must produce a numpy ndarray of dtype bool8, uint8, uint16, or float32 when '
                'passed through numpy.asarray(data).  So, if data is, itself, an ndarray, then data.dtype must be '
                'one of bool8, uint8, uint16, or float32.')
        if specified_float_range is None:
            self.specified_float_range = None
        elif specified_float_range is not ...:
            specified_float_range = tuple(specified_float_range)
            if len(specified_float_range) != 2:
                raise ValueError(
                    'specified_float_range must either be {} or an iterable of two elements castable to float'.format(
                        'None' if _called_by_init else 'None, Ellipses, '))
            specified_float_range = float(specified_float_range[0]), float(specified_float_range[1])
            if not specified_float_range[0] < specified_float_range[1]:
                raise ValueError(
                    'If the value supplied for float_range is {}, float_range[0] must be < float_range[1].'.format(
                        'not None' if _called_by_init else 'neither None nor Ellipses'))
            self.specified_float_range = specified_float_range

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
                256,
                with_overflow_bins=False,
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

        if name is not ...:
            self.name = name
        if mask is not ...:
            self.set_mask(mask, mask_shape_is_width_height)
        self.refresh()

    set_data.__doc__ = __init__.__doc__ + '\n\n' + set_data.__doc__

    def set_mask(self, mask, shape_is_width_height=True):
        if self.dtype is numpy.float32:

        if mask is None:
            self._mask = None
        else:
            mask = numpy.asarray(mask, numpy.bool)
            if mask.ndim != 2:
                raise ValueError('mask argument must be None or a 2D iterable of elements castable to bool with the same dimensions as .data.')
            if not shape_is_width_height:
                mask = mask.transpose(1, 0)
                desired_strides = (1, mask.shape[0])
                if desired_strides != mask.strides:
                    d = mask
                    mask = numpy.ndarray(d.shape, strides=desired_strides, dtype=numpy.bool)
                    mask.flat = d.flat
            if not mask.shape == self._data.shape:
                raise ValueError('mask argument must be None or a 2D iterable of elements castable to bool with the same dimensions as .data.')
            self._mask = mask
        self.refresh(data_changed_signal=False, mask_changed_signal=True)

    def generate_contextual_info_for_pos(self, x, y, include_image_name=True):
        sz = self.size
        component_format_str = '{}' if self.dtype is numpy.float32 else '{}'
        if 0 <= x < sz.width() and 0 <= y < sz.height():
            type_ = self.type
            num_channels = self.num_channels
            mst = ''
            if include_image_name:
                name = self.name
                if name:
                    mst += '"' + name + '" '
            mst+= 'x:{} y:{} '.format(x, y)
            vt = '(' + ' '.join((c + ':' + component_format_str for c in self.type)) + ')'
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
    def mask(self):
        return self._mask

    @property
    def mask_T(self):
        if self._mask is not None:
            return self._mask.transpose(1, 0)

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
        outside of the interval represented by the value of the .range property (12-bit-per-channel intensity values are stored in
        16-bit unsigned integers, meaning that a nominally 12-bit-per-channel image may have a maximum value up to 65535.)"""
        return self.stats_future.result().min_max_intensity

    @property
    def range(self):
        """The range of valid values that may be assigned to any component of any image pixel.  For 8-bit-per-channel integer images,
        this is always [0,255], for 12-bit-per-channel integer images, [0,4095], for 16-bit-per-channel integer images, [0,65535].
        For floating point images, .range is the minimum and maximum values over all channels of all (non-masked) pixels."""
        return self._range