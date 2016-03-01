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
        pixels with non-zero mask counterparts contribute to the histogram and extremae.  Mask pixels outside of im have no impact.  If mask is None, all
        image pixels are included.
        * Plain instance attributes computed by .refresh() and .set(..): .size, .is_grayscale, .num_channels, .has_alpha_channel, .is_binary,
        .is_twelve_bit.  Although nothing prevents assigning over these attributes, doing so is not advised.  The .data_changed signal is emitted
        to indicate that the value of any or all of these attributes has changed.
        * Properties with individual change signals: .name.  It is safe to assign None in addition anything else that str(..) accepts as its argument
        to .name.  When the value of .name is modified, .name_changed is emitted.

    Additionally, emission of .data_changed, .mask_changed, .imposed_float_range_changed, or .name_changed causes emission of .changed."""
    changed = Qt.pyqtSignal(object)
    data_changed = Qt.pyqtSignal(object)
    mask_changed = Qt.pyqtSignal(object)
    name_changed = Qt.pyqtSignal(object)
    imposed_float_range_changed = Qt.pyqtSignal(object)

    def __init__(
            self,
            data,
            mask=None,
            parent=None,
            is_twelve_bit=False,
            imposed_float_range=None,
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
        self.imposed_float_range_changed.connect(self.changed)
        self.update(data, mask, is_twelve_bit, imposed_float_range, shape_is_width_height, mask_shape_is_width_height, name, _called_by_init=True)

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

    def refresh(self, data_changed=False, mask_changed=False, is_twelve_bit_changed=False, imposed_float_range_changed=False):
        """The .refresh method should be called after modifying the contents of .data, .mask, and/or after replacing .is_twelve_bit or .imposed_float_range
        by assignment, with True supplied for the respective _changed argument.  It is assumed that only those changes may have occurred, and that
        the shape, strides, and dtype of .data and/or .mask have not been changed (except by the .set method).

        The .refresh method is primarily useful to cause a user interface to update in response to data changes caused by manipulation of .data.data or
        another numpy view of the same memory.  (You probably want to use the .set method in most cases.)"""
        if self.dtype is numpy.float32:
            if data_changed or mask_changed:
                extremae = ndimage_statistics.extremae(self._data, self._mask)
            else:
                extremae = self.extremae
            if self.imposed_float_range is None and data_changed:
                self._range = extremae if self.is_grayscale else numpy.array((extremae[:,0].min(), extremae[:,1].max()), dtype=numpy.float32)

        if self.is_grayscale:
            if self.dtype is numpy.float32:
                self.stats_future = ndimage_statistics.statistics(
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
        if data_changed or is_twelve_bit_changed:
            self.data_changed.emit(self)
        if mask_changed:
            self.mask_changed.emit(self)

    def set(self,
            data=...,
            mask=...,
            is_twelve_bit=...,
            imposed_float_range=...,
            data_shape_is_width_height=True,
            mask_shape_is_width_height=True,
            name=...):
        """In addition to the values __init__ accepts for the data, mask, name, is_twelve_bit, and imposed_float_range arguments, set accepts ...
        (Ellipses) for these arguments to indicate No Change.  That is, the contents of i.data, i.name, i.mask, i.is_twelve_bit, and
        i.specified_float_range are left unchanged by i.set_data(..) if their corresponding arguments are Ellipses (as they are by default)."""
        if data is ...:
            data_changed = False
            data = self._data
        else:
            if data is None:
                raise ValueError('The data argument must not be None.')
            data_changed = True

        if mask is ...:
            mask_changed = False
            mask = self._mask
        else:
            mask_changed = True

        if is_twelve_bit is ...:
            is_twelve_bit_changed = False
            is_twelve_bit = self.is_twelve_bit
        else:
            is_twelve_bit_changed = True

        imposed_float_range_changed = imposed_float_range is ...

        if not any((data_changed, mask_changed, is_twelve_bit_changed, imposed_float_range_changed)):
            return

        if data_changed:
            data = numpy.asarray(data)
            if not (data.ndim == 2 or (data.ndim == 3 and data.shape[2] in (2,3,4))):
                raise ValueError('data argument must be a 2D (grayscale) or 3D (grayscale with alpha, rgb, or rgba) iterable.')
            if data.dtype.type is numpy.float64:
                data = data.astype(numpy.float32)
            elif data.dtype.type not in (numpy.bool, numpy.uint8, numpy.uint16, numpy.float32):
                raise ValueError(
                    'If the data argument contains Python float or numpy float64 values, these are converted to float32.  In all other cases, the data '
                    'argument must produce a numpy ndarray of dtype bool8, uint8, uint16, or float32 when passed through numpy.asarray(data).  So, if the '
                    'data argument is an ndarray, then data.dtype must be one of bool8, uint8, uint16, float32, float64, or float.')
            if data.dtype.type is not numpy.uint16:
                if is_twelve_bit:
                    if is_twelve_bit_changed:
                        ValueError('The is_twelve_bit argument may only be True if data is of dtype numpy.uint16.')
                    else:
                        # Do not require specification of is_twelve_bit=False when .is_twelve_bit is True and .data is replaced with a non-uint16 array;
                        # instead, automatically set .is_twelve_bit to False.
                        is_twelve_bit = False
                        is_twelve_bit_changed = True
            if data_shape_is_width_height:
                data = data.transpose(*(1,0,2)[:data.ndim])
            bpe = data.itemsize
            desired_strides = (bpe, data.shape[0]*bpe) if data.ndim == 2 else (data.shape[2]*bpe, data.shape[0]*data.shape[2]*bpe, bpe)
            if desired_strides != data.strides:
                _data = data
                data = numpy.ndarray(data.shape, strides=desired_strides, dtype=data.dtype.type)
                data.flat = _data.flat
        elif is_twelve_bit_changed:
            if data.dtype.type is not numpy.uint16:
                # Explicitly supplying True for is_twelve_bit for non-uint16 data is, however, not allowed
                raise ValueError('The is_twelve_bit argument may only be True if data is of dtype numpy.uint16.')

        if imposed_float_range_changed:
            if imposed_float_range is not None:
                imposed_float_range = numpy.asarray(imposed_float_range, dtype=numpy.float32)
                if imposed_float_range.ndim != 1 or imposed_float_range.shape[0] != 2:
                    raise ValueError(
                        'The imposed_float_range argument must either be None, Ellipses, or must, when passed through '
                        'numpy.asrray(imposed_float_range, dtype=numpy.float32), yield a one dimensional, two element array.')
                if imposed_float_range[0] > imposed_float_range[1]:
                    raise ValueError(
                        'If the imposed_float_range argument is specified and is neither None nor Ellipses, the second element of float_range must '
                        'be less than or equal to the first.')

        if mask_changed:
            if mask is not None:
                mask = numpy.asarray(data)
                if mask.ndim != 2:
                    raise ValueError('mask argument must be None or a 2D iterable.')
                if mask.dtype.type not in (numpy.bool8, numpy.uint8):
                    mask = mask.astype(numpy.bool8)
                if mask_shape_is_width_height:
                    mask = mask.transpose(1, 0)
                desired_strides = 1, mask.shape[0]
                if desired_strides != mask.strides:
                    _mask = mask
                    mask = numpy.ndarray(mask.shape, strides=desired_strides, dtype=mask.dtype)
                    mask.flat = _mask.flat

        if data_changed or mask_changed:
            if mask is not None and (mask.shape[0] < data.shape[0] or mask.shape[1] < data.shape[1]):
                raise ValueError("mask width and height must be at least as large as data's.")

        if data_changed:
            self._data = data
            if data.ndim == 1:
                self.type = 'G'
                self.is_grayscale = True
            else:
                self.type = {2: 'Ga', 3: 'rgb', 4: 'rgba'}[self._data.shape[2]]
                self.is_grayscale = False
            if self.dtype is numpy.float32:
                pass # ._range is updated by .refresh() for float32 images as ._range may depend on .data
            elif self.dtype is numpy.uint8:
                self._range = numpy.array((0,255), dtype=numpy.uint8)
            elif self.dtype is numpy.uint16:
                self._range = numpy.array((0,65535), dtype=numpy.uint16)
            else:
                raise NotImplementedError('Add an elif statement above here to set ._range for your data type.')
        if mask_changed:
            self._mask = mask
        if is_twelve_bit_changed:
            self.is_twelve_bit = is_twelve_bit
        if imposed_float_range_changed:
            self.imposed_float_range = imposed_float_range

        self.refresh(data_changed, mask_changed, is_twelve_bit_changed, imposed_float_range_changed)

    set.__doc__ = __init__.__doc__ + '\n\n' + set.__doc__

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
        For floating point images, this is min/max values for all components of all pixels, unless specified with the imposed_float_range
        argument to the .__init__ or .set methods."""
        return self._range