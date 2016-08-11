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
import OpenGL
import OpenGL.GL as PyGL
from PyQt5 import Qt
import textwrap
from .async_texture import AsyncTexture
from .ndimage_statistics import ndimage_statistics
from . import om

class Image(Qt.QObject):
    """"""
    name_changed = Qt.pyqtSignal(object)
    dtype_changed = Qt.pyqtSignal(object)
    type_changed = Qt.pyqtSignal(object)
    num_channels_changed = Qt.pyqtSignal(object)
    size_changed = Qt.pyqtSignal(object)
    extrema_changed = Qt.pyqtSignal(object)
    histogram_changed = Qt.pyqtSignal(object)

    IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT = {
        'G'   : Qt.QOpenGLTexture.R32F,
        'Ga'  : Qt.QOpenGLTexture.RG32F,
        'rgb' : Qt.QOpenGLTexture.RGB32F,
        'rgba': Qt.QOpenGLTexture.RGBA32F
    }
    NUMPY_DTYPE_TO_GL_PIXEL_TYPE = {
        bool         : PyGL.GL_UNSIGNED_BYTE,
        numpy.uint8  : PyGL.GL_UNSIGNED_BYTE,
        numpy.uint16 : PyGL.GL_UNSIGNED_SHORT,
        numpy.float32: PyGL.GL_FLOAT
    }
    IMAGE_TYPE_TO_GL_PIX_FORMAT = {
        'G'   : PyGL.GL_RED,
        'Ga'  : PyGL.GL_RG,
        'rgb' : PyGL.GL_RGB,
        'rgba': PyGL.GL_RGBA
    }
    NUMPY_INTEGER_DTYPE_TO_DEFAULT_DRANGE = {
        bool         : numpy.array((0, 1), dtype=bool),
        numpy.uint8  : numpy.array((0, 255), dtype=numpy.uint8),
        numpy.uint16 : numpy.array((0, 65535), dtype=numpy.uint16)
    }
    for a in NUMPY_INTEGER_DTYPE_TO_DEFAULT_DRANGE.values():
        a.flags.writeable = False
    del a

    properties = []

    name = property(
        Qt.QObject.objectName,
        lambda self, name: self.setObjectName('' if name is None else name),
        doc='Property proxy for QObject::objectName Qt property, which is directly accessible via the objectName getter and '
            'setObjectName setter.  Upon change, objectNameChanged is emitted.'
    )

    def _take_data(self, v):
        if v is None:
            return
        v = numpy.asarray(v)
        if not (v.ndim == 2 or (v.ndim == 3 and v.shape[2] in (2, 3, 4))):
            raise ValueError('The value supplied for data must be a 2D (grayscale) or 3D (grayscale with alpha, rgb, or rgba) iterable or None.')
        if v.dtype not in (bool, numpy.uint8, numpy.uint16, numpy.float32):
            if numpy.issubdtype(v.dtype, numpy.floating) or numpy.issubdtype(v.dtype, numpy.integer):
                v = v.astype(numpy.float32)
            else:
                raise ValueError('Image data must be integer or floating-point.')
        bpe = v.itemsize
        desired_strides = (bpe, v.shape[0] * bpe) if v.ndim == 2 else (v.shape[2] * bpe, v.shape[0] * v.shape[2] * bpe, bpe)
        if desired_strides != v.strides:
            v_ = v
            v = numpy.ndarray(v.shape, strides=desired_strides, dtype=v.dtype)
            v.flat = v_.flat
        return v

    def _post_set_data(self, v):
        if v is None:
            dtype = None
            self._stats_future = None
            size = None
            type_ = None
            num_channels = None
            is_grayscale = False
        else:
            if v.ndim == 2:
                type_ = 'G'
                is_grayscale = True
                num_channels = 1
            else:
                type_ = {2: 'Ga', 3: 'rgb', 4: 'rgba'}[v.shape[2]]
                is_grayscale = False
                num_channels = self._data.shape[2]
            self.size = Qt.QSize(*self._data.shape[:2])
            self.has_alpha_channel = self.type in ('Ga', 'rgba')
            dtype = v.dtype
            self._stats_future =
            size = Qt.QSize(v.shape[0], v.shape[1])
        if dtype != self._dtype:
            self._dtype = dtype
            self.dtype_changed.emit(self)
        if size != self._size:
            self._size = size
            self.size_changed.emit(self)


    data = om.Property(
        properties=properties,
        name='data',
        default_value_callback=lambda image: None,
        take_arg_callback=_take_data,
        post_set_callback=_post_set_data
    )

    @property
    def valid_specials(self):
        return (None, '12bit') if self.dtype == numpy.uint16 else (None,)

    @property
    def dtype(self):
        return self._dtype

    def __init__(
            self,
            data=None,
            drange=None,
            special=None,
            mask=None,
            parent=None,
            name=None,
            immediate_texture_upload=True,
            use_open_mp=False):
        """
        The shape of image and mask data is interpreted as (x,y) for 2-d arrays and (x,y,c) for 3-d arrays.  If your image or mask was loaded as (y,x),
        array.T will produce an (x,y)-shaped array.  In case of (y,x,c) image data, array.swapaxes(0,1) is required."""
        super().__init__(parent)
        for property in self.properties:
            property.instantiate(self)
        self._type = 'EMPTY'
        self._dtype = None
        self._stats_future = None
        self._size = None
        self._num_channels = None
        self.set(
            data=data,
            drange=drange,
            special=special,
            mask=mask,
            name=name,
            immediate_texture_upload=immediate_texture_upload,
            use_open_mp=use_open_mp,
            _emit_change_signals=False
        )
        self.setObjectName(name)
        self.objectNameChanged.connect(lambda v: self.name_changed.emit(self))

    @classmethod
    def from_qimage(
            cls,
            qimage,
            drange=None,
            special=None,
            mask=None,
            parent=None,
            name=None,
            immediate_texture_upload=True,
            use_open_mp=False):
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
            return cls(
                data=npyimage.copy(),
                drange=drange,
                special=special,
                mask=mask,
                parent=parent,
                name=name,
                immediate_texture_upload=immediate_texture_upload,
                use_open_mp=use_open_mp)

    def __repr__(self):
        name = self.getObjectName()
        name = 'with name "{}"'.format(name) if name else 'unnamed'
        if self._data is None:
            return '{}; {}, ({}){}>'.format(
                super().__repr__()[:-1],
                name,
                self._type,
                '' if self._mask is None else ', masked'
            )
        num_channels = self._num_channels
        return '{}; {}, {}x{}, {}{} channel{} ({}), drange {}-{}>'.format(
            super().__repr__()[:-1],
            name,
            self._size.width(),
            self._size.height(),
            '' if self._mask is None else 'masked, ',
            num_channels,
            '' if num_channels == 1 else 's',
            self._type,
            self._drange[0],
            self.range[1]
        )

    def refresh(
            self,
            data_changed=True,
            mask_changed=False,
            is_twelve_bit_changed=False,
            imposed_float_range_changed=False,
            immediate_texture_upload=True,
            use_open_mp=False):
        """
        The .refresh method should be called after modifying the contents of .data, .mask, and/or after replacing .is_twelve_bit or .imposed_float_range
        by assignment, with True supplied for the respective _changed argument.  It is assumed that only those changes may have occurred, and that
        the shape, strides, and dtype of .data and/or .mask have not been changed (except by the .set method).

        The .refresh method is primarily useful to cause a user interface to update in response to data changes caused by manipulation of .data.data or
        another numpy view of the same memory.  (You probably want to use the .set method in most cases.)"""
        if self.dtype == numpy.float32:
            if data_changed:
                self.async_texture = AsyncTexture(
                    self._data,
                    Image.IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT[self.type],
                    Image.IMAGE_TYPE_TO_GL_PIX_FORMAT[self.type],
                    Image.NUMPY_DTYPE_TO_GL_PIXEL_TYPE[self.dtype.type],
                    immediate_texture_upload,
                    self.name)
            if data_changed or mask_changed:
                extremae = ndimage_statistics.extremae(self._data, self._mask)
            else:
                extremae = self.extremae
            if self.imposed_float_range is None:
                self._range = extremae if self.is_grayscale else numpy.array((extremae[:,0].min(), extremae[:,1].max()), dtype=numpy.float32)
            else:
                self._range = self.imposed_float_range
            if data_changed or mask_changed or imposed_float_range_changed:
                histogram = ndimage_statistics.histogram(self._data, 1024, self._range, self._mask)
            else:
                histogram = self.histogram
            self.stats_future = ndimage_statistics.bundle_float_stats_into_future(histogram, extremae)
        else:
            if data_changed or mask_changed or is_twelve_bit_changed:
                data = self._data
                if data_changed:
                    t = self.type
                    self.async_texture = AsyncTexture(
                        data,
                        Image.IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT[t],
                        Image.IMAGE_TYPE_TO_GL_PIX_FORMAT[t],
                        Image.NUMPY_DTYPE_TO_GL_PIXEL_TYPE[self.dtype.type],
                        immediate_texture_upload,
                        self.name)
                self.stats_future = ndimage_statistics.statistics(
                    data.astype(numpy.uint8) if self.dtype == bool else data, # TODO: fix ndimage_statistics so that bool => uint8 conversion is not required
                    self.is_twelve_bit,
                    self.mask,
                    return_future=True,
                    use_open_mp=use_open_mp)
        if data_changed or mask_changed or is_twelve_bit_changed or imposed_float_range_changed:
            self.data_changed.emit(self)
        if mask_changed:
            self.mask_changed.emit(self)

    def set(self,
            data=...,
            drange=...,
            special=...,
            mask=...,
            name=...,
            immediate_texture_upload=True,
            use_open_mp=False,
            _emit_change_signals=True):
        """
        In addition to the values __init__ accepts for the data, drange, special, mask, and name, set accepts ... (Ellipses) for these
        arguments to indicate No Change.  That is, the contents of i.data, i.drange, i.special, i.mask, and i.name are left unchanged
        by i.set(..) if their corresponding arguments are Ellipses (as they are by default), except in the case where i.drange or
        i.special are modified as a consequence of supplying non-ellipses for data or special."""
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

        imposed_float_range_changed = imposed_float_range is not ...
        name_changed = name is not ...

        if not any((data_changed, mask_changed, is_twelve_bit_changed, imposed_float_range_changed, name_changed)):
            return

        if data_changed:
            data = numpy.asarray(data)
            if not (data.ndim == 2 or (data.ndim == 3 and data.shape[2] in (2,3,4))):
                raise ValueError('data argument must be a 2D (grayscale) or 3D (grayscale with alpha, rgb, or rgba) iterable.')
            if data.dtype not in (bool, numpy.uint8, numpy.uint16, numpy.float32):
                if numpy.issubdtype(data.dtype, numpy.floating) or numpy.issubdtype(data.dtype, numpy.integer):
                    data = data.astype(numpy.float32)
                else:
                    raise ValueError('Image data must be integer or floating-point.')
            if data.dtype != numpy.uint16:
                if is_twelve_bit:
                    if is_twelve_bit_changed:
                        ValueError('The is_twelve_bit argument may only be True if data is of type uint16.')
                    else:
                        # Do not require specification of is_twelve_bit=False when .is_twelve_bit is True and .data is replaced with a non-uint16 array;
                        # instead, automatically set .is_twelve_bit to False.
                        is_twelve_bit = False
                        is_twelve_bit_changed = True
            bpe = data.itemsize
            desired_strides = (bpe, data.shape[0]*bpe) if data.ndim == 2 else (data.shape[2]*bpe, data.shape[0]*data.shape[2]*bpe, bpe)
            if desired_strides != data.strides:
                _data = data
                data = numpy.ndarray(data.shape, strides=desired_strides, dtype=data.dtype)
                data.flat = _data.flat
        elif is_twelve_bit_changed:
            if data.dtype != numpy.uint16:
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
                mask = numpy.asarray(mask)
                if mask.ndim != 2:
                    raise ValueError('mask argument must be None or a 2D iterable.')
                if mask.dtype != bool:
                    mask = mask.astype(bool)
                desired_strides = 1, mask.shape[0]
                if desired_strides != mask.strides:
                    _mask = mask
                    mask = numpy.ndarray(mask.shape, strides=desired_strides, dtype=mask.dtype)
                    mask.flat = _mask.flat

        if data_changed:
            self._data = data
            if data.ndim == 2:
                self.type = 'G'
                self.is_grayscale = True
                self.num_channels = 1
            else:
                self.type = {2: 'Ga', 3: 'rgb', 4: 'rgba'}[self._data.shape[2]]
                self.is_grayscale = False
                self.num_channels = self._data.shape[2]
            self.size = Qt.QSize(*self._data.shape[:2])
            self.has_alpha_channel = self.type in ('Ga', 'rgba')
            if self.dtype == numpy.float32:
                pass # ._range is updated by .refresh() for float32 images as ._range may depend on .data
            elif self.dtype == bool:
                self._range = numpy.array((False, True), dtype=bool)
            elif self.dtype == numpy.uint8:
                self._range = numpy.array((0,255), dtype=numpy.uint8)
            elif self.dtype == numpy.uint16:
                self._range = numpy.array((0,65535), dtype=numpy.uint16)
            else:
                raise NotImplementedError('Add an elif statement above here to set ._range for your data type.')
        if mask_changed:
            self._mask = mask
        if is_twelve_bit_changed:
            self.is_twelve_bit = is_twelve_bit
        if imposed_float_range_changed:
            self.imposed_float_range = imposed_float_range
        if name_changed:
            self.name = name

        self.refresh(data_changed, mask_changed, is_twelve_bit_changed, imposed_float_range_changed, immediate_texture_upload, use_open_mp)

    set.__doc__ = textwrap.dedent(set.__doc__) + '\n' + textwrap.dedent(__init__.__doc__)

    def generate_contextual_info_for_pos(self, x, y, include_image_name=True):
        sz = self.size
        component_format_str = '{}' if self.dtype == numpy.float32 else '{}'
        if 0 <= x < sz.width() and 0 <= y < sz.height():
            type_ = self.type
            num_channels = self.num_channels
            mst = ''
            if include_image_name:
                name = self.name
                if name:
                    mst += '"' + name + '" '
            mst+= 'x:{} y:{} '.format(x, y)
            mask = self._mask
            if mask is None:
                masked = ''
            else:
                mx = int(x * mask.shape[0] / self._data.shape[0])
                my = int(y * mask.shape[1] / self._data.shape[1])
                masked = 'MASKED ' if mask[mx,my] == 0 else ''
            vt = '(' + masked + ' '.join((c + ':' + component_format_str for c in self.type)) + ')'
            if num_channels == 1:
                vt = vt.format(self.data[x, y])
            else:
                vt = vt.format(*self.data[x, y])
            return mst+vt



    @property
    def data(self):
        """Image data as numpy array in shape = (width, height, [channels]) convention."""
        return self._data

    @data.setter
    def data(self, v):
        self.set(data=v)

    @property
    def type(self):
        return self._type

    @property
    def dtype(self):
        return None if self._data is None else self._data.dtype

    @property
    def valid_specials(self):
        return (None, '12bit') if self.dtype == numpy.uint16 else (None,)

    @property
    def special(self):
        return self._special

    @special.setter
    def special(self, v):
        self.set(special=v)

    @property
    def drange(self):
        return self._drange

    @drange.setter
    def drange(self, v):
        self.set(drange=v)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, v):
        self.set(mask=v)

    @property
    def histogram(self):
        histogram = self.stats_future.result().histogram
        if self.dtype == bool:
            return histogram[:2]
        return histogram

    @property
    def channel_histograms(self):
        channel_histograms = self.stats_future.result().channel_histograms
        if self.dtype == bool:
            return [channel_histogram[:2] for channel_histogram in channel_histograms]
        return channel_histograms

    @property
    def max_histogram_bin(self):
        return self.stats_future.result().max_bin

    @property
    def extrema(self):
        """The actual minimum and maximum intensity values across all channels, excluding alpha.  drange is not enforced, so it is
        possible for min and/or max to fall outside of the interval represented by the value of the drange property."""
        if self.data is None:
            return
        return self.stats_future.result.extrema

    @property
    def channel_extrema(self):
        """The actual per-channel minimum and maximum intensity values.  drange is not enforced, so it is possible for min and/or
        max to fall outside of the interval represented by the value of the drange property."""
        if self.data is None:
            return
        if self.is_grayscale:
            return self.extrema
        return self.stats_future.channel_extrema