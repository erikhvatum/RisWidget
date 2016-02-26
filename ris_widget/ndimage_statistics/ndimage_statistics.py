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
# Authors: Erik Hvatum <ice.rikh@gmail.com>, Zach Pincus

import numpy
import numpy.ma
from collections import namedtuple
import concurrent.futures as futures
import multiprocessing

pool = futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() + 1)

NDImageStatistics = namedtuple('NDImageStatistics', ('histogram', 'max_bin', 'min_max_intensity'))

try:
    from . import _ndimage_statistics

    def _min_max(im, mask=None):
        min_max = numpy.zeros((2,), dtype=numpy.float32)
        if mask is None:
            _ndimage_statistics.min_max_float32(im, min_max)
        else:
            _ndimage_statistics.masked_min_max_float32(im, mask, min_max)
        return min_max

    def _histogram(im, bin_count, range_, mask=None, with_overflow_bins=False):
        hist = numpy.zeros((bin_count,), dtype=numpy.uint32)
        if mask is None:
            _ndimage_statistics.ranged_hist_float32(im, range_[0], range_[1], bin_count, with_overflow_bins, hist)
        else:
            _ndimage_statistics.masked_ranged_hist_float32(im, mask, range_[0], range_[1], bin_count, with_overflow_bins, hist)
        return hist

    def _statistics(im, twelve_bit, mask=None):
        if im.dtype.type is numpy.uint8:
            hist = numpy.zeros((256,), dtype=numpy.uint32)
            min_max = numpy.zeros((2,), dtype=numpy.uint8)
            if mask is None:
                _ndimage_statistics.hist_min_max_uint8(im, hist, min_max)
            else:
                _ndimage_statistics.masked_hist_min_max_uint8(im, mask, hist, min_max)
        elif im.dtype.type is numpy.uint16:
            hist = numpy.zeros((1024,), dtype=numpy.uint32)
            min_max = numpy.zeros((2,), dtype=numpy.uint16)
            if mask is None:
                if twelve_bit:
                    _ndimage_statistics.hist_min_max_uint12(im, hist, min_max)
                else:
                    _ndimage_statistics.hist_min_max_uint16(im, hist, min_max)
            else:
                if twelve_bit:
                    _ndimage_statistics.masked_hist_min_max_uint12(im, mask, hist, min_max)
                else:
                    _ndimage_statistics.masked_hist_min_max_uint16(im, mask, hist, min_max)
        return NDImageStatistics(hist, hist.argmax(), min_max)

except ImportError:
    import warnings
    warnings.warn('warning: Failed to load _ndimage_statistics binary module; using slow histogram and extrema computation methods.')

    def _min_max(im, mask=None):
        if mask is not None:
            im = numpy.ma.array(im, dtype=im.dtype, copy=False, mask=~mask)
        return numpy.array((im.min(), im.max()), dtype=im.dtype)

    def _histogram(im, bin_count, range_, mask=None, with_overflow_bins=False):
        if with_overflow_bins:
            assert bin_count >= 3
            hist = numpy.zeros((bin_count,), dtype=numpy.uint32)
            hist[1:-1] = numpy.histogram(im, bins=bin_count-2, range=range_, density=False, weights=mask)[0].astype(numpy.uint32)
            if mask is not None:
                im = numpy.ma.array(im, dtype=im.dtype, copy=False, mask=~mask)
            hist[0] = (im < range_[0]).sum()
            hist[-1] = (im > range_[1]).sum()
            return hist
        else:
            assert bin_count >= 1
            return numpy.histogram(im, bins=bin_count, range=range_, density=False, weights=mask)[0].astype(numpy.uint32)

    def _statistics(im, twelve_bit, mask=None):
        if im.dtype.type is numpy.uint8:
            min_max = numpy.zeros((2,), dtype=numpy.uint8)
            bin_count = 256
            range_ = (0,255)
        elif im.dtype.type is numpy.uint16:
            min_max = numpy.zeros((2,), dtype=numpy.uint16)
            bin_count = 1024
            range_ = (0,4095) if twelve_bit else (0,65535)
        else:
            raise NotImplementedError('Support for dtype of supplied im argument not implemented.')
        hist = numpy.histogram(im, bins=bin_count, range=range, density=False, weights=mask)[0].astype(numpy.uint32)
        if mask is not None:
            im = numpy.ma.array(im, dtype=im.dtype, copy=False, mask=~mask)
        min_max[0] = im.min()
        min_max[1] = im.max()
        return NDImageStatistics(hist, hist.argmax(), min_max)

def min_max(im, mask=None, return_future=False):
    """Supports only float32 as that's the only image dtype for which we ever do min/max and histogram in two separate steps, for the simple reason that
    it is sometimes necessary to find the range over which the histogram is to be calculated for float32 images, but never in the case of uint8, 12, or
    16 images, whose range is that of their data type.  (uint12 being a somewhat special case; a flag indicates that although the stored as uint16, the
    highest 4 bits of each element remain 0.)

    im: The 2D or 3D float32 ndarray for which min and max values are found.
    mask: None or a 2D bool or uint8 ndarray with neither dimension smaller than the corresponding dimension of im.  If mask is not None, only image
    pixels with non-zero mask counterparts contribute to min_max.  Pixels of mask outside of im have no impact.  If mask is None, all image pixels are
    included.
    return_future: If not False, a concurrent.futures.Future is returned.

    Returns a channel_count x 2 float32 numpy array containing the min and max element values over the masked region or entirety of im."""
    assert im.dtype.type is numpy.float32
    assert im.ndim in (2,3)
    if mask is not None:
        assert mask.ndim == 2
        assert im.shape[0] <= mask.shape[0]
        assert im.shape[1] <= mask.shape[1]
        mask = mask[:im.shape[0], :im.shape[1]]
    if im.ndim == 2:
        def proc():
            return _min_max(im, mask)
    else:
        def axis_proc(axis):
            return _min_max(im[..., axis], mask)
        def proc():
            futes = [pool.submit(axis_proc, axis) for axis in range(im.shape[2])]
            return numpy.vstack(fute.result() for fute in futes)
    return pool.submit(proc) if return_future else proc()

def histogram(im, bin_count, range_, mask=None, with_overflow_bins=False, return_future=False):
    """Supports only float32 as that's the only image dtype for which we ever do min/max and histogram in two separate steps, for the simple reason that
    it is sometimes necessary to find the range over which the histogram is to be calculated for float32 images, but never in the case of uint8, 12, or
    16 images, whose range is that of their data type.  (uint12 being a somewhat special case; a flag indicates that although the stored as uint16, the
    highest 4 bits of each element remain 0.)

    im: The 2D or 3D float32 ndarray for which histogram is computed.
    range_: An indexable sequence of at least two elements, castable to float32, representing the closed interval which is divided into bin_count number
    of bins comprising the histogram.
    mask: None or a 2D bool or uint8 ndarray with neither dimension smaller than the corresponding dimension of im.  If mask is not None, only image
    pixels with non-zero mask counterparts contribute to the histogram.  Mask pixels outside of im have no impact.  If mask is None, all image pixels
    are included.
    with_overfloat_bins: If true, the first and last histogram bins represent the number of image pixels falling below and above range_, respectively.
    return_future: If not False, a concurrent.futures.Future is returned.

    Returns a channel_count x bin_count numpy array uint32 values."""
    assert im.dtype.type is numpy.float32
    assert im.ndim in (2,3)
    assert range_[0] < range_[1]
    if mask is not None:
        assert mask.ndim == 2
        assert im.shape[0] <= mask.shape[0]
        assert im.shape[1] <= mask.shape[1]
        mask = mask[:im.shape[0], :im.shape[1]]
    if with_overflow_bins:
        assert bin_count >= 4
    else:
        assert bin_count >= 2
    if im.ndim == 2:
        def proc():
            return _histogram(im, bin_count, range_, mask, with_overflow_bins)
    else:
        def axis_proc(axis):
            return _histogram(im[..., axis], bin_count, range_, mask, with_overflow_bins)
        def proc():
            futes = [pool.submit(axis_proc, axis) for axis in range(im.shape[2])]
            return numpy.vstack(fute.result() for fute in futes)
    return pool.submit(proc) if return_future else proc()

def statistics(im, twelve_bit=False, mask=None, return_future=False):
    assert im.ndim in (2,3)
    if im.dtype.type is numpy.uint8:
        assert twelve_bit == False
    elif im.dtype.type is numpy.uint16:
        pass
    else:
        raise NotImplementedError('Support for dtype of supplied im argument not implemented.')
    if mask is not None:
        assert mask.ndim == 2
        assert im.shape[0] <= mask.shape[0]
        assert im.shape[1] <= mask.shape[1]
        mask = mask[:im.shape[0], :im.shape[1]]
    if im.ndim == 2:
        def proc():
            return _statistics(im, twelve_bit, mask)
    else:
        def axis_proc(axis):
            return _statistics(im[..., axis], twelve_bit, mask)
        def proc():
            futes = [pool.submit(axis_proc, axis) for axis in range(im.shape[2])]
            return NDImageStatistics(
                numpy.vstack((fute.result().histogram for fute in futes)),
                numpy.hstack((fute.result().max_bin for fute in futes)),
                numpy.vstack((fute.result().min_max_intensity for fute in futes)))
    return pool.submit(proc) if return_future else proc()