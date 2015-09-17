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
# Authors: Zach Pincus, Erik Hvatum <ice.rikh@gmail.com>

import numpy
from collections import namedtuple
import concurrent.futures as futures

pool = futures.ThreadPoolExecutor(max_workers=16)

NDImageStatistics = namedtuple('NDImageStatistics', ('histogram', 'max_bin', 'min_max_intensity'))

try:
    from . import _ndimage_statistics

    def find_min_max(im, mask=None):
        im = numpy.asarray(im)
        if mask is not None:
            mask = numpy.asarray(mask, dtype=numpy.uint8)
            assert im.shape == mask.shape
        if im.dtype == numpy.float32:
            if im.ndim == 2:
                min_max = numpy.empty((2,), dtype=numpy.float32)
                _ndimage_statistics.min_max_float32(im, min_max) if mask is None else _ndimage_statistics.masked_min_max_float32(im, mask, min_max)
                return min_max
            if im.ndim == 3:
                min_max = numpy.empty((im.shape[2], 2), dtype=numpy.float32)
                if mask is None:
                    for c in range(im.shape[2]):
                        _ndimage_statistics.min_max_float32(im[...,c], min_max[c])
                else:
                    for c in range(im.shape[2]):
                        _ndimage_statistics.min_max_float32(im[...,c], mask, min_max[c])
                return min_max
            else:
                raise ValueError('im must be 2D or 3D iterable / ndarray.')
        else:
            raise TypeError('im argument type must be a numpy.ndarray with dtype float32')

    def compute_ranged_histogram(im, min_max, bin_count, with_overflow_bins=False, mask=None, return_future=False, make_ndimage_statistics_tuple=False):
        im = numpy.asarray(im)
        if mask is not None:
            mask = numpy.asarray(mask, dtype=numpy.uint8)
            assert im.shape == mask.shape
        if im.dtype == numpy.float32:
            if im.ndim == 2:
                def fn():
                    histogram = numpy.empty((bin_count,), dtype=numpy.uint32)
                    if mask is None:
                        _ndimage_statistics.ranged_hist_float32(im, min_max[0], min_max[1], bin_count, with_overflow_bins, histogram)
                    else:
                        _ndimage_statistics.masked_ranged_hist_float32(im, mask, min_max[0], min_max[1], bin_count, with_overflow_bins, histogram)
                    if make_ndimage_statistics_tuple:
                        return NDImageStatistics(histogram, histogram.argmax(), min_max)
                    return histogram
            elif im.ndim == 3:
                def fn():
                    histogram = numpy.empty((im.shape[2], bin_count), dtype=numpy.uint32)
                    if mask is None:
                        for c in range(im.shape[2]):
                            _ndimage_statistics.ranged_hist_float32(im[...,c], min_max[c,0], min_max[c,1], bin_count, with_overflow_bins, histogram[c])
                    else:
                        for c in range(im.shape[2]):
                            _ndimage_statistics.masked_ranged_hist_float32(im[...,c], mask, min_max[c,0], min_max[c,1], bin_count, with_overflow_bins, histogram[c])
                    if make_ndimage_statistics_tuple:
                        return NDImageStatistics(histogram, numpy.hstack(histogram[c].argmax() for c in range(im.shape[2])), min_max)
                    return histogram
            else:
                raise ValueError('im must be 2D or 3D iterable / ndarray.')
        else:
            raise TypeError('im argument type must be a numpy.ndarray with dtype float32')
        if return_future:
            return pool.submit(fn)
        else:
            return fn()
    
    def compute_ndimage_statistics(im, mask=None, twelve_bit=False, n_bins=1024, hist_max=None, hist_min=None, n_threads=1, return_future=False):
        im = numpy.asarray(im)
        extra_args = ()
        if mask is not None:
            mask = numpy.asarray(mask, dtype=numpy.uint8)
            assert im.shape == mask.shape
        if im.dtype == numpy.uint8:
            hist_min_max = _ndimage_statistics.hist_min_max_uint8 if mask is None else _ndimage_statistics.masked_hist_min_max_uint8
            n_bins = 256
        elif im.dtype == numpy.uint16:
            n_bins = 1024
            if twelve_bit:
                hist_min_max = _ndimage_statistics.hist_min_max_uint12 if mask is None else _ndimage_statistics.masked_hist_min_max_uint12
            else:
                hist_min_max = _ndimage_statistics.hist_min_max_uint16 if mask is None else _ndimage_statistics.masked_hist_min_max_uint16
        elif im.dtype == numpy.float32:
            hist_min_max = _ndimage_statistics.hist_min_max_float32 if mask is None else _ndimage_statistics.masked_hist_min_max_float32
            if hist_max is None:
                hist_max = im.max()
            if hist_min is None:
                hist_min = im.min()
            extra_args = (hist_min, hist_max)
        else:
            raise TypeError('im argument type must be uint8, uint16, or float32')

        slices = [im[i::n_threads] for i in range(n_threads)]
        histograms = numpy.empty((n_threads, n_bins), dtype=numpy.uint32)
        min_maxs = numpy.empty((n_threads, 2), dtype=im.dtype)
        if mask is None:
            futures = [pool.submit(hist_min_max, arr_slice, hist_slice, min_max, *extra_args) for
                       arr_slice, hist_slice, min_max in zip(slices, histograms, min_maxs)]
        else:
            mslices = [mask[i::n_threads] for i in range(n_threads)]
            futures = [pool.submit(hist_min_max, mslice, arr_slice, hist_slice, min_max, *extra_args) for
                       arr_slice, mslice, hist_slice, min_max in zip(slices, mslices, histograms, min_maxs)]

        def get_result():
            for future in futures:
                future.result()

            histogram = histograms.sum(axis=0, dtype=numpy.uint32)
            max_bin = histogram.argmax()

            return NDImageStatistics(histogram, max_bin, (min_maxs[:,0].min(), min_maxs[:,1].max()))

        if return_future:
            return pool.submit(get_result)
        else:
            return get_result()
except ImportError:
    import warnings
    warnings.warn('warning: Failed to load _ndimage_statistics binary module; using slow histogram and extrema computation methods.')

    def find_min_max(im, mask=None):
        im = numpy.asarray(im)
        if mask is not None:
            raise NotImplementedError()
        if im.ndim == 2:
            return numpy.array((im.min(), im.max()), dtype=im.dtype)
        elif im.ndim == 3:
            return numpy.array([(im[...,c].min(), im[...,c].max()) for c in range(im.shape[2])], dtype=im.dtype)
        else:
            raise ValueError('im must be 2D or 3D iterable / ndarray.')

    def compute_ranged_histogram(im, min_max, bin_count, with_overflow_bins=False, mask=None, return_future=False, make_ndimage_statistics_tuple=False):
        im = numpy.asarray(im)
        def fn():
            if im.ndim == 2:
                histogram = numpy.histogram(im, bins=bin_count, range=min_max, density=False, weights=mask)[0].astype(numpy.uint32)
                if make_ndimage_statistics_tuple:
                    return NDImageStatistics(histogram, histogram.argmax(), min_max)
                return histogram
            elif im.ndim == 3:
                histogram = numpy.vstack(numpy.histogram(im[c], bins=bin_count, range=min_max[c], density=False, weights=mask)[0].astype(numpy.uint32) for c in range(im.shape[2]))
                if make_ndimage_statistics_tuple:
                    return NDImageStatistics(histogram, numpy.hstack(histogram[c].argmax() for c in range(im.shape[2])), min_max)
                return histogram
            else:
                raise ValueError('im must be 2D or 3D iterable / ndarray.')
        if return_future:
            return pool.submit(fn)
        else:
            return fn()

    def compute_ndimage_statistics(im, mask=None, twelve_bit=False, n_bins=1024, hist_max=None, hist_min=None, n_threads=None, return_future=False):
        im = numpy.asarray(im)
        if im.dtype == numpy.uint8:
            n_bins = 256
            histogram_range = (0, 255)
            image_range = (im.min(), im.max())
        elif im.dtype == numpy.uint16:
            n_bins = 1024
            histogram_range = (0, 4095 if twelve_bit else 65535)
            image_range = (im.min(), im.max())
        else:
            histogram_range = image_range = (
                im.min() if hist_min is None else hist_min,
                im.max() if hist_max is None else hist_max)

        def get_result():
            histogram = numpy.histogram(im, bins=n_bins, range=histogram_range, density=False, weights=mask)[0].astype(numpy.uint32)
            max_bin = histogram.argmax()

            return NDImageStatistics(histogram, max_bin, image_range)

        if return_future:
            return pool.submit(get_result)
        else:
            return get_result()

def compute_multichannel_ndimage_statistics(im, mask=None, twelve_bit=False, n_bins=1024, hist_max=None, hist_min=None, n_threads=1, return_future=False):
    """Uses im.shape[2] * n_threads number of threads."""
    im = numpy.asarray(im)

    if return_future:
        def function():
            futures = [compute_ndimage_statistics(im[...,channel_idx], mask, twelve_bit, n_bins, hist_max, hist_min, n_threads, True) for channel_idx in range(im.shape[2])]
            return NDImageStatistics(
                numpy.vstack((future.result().histogram for future in futures)),
                numpy.hstack((future.result().max_bin for future in futures)),
                numpy.vstack((future.result().min_max_intensity for future in futures)))
        return pool.submit(function)
    else:
        statses = [compute_ndimage_statistics(im[...,channel_idx], mask, twelve_bit, n_bins, hist_max, hist_min, n_threads, False) for channel_idx in range(im.shape[2])]
        return NDImageStatistics(
            numpy.vstack((stats.histogram for stats in statses)),
            numpy.hstack((stats.max_bin for stats in statses)),
            numpy.vstack((stats.min_max_intensity for stats in statses)))
