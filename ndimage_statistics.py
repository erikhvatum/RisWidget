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
    
    def compute_ndimage_statistics(array, twelve_bit=False, n_bins=1024, hist_max=None, hist_min=None, n_threads=8, return_future=False):
        array = numpy.asarray(array)
        extra_args = ()
        if array.dtype == numpy.uint8:
            hist_min_max = _ndimage_statistics.hist_min_max_uint8
            n_bins = 256
        elif array.dtype == numpy.uint16:
            n_bins = 1024
            if twelve_bit:
                hist_min_max = _ndimage_statistics.hist_min_max_uint12
            else:
                hist_min_max = _ndimage_statistics.hist_min_max_uint16
        elif array.dtype == numpy.float32:
            hist_min_max = _ndimage_statistics.hist_min_max_float32
            if hist_max is None:
                hist_max = array.max()
            if hist_min is None:
                hist_min = array.min()
            extra_args = (hist_min, hist_max)
        else:
            raise TypeError('array argument type must be uint8, uint16, or float32')

        slices = [array[i::n_threads] for i in range(n_threads)]
        histograms = numpy.empty((n_threads, n_bins), dtype=numpy.uint32)
        min_maxs = numpy.empty((n_threads, 2), dtype=array.dtype)
        futures = [pool.submit(hist_min_max, arr_slice, hist_slice, min_max, *extra_args) for
                   arr_slice, hist_slice, min_max in zip(slices, histograms, min_maxs)]

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

    def scompute_ndimage_statistics(array, twelve_bit=False, n_bins=1024, hist_max=None, hist_min=None, n_threads=8, return_future=False):
        array = numpy.asarray(array)
        extra_args = ()
        if array.dtype == numpy.uint8:
            hist_min_max = _ndimage_statistics.shist_min_max_uint8
            n_bins = 256
        elif array.dtype == numpy.uint16:
            n_bins = 1024
            if twelve_bit:
                hist_min_max = _ndimage_statistics.shist_min_max_uint12
            else:
                hist_min_max = _ndimage_statistics.shist_min_max_uint16
        elif array.dtype == numpy.float32:
            hist_min_max = _ndimage_statistics.hist_min_max_float32
            if hist_max is None:
                hist_max = array.max()
            if hist_min is None:
                hist_min = array.min()
            extra_args = (hist_min, hist_max)
        else:
            raise TypeError('array argument type must be uint8, uint16, or float32')

        slices = [array[i::n_threads] for i in range(n_threads)]
        histograms = numpy.empty((n_threads, n_bins), dtype=numpy.uint32)
        min_maxs = numpy.empty((n_threads, 2), dtype=array.dtype)
        futures = [pool.submit(hist_min_max, arr_slice, hist_slice, min_max, *extra_args) for
                   arr_slice, hist_slice, min_max in zip(slices, histograms, min_maxs)]

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
    import sys
    print('warning: Failed to load _ndimage_statistics binary module; using slow histogram and extrema computation methods.', file=sys.stderr)

    def compute_ndimage_statistics(array, twelve_bit=False, n_bins=1024, hist_max=None, hist_min=None, n_threads=None, return_future=False):
        if array.dtype == numpy.uint8:
            n_bins = 256
            histogram_range = (0, 255)
            image_range = (array.min(), array.max())
        elif array.dtype == numpy.uint16:
            n_bins = 1024
            histogram_range = (0, 4095 if twelve_bit else 65535)
            image_range = (array.min(), array.max())
        else:
            histogram_range = image_range = (
                array.min() if hist_min is None else hist_min,
                array.max() if hist_max is None else hist_max)

        def get_result():
            histogram = numpy.histogram(array, bins=n_bins, range=histogram_range, density=False)[0].astype(numpy.uint32)
            max_bin = histogram.argmax()

            return NDImageStatistics(histogram, max_bin, image_range)

        if return_future:
            return pool.submit(get_result)
        else:
            return get_result()

def compute_multichannel_ndimage_statistics(array, twelve_bit=False, n_bins=1024, hist_max=None, hist_min=None, n_threads=2, return_future=False):
    array = numpy.asarray(array)

    if return_future:
        def function():
            futures = [compute_ndimage_statistics(array[...,channel_idx], twelve_bit, n_bins, hist_max, hist_min, n_threads, True) for channel_idx in range(array.shape[2])]
            return NDImageStatistics(
                numpy.vstack((future.result().histogram for future in futures)),
                numpy.hstack((future.result().max_bin for future in futures)),
                numpy.vstack((future.result().min_max_intensity for future in futures)))
        return pool.submit(function)
    else:
        statses = [compute_ndimage_statistics(array[...,channel_idx], twelve_bit, n_bins, hist_max, hist_min, n_threads, False) for channel_idx in range(array.shape[2])]
        return NDImageStatistics(
            numpy.vstack((stats.histogram for stats in statses)),
            numpy.hstack((stats.max_bin for stats in statses)),
            numpy.vstack((stats.min_max_intensity for stats in statses)))
