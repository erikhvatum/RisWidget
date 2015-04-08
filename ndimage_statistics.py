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

NDImageStatistics = namedtuple('NDImageStatistics', ('histogram', 'max_bin', 'min_intensity', 'max_intensity'))

try:
    from . import _ndimage_statistics
    import concurrent.futures as futures

    pool = futures.ThreadPoolExecutor(max_workers=16)

    def compute_ndimage_statistics(array, twelve_bit=False, n_bins=1024, hist_max=None, hist_min=None, n_threads=8):
        array = numpy.asarray(array)
        extra_args = ()
        if array.dtype == numpy.uint8:
            function = _ndimage_statistics.hist_min_max_uint8
            n_bins = 256
        elif array.dtype == numpy.uint16:
            n_bins = 1024
            if twelve_bit:
                function = _ndimage_statistics.hist_min_max_uint12
            else:
                function = _ndimage_statistics.hist_min_max_uint16
        elif array.dtype == numpy.float32:
            function = _ndimage_statistics.hist_min_max_float32
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
        futures = [pool.submit(function, arr_slice, hist_slice, min_max, *extra_args) for
                   arr_slice, hist_slice, min_max in zip(slices, histograms, min_maxs)]
        for future in futures:
            future.result()

        histogram = histograms.sum(axis=0, dtype=numpy.uint32)
        max_bin = histogram.argmax()

        return NDImageStatistics(histogram, max_bin, min_maxs[:,0].min(), min_maxs[:,1].max())

except ImportError:
    import sys
    print('warning: Failed to load _ndimage_statistics binary module; using slow histogram and extrema computation methods.', file=sys.stderr)

    def compute_ndimage_statistics(array, twelve_bit=False, n_bins=1024, hist_max=None, hist_min=None, n_threads=None):
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

        histogram = numpy.histogram(array, bins=n_bins, range=histogram_range, density=False)[0].astype(numpy.uint32)
        max_bin = histogram.argmax()

        return NDImageStatistics(histogram, max_bin, image_range[0], image_range[1])
