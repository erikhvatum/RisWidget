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
# Authors: Erik Hvatum <ice.rikh@gmail.com>, Zach Pincus

from ._common import *

# In general, we try to avoid page write contention between threads by allocating write targets
# (such as min/max and histogram arrays) in the worker threads.  Allocating all thread write targets
# as a single numpy array into which individual threads are given views may seem tempting but
# would cause massive contention, and it is therefore avoided.

try:
    from ._measures_fast import _min_max, _histogram, _statistics
    USING_FAST_MEASURES = True
except ImportError:
    import warnings
    warnings.warn('warning: Failed to load _ndimage_statistics binary module; using slow histogram and extrema computation methods.')
    from ._measures_slow import _min_max, _histogram, _statistics
    USING_FAST_MEASURES = False

def extremae(im, mask=None, per_channel_thread_count=2, return_future=False):
    """im: The 2D or 3D ndarray for which min and max values are found.
    mask: None or a 2D bool or uint8 ndarray with neither dimension smaller than the corresponding dimension of im.  If mask is not None, only image
    pixels with non-zero mask counterparts contribute to min_max.  Pixels of mask outside of im have no impact.  If mask is None, all image pixels are
    included.
    return_future: If not False, a concurrent.futures.Future is returned.

    Returns a channel_count x 2 numpy array containing the min and max element values over the masked region or entirety of im."""
    assert im.ndim in (2, 3)
    if mask is not None:
        assert mask.ndim == 2
        assert mask.dtype in (numpy.uint8, numpy.bool)
    if im.ndim == 2:
        def proc():
            return _min_max(im, mask)
    else:
        def channel_proc(channel):
            return _min_max(im[..., channel], mask)
        def proc():
            futes = [pool.submit(channel_proc, channel) for channel in range(im.shape[2])]
            ret = numpy.vstack(fute.result() for fute in futes)
            if ret.dtype != im.dtype:
                ret = ret.astype(im.dtype)
            return ret
    return pool.submit(proc) if return_future else proc()

# In the unlikely scenario where float32 histogram performance proves critically important on a scope system, it would be useful to look at
# http://cuda-programming.blogspot.com/2013/03/optimization-in-histogram-cuda-code.html  The various approaches taken here are generally the same
# as those we tried in opencl, with the wrinkle that in part 4, the author goes one step further than we did, using nvidia's absurdly good cuda profiler
# to identify memory bank contention, which he resolves for a huge throughput improvement.
def histogram(im, bin_count, range_, mask=None, with_overflow_bins=False, per_channel_thread_count=2, return_future=False):
    """im: The 2D or 3D ndarray for which histogram is computed.
    range_: An indexable sequence of at least two elements, castable to float32, representing the closed interval which is divided into bin_count number
    of bins comprising the histogram.
    mask: None or a 2D bool or uint8 ndarray with neither dimension smaller than the corresponding dimension of im.  If mask is not None, only image
    pixels with non-zero mask counterparts contribute to the histogram.  Mask pixels outside of im have no impact.  If mask is None, all image pixels
    are included.
    with_overfloat_bins: If true, the first and last histogram bins represent the number of image pixels falling below and above range_, respectively.
    return_future: If not False, a concurrent.futures.Future is returned.

    Returns a channel_count x bin_count numpy array uint32 values."""
    assert im.ndim in (2, 3)
    assert range_[0] < range_[1]
    if mask is not None:
        assert mask.ndim == 2
        assert mask.dtype in (numpy.uint8, numpy.bool)
    if with_overflow_bins:
        assert bin_count >= 4
    else:
        assert bin_count >= 2
    if im.ndim == 2:
        def proc():
            return _histogram(im, bin_count, range_, mask, with_overflow_bins)
    else:
        def channel_proc(axis):
            return _histogram(im[..., axis], bin_count, range_, mask, with_overflow_bins)
        def proc():
            futes = [pool.submit(channel_proc, axis) for axis in range(im.shape[2])]
            return numpy.vstack(fute.result() for fute in futes)
    return pool.submit(proc) if return_future else proc()

def statistics(im, is_twelve_bit=False, mask=None, per_channel_thread_count=2, return_future=False):
    assert im.ndim in (2, 3)
    if im.ndim == 2:
        def proc():
            return _statistics(im, is_twelve_bit, mask)
    else:
        def channel_proc(channel):
            return _statistics(im[..., channel], is_twelve_bit, mask)
        def proc():
            results = [channel_proc(channel) for channel in range(im.shape[2])]
            return NDImageStatistics(
                numpy.vstack((result.histogram for result in results)),
                numpy.hstack((result.max_bin for result in results)),
                numpy.vstack((result.min_max_intensity for result in results)))
    return pool.submit(proc) if return_future else proc()

def bundle_float_stats_into_future(histogram, extremae):
    def proc():
        return NDImageStatistics(
            histogram,
            histogram.argmax() if histogram.ndim == 1 else numpy.array([ch.argmax() for ch in histogram]),
            extremae)
    return pool.submit(proc)