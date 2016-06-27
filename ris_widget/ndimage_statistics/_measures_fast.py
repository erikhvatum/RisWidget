# The MIT License (MIT)
#
# Copyright (c) 2016 WUSTL ZPLAB
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

from ._common import *
from . import _ndimage_statistics

def _min_max(im, mask=None, roi_center_and_radius=None):
    min_max = numpy.empty((2,), dtype=im.dtype)
    if mask is not None:
        _ndimage_statistics.masked_min_max(im, mask, min_max)
    elif roi_center_and_radius is not None:
        assert roi_center_and_radius[1] >= 0
        _ndimage_statistics.roi_min_max(im, roi_center_and_radius[0][0], roi_center_and_radius[0][1], roi_center_and_radius[1], min_max)
    else:
        _ndimage_statistics.min_max(im, min_max)
    return min_max

def _min_max_branching(im, mask=None, roi_center_and_radius=None):
    min_max = numpy.empty((2,), dtype=im.dtype)
    if mask is not None:
        _ndimage_statistics.masked_min_max(im, mask, min_max)
    elif roi_center_and_radius is not None:
        assert roi_center_and_radius[1] >= 0
        _ndimage_statistics.roi_branching_min_max(im, roi_center_and_radius[0][0], roi_center_and_radius[0][1], roi_center_and_radius[1], min_max)
    else:
        _ndimage_statistics.min_max(im, min_max)
    return min_max

def _histogram(im, bin_count, range_, mask=None, roi_center_and_radius=None, with_overflow_bins=False):
    hist = numpy.zeros((bin_count,), dtype=numpy.uint32)
    if mask is not None:
        _ndimage_statistics.masked_ranged_hist(im, mask, range_, hist, with_overflow_bins)
#   elif roi_center_and_radius is not None:
#       assert roi_center_and_radius[1] >= 0
#       _ndimage_statistics.roi_ranged_hist(im, roi_center_and_radius[0], roi_center_and_radius[1], range_, hist, with_overflow_bins)
    else:
        _ndimage_statistics.ranged_hist(im, range_, hist, with_overflow_bins)
    return hist

def _statistics(im, is_twelve_bit, mask=None, roi_center_and_radius=None, use_open_mp=False):
    hist = numpy.empty((256 if im.dtype == numpy.uint8 else 1024,), dtype=numpy.uint32)
    min_max = numpy.empty((2,), dtype=im.dtype)
    if mask is not None:
        _ndimage_statistics.masked_hist_min_max(im, mask, hist, min_max, is_twelve_bit, use_open_mp)
#   elif roi_center_and_radius is not None:
#       assert roi_center_and_radius[1] >= 0
#       _ndimage_statistics.roi_hist_min_max(im, roi_center_and_radius[0], roi_center_and_radius[1], hist, min_max, is_twelve_bit)
    else:
        _ndimage_statistics.hist_min_max(im, hist, min_max, is_twelve_bit)
    return NDImageStatistics(hist, hist.argmax(), min_max)