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

def _stretch_mask(im, mask):
    if im.shape != mask.shape:
        x_lut = numpy.linspace(0, mask.shape[0], im.shape[0], endpoint=False, dtype=numpy.uint32)
        y_lut = numpy.linspace(0, mask.shape[1], im.shape[1], endpoint=False, dtype=numpy.uint32)
        xx_lut, yy_lut = numpy.meshgrid(x_lut, y_lut, indexing='xy')
        mask = mask[xx_lut, yy_lut].T
    return mask

def _min_max(im, mask=None):
    if mask is not None:
        mask = _stretch_mask(im, mask)
        im = numpy.ma.array(im, dtype=im.dtype, copy=False, mask=~mask)
    return numpy.array((im.min(), im.max()), dtype=im.dtype)

def _histogram(im, bin_count, range_, mask=None, with_overflow_bins=False):
    if mask is not None:
        mask = _stretch_mask(im, mask)
        im = im[mask]
    if with_overflow_bins:
        assert bin_count >= 3
        hist = numpy.zeros((bin_count,), dtype=numpy.uint32)
        hist[1:-1] = numpy.histogram(im, bins=bin_count - 2, range=range_, density=False)[0].astype(numpy.uint32)
        hist[0] = (im < range_[0]).sum()
        hist[-1] = (im > range_[1]).sum()
        return hist
    else:
        assert bin_count >= 1
        return numpy.histogram(im, bins=bin_count, range=range_, density=False)[0].astype(numpy.uint32)

def _statistics(im, twelve_bit, mask=None, use_open_mp=False):
    if im.dtype == numpy.uint8:
        min_max = numpy.zeros((2,), dtype=numpy.uint8)
        bin_count = 256
        range_ = (0, 255)
    elif im.dtype == numpy.uint16:
        min_max = numpy.zeros((2,), dtype=numpy.uint16)
        bin_count = 1024
        range_ = (0, 4095) if twelve_bit else (0, 65535)
    else:
        raise NotImplementedError('Support for dtype of supplied im argument not implemented.')
    if mask is not None:
        mask = _stretch_mask(im, mask)
        im = im[mask]
    hist = numpy.histogram(im, bins=bin_count, range=range_, density=False)[0].astype(numpy.uint32)
    min_max[0] = im.min()
    min_max[1] = im.max()
    return NDImageStatistics(hist, hist.argmax(), min_max)