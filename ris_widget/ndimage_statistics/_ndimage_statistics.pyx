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

import cython
from libcpp cimport bool
cimport numpy
import numpy

cdef extern from "_ndimage_statistics_impl.h":
    # The following two lines work around Cython's lack of support for templates parameterized by literals
    ctypedef bool bool_t "true"
    ctypedef bool bool_f "false"
    size_t bin_count[C]()
    void _min_max[C](
        const C* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
        C* min_max
        )
    void _masked_min_max[C](
        const C* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
        const numpy.uint8_t* mask, const Py_ssize_t* mask_shape, const Py_ssize_t* mask_strides,
        C* min_max
        )
    void _ranged_hist[C, with_overflow_bins](
        const C* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
        const C& range_min, const C& range_max, const Py_ssize_t& bin_count,
        numpy.uint32_t* hist
        )
    void _masked_ranged_hist[C, with_overflow_bins](
        const C* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
        const numpy.uint8_t* mask, const Py_ssize_t* mask_shape, const Py_ssize_t* mask_strides,
        const C& range_min, const C& range_max, const Py_ssize_t& bin_count,
        numpy.uint32_t* hist
        )
    void _hist_min_max[C, is_twelve_bit](
        const C* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
        numpy.uint32_t* hist, C* min_max
        )
    void _masked_hist_min_max[C, is_twelve_bit](
        const C* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
        const numpy.uint8_t* mask, const Py_ssize_t* mask_shape, const Py_ssize_t* mask_strides,
        numpy.npy_uint32* hist,
        C* min_max
        )

cpdef min_max_float32(numpy.float32_t[:, :] im, numpy.float32_t[:] min_max):
    assert min_max.shape[0] >= 2
    _min_max[numpy.float32_t](
        &im[0][0], &im.shape[0], &im.strides[0],
        &min_max[0])

cpdef masked_min_max_float32(numpy.float32_t[:, :] im, numpy.uint8_t[:, :] mask, numpy.float32_t[:] min_max):
    assert min_max.shape[0] >= 2
    assert mask.shape[0] >= im.shape[0]
    assert mask.shape[1] >= im.shape[1]
    _masked_min_max[numpy.float32_t](
        &im[0][0], &im.shape[0], &im.strides[0],
        &mask[0][0], &mask.shape[0], &mask.strides[0],
        &min_max[0])

cpdef ranged_hist_float32(numpy.float32_t[:, :] im, range_min, range_max, bin_count, with_overflow_bins, numpy.uint32_t[:] hist):
    assert bin_count >= (4 if with_overflow_bins else 2)
    assert hist.shape[0] == bin_count
    assert range_min < range_max
    if with_overflow_bins:
        _ranged_hist[numpy.float32_t, bool_t](
            &im[0][0], &im.shape[0], &im.strides[0],
            range_min, range_max, bin_count,
            &hist[0])
    else:
        _ranged_hist[numpy.float32_t, bool_f](
            &im[0][0], &im.shape[0], &im.strides[0],
            range_min, range_max, bin_count,
            &hist[0])

cpdef masked_ranged_hist_float32(numpy.float32_t[:, :] im, numpy.uint8_t[:, :] mask, range_min, range_max, bin_count, with_overflow_bins, numpy.uint32_t[:] hist):
    assert bin_count >= (4 if with_overflow_bins else 2)
    assert hist.shape[0] == bin_count
    assert range_min < range_max
    if with_overflow_bins:
        _masked_ranged_hist[numpy.float32_t, bool_t](
            &im[0][0], &im.shape[0], &im.strides[0],
            &mask[0][0], &mask.shape[0], &mask.strides[0],
            range_min, range_max, bin_count,
            &hist[0])
    else:
        _masked_ranged_hist[numpy.float32_t, bool_f](
            &im[0][0], &im.shape[0], &im.strides[0],
            &mask[0][0], &mask.shape[0], &mask.strides[0],
            range_min, range_max, bin_count,
            &hist[0])

cpdef hist_min_max_uint8(numpy.uint8_t[:, :] im, numpy.uint32_t[:] hist, numpy.uint8_t[:] min_max):
    assert min_max.shape[0] >= 2
    assert hist.shape[0] == bin_count[numpy.uint8_t]()
    _hist_min_max[numpy.uint8_t, bool_f](
        &im[0][0], &im.shape[0], &im.strides[0],
        &hist[0],
        &min_max[0])

cpdef hist_min_max_uint12(numpy.uint16_t[:, :] im, numpy.uint32_t[:] hist, numpy.uint16_t[:] min_max):
    assert min_max.shape[0] >= 2
    assert hist.shape[0] == bin_count[numpy.uint16_t]()
    _hist_min_max[numpy.uint16_t, bool_t](
        &im[0][0], &im.shape[0], &im.strides[0],
        &hist[0],
        &min_max[0])

cpdef hist_min_max_uint16(numpy.uint16_t[:, :] im, numpy.uint32_t[:] hist, numpy.uint16_t[:] min_max):
    assert min_max.shape[0] >= 2
    assert hist.shape[0] == bin_count[numpy.uint16_t]()
    _hist_min_max[numpy.uint16_t, bool_f](
        &im[0][0], &im.shape[0], &im.strides[0],
        &hist[0],
        &min_max[0])

cpdef masked_hist_min_max_uint8(numpy.uint8_t[:, :] im, numpy.uint8_t[:, :] mask, numpy.uint32_t[:] hist, numpy.uint8_t[:] min_max):
    assert min_max.shape[0] >= 2
    assert hist.shape[0] == bin_count[numpy.uint8_t]()
    _masked_hist_min_max[numpy.uint8_t, bool_f](
        &im[0][0], &im.shape[0], &im.strides[0],
        &mask[0][0], &mask.shape[0], &mask.strides[0],
        &hist[0],
        &min_max[0])

cpdef masked_hist_min_max_uint12(numpy.uint16_t[:, :] im, numpy.uint8_t[:, :] mask, numpy.uint32_t[:] hist, numpy.uint16_t[:] min_max):
    assert min_max.shape[0] >= 2
    assert hist.shape[0] == bin_count[numpy.uint16_t]()
    _masked_hist_min_max[numpy.uint16_t, bool_t](
        &im[0][0], &im.shape[0], &im.strides[0],
        &mask[0][0], &mask.shape[0], &mask.strides[0],
        &hist[0], &min_max[0])

cpdef masked_hist_min_max_uint16(numpy.uint16_t[:, :] im, numpy.uint8_t[:, :] mask, numpy.uint32_t[:] hist, numpy.uint16_t[:] min_max):
    assert min_max.shape[0] >= 2
    assert hist.shape[0] == bin_count[numpy.uint16_t]()
    _masked_hist_min_max[numpy.uint16_t, bool_f](
        &im[0][0], &im.shape[0], &im.strides[0],
        &mask[0][0], &mask.shape[0], &mask.strides[0],
        &hist[0], &min_max[0])
