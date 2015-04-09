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
cimport numpy
import numpy

cdef extern from "_ndimage_statistics_impl.h":
    void _hist_min_max_uint16(const numpy.uint16_t* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
                              numpy.uint32_t* hist, numpy.uint16_t* min_max)
    void _hist_min_max_uint12(const numpy.uint16_t* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
                              numpy.uint32_t* hist, numpy.uint16_t* min_max)
    void _hist_min_max_uint8(const numpy.uint8_t* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
                              numpy.uint32_t* hist, numpy.uint8_t* min_max)

cpdef hist_min_max_uint16(numpy.uint16_t[:, :] arr, numpy.uint32_t[:] hist, numpy.uint16_t[:] min_max):
    assert hist.shape[0] == 1024
    _hist_min_max_uint16(&arr[0][0], &arr.shape[0], &arr.strides[0],
                         &hist[0], &min_max[0])

cpdef hist_min_max_uint12(numpy.uint16_t[:, :] arr, numpy.uint32_t[:] hist, numpy.uint16_t[:] min_max):
    assert hist.shape[0] == 1024
    _hist_min_max_uint12(&arr[0][0], &arr.shape[0], &arr.strides[0],
                         &hist[0], &min_max[0])

cpdef hist_min_max_uint8(numpy.uint8_t[:, :] arr, numpy.uint32_t[:] hist, numpy.uint8_t[:] min_max):
    assert hist.shape[0] == 256
    _hist_min_max_uint8(&arr[0][0], &arr.shape[0], &arr.strides[0],
                         &hist[0], &min_max[0])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef shist_min_max_uint16(numpy.uint16_t[:, :] arr, numpy.uint32_t[:] hist, numpy.uint16_t[:] min_max):
    cdef numpy.uint16_t v
    assert hist.shape[0] == 1024
    cdef Py_ssize_t i, j
    with nogil:
        hist[:] = 0
        min_max[0] = arr[0,0]
        min_max[1] = arr[0,0]
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                hist[v >> 6] += 1
                if v < min_max[0]:
                    min_max[0] = v
                elif v > min_max[1]:
                    min_max[1] = v


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef shist_min_max_uint12(numpy.uint16_t[:, :] arr, numpy.uint32_t[:] hist, numpy.uint16_t[:] min_max):
    cdef numpy.uint16_t v
    assert hist.shape[0] == 1024
    cdef Py_ssize_t i, j
    with nogil:
        hist[:] = 0
        min_max[0] = arr[0,0]
        min_max[1] = arr[0,0]
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                hist[v >> 2] += 1
                if v < min_max[0]:
                    min_max[0] = v
                elif v > min_max[1]:
                    min_max[1] = v

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef shist_min_max_uint8(numpy.uint8_t[:, :] arr, numpy.uint32_t[:] hist, numpy.uint8_t[:] min_max):
    cdef numpy.uint8_t v
    assert hist.shape[0] == 256
    cdef Py_ssize_t i, j
    with nogil:
        hist[:] = 0
        min_max[0] = arr[0,0]
        min_max[1] = arr[0,0]
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                hist[v] += 1
                if v < min_max[0]:
                    min_max[0] = v
                elif v > min_max[1]:
                    min_max[1] = v

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef hist_min_max_float32(numpy.float32_t[:, :] arr, numpy.uint32_t[:] hist, numpy.float32_t[:] min_max, numpy.float32_t hist_min, numpy.float32_t hist_max):
    cdef numpy.float32_t v
    cdef numpy.uint32_t n_bins = hist.shape[0]
    cdef Py_ssize_t i, j, bin
    cdef numpy.float32_t bin_factor = (n_bins - 1) / (hist_max - hist_min)
    with nogil:
        hist[:] = 0
        min_max[0] = arr[0,0]
        min_max[1] = arr[0,0]
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                bin = <Py_ssize_t> (bin_factor * (v - hist_min))
                hist[bin] += 1
                if v < min_max[0]:
                    min_max[0] = v
                elif v > min_max[1]:
                    min_max[1] = v
