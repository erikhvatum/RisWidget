// The MIT License (MIT)
//
// Copyright (c) 2015 WUSTL ZPLAB
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Authors: Erik Hvatum <ice.rikh@gmail.com>

#pragma once
#include <Python.h>
#include <numpy/npy_common.h>

#define _USE_MATH_DEFINES
#include <cmath>

void _hist_min_max_uint16(const npy_uint16* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
                          npy_uint32* hist, npy_uint16* min_max);

void _hist_min_max_uint12(const npy_uint16* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
                          npy_uint32* hist, npy_uint16* min_max);

void _hist_min_max_uint8(const npy_uint8* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
                         npy_uint32* hist, npy_uint8* min_max);
