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

#define _USE_MATH_DEFINES
#include <cmath>

#if defined(__UINT32_MAX__) || defined(UINT32_MAX)
 #include <inttypes.h>
#else
 typedef unsigned char uint8_t;
 typedef unsigned short uint16_t;
 typedef unsigned long uint32_t;
 typedef unsigned long long uint64_t;
#endif

void _hist_min_max_uint16(const uint16_t* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
                          uint32_t* hist, uint16_t* min_max);

void _hist_min_max_uint12(const uint16_t* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
                          uint32_t* hist, uint16_t* min_max);

void _hist_min_max_uint8(const uint8_t* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
                         uint32_t* hist, uint8_t* min_max);
