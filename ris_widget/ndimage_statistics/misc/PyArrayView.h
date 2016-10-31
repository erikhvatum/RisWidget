// The MIT License (MIT)
//
// Copyright (c) 2016 WUSTL ZPLAB
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
#include <atomic>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <memory>

// Similar to pybind11::buffer_info except with less indirection, a GIL-safe destructor, and easily stored in an
// std::shared_ptr. Useful for working with an array received from Python.
struct PyArrayView
  : Py_buffer
{
    // Not GIL-safe (requires but does not attempt to acquire GIL) 
    explicit PyArrayView(pybind11::array& array, bool writeable=false);
    PyArrayView() = delete;
    PyArrayView(const PyArrayView&) = delete;
    PyArrayView& operator = (const PyArrayView&) = delete;
    // GIL-safe (does not require GIL)
    PyArrayView(PyArrayView&& rhs);
    // GIL-safe (requires and does acquire the GIL)
    ~PyArrayView();

    std::atomic<bool> is_vacated;
};
