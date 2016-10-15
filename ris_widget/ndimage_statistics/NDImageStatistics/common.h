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
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _ndimage_statistics_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/npy_common.h>
#include <numpy/arrayobject.h>
#define _USE_MATH_DEFINES
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>
#include <stdexcept>
#include <sstream>
#include <string>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>
#include "../misc/Luts.h"
#include "../misc/PyArrayView.h"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(_T_, std::shared_ptr<_T_>);

// The second template parameter of py::array_t<..> is ExtraFlags, an enum value that defaults to py::array::forecast.
// py::array::forecast causes array typecasting/conversion in order to satisfy an overloaded c++ function argument when
// we instead want the overload not requiring typecasting to be called - or a ValueError exception if there is no
// matching overload. Specifying 0 for this argument seems to work, even for zp-strided RGB images, although
// the enum definition suggests 0 is actually equivalent to c_contig.
template<typename T> using typed_array_t = py::array_t<T, 0>;
using mask_tuple_t = std::tuple<std::tuple<double, double>, double>;

extern std::unordered_map<std::type_index, std::string> component_type_names;

template<typename T>
std::uint16_t max_bin_count()
{
    return 1024;
}

template<>
std::uint16_t max_bin_count<std::uint8_t>();

template<>
std::uint16_t max_bin_count<std::int8_t>();

// GIL-aware deleter for Python objects likely to be released and refcount-decremented on another (non-Python) thread
void safe_py_deleter(py::object* py_obj);

// Returns {true, log2(v)} if v is a power of two and {false, 0} otherwise. v may be signed but must be positive.
template<typename T>
std::pair<bool, std::int16_t> power_of_two(T v)
{
    if(!std::is_integral<T>::value)
        throw std::domain_error("power_of_two operates on positive integer and unsigned integer values only.");
    if(v <= 0)
        throw std::out_of_range("The argument supplied to power_of_two must be positive.");
    std::int16_t true_bit_pos=-1;
    for(std::int16_t bit_pos=0; bit_pos < sizeof(T)*8; ++bit_pos, v >>= 1)
    {
        if((v & 1) == 1)
        {
            if(true_bit_pos != -1)
                return {false, 0};
            true_bit_pos = bit_pos;
        }
    }
    return {true, true_bit_pos};
}

// Returns true if v is zero, normal, or subnormal. Equivalent to Python's math.isfinite(..) function.
template<typename T>
bool is_finite(const T& v)
{
    switch(std::fpclassify(v))
    {
    case FP_NORMAL:
    case FP_SUBNORMAL:
    case FP_ZERO:
        return true;
    default:
        return false;
    }
}
