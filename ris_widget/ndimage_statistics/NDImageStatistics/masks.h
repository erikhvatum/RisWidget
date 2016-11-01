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
#include "common.h"

template<typename T>
struct Mask
{
    static void expose_via_pybind11(py::module& m);

    Mask() = default;
    Mask(const Mask&) = delete;
    Mask& operator = (const Mask&) = delete;
    virtual ~Mask() = default;
};

enum class BitmapMaskDimensionVsImage
{
    Smaller,
    Same,
    Larger
};

template<typename T, BitmapMaskDimensionVsImage T_W, BitmapMaskDimensionVsImage T_H>
struct BitmapMask
  : Mask<T>
{
    static void expose_via_pybind11(py::module& m);

    explicit BitmapMask(std::unique_ptr<PyArrayView>&& bitmap_view_);

    std::unique_ptr<PyArrayView> bitmap_view;
};

template<typename T>
struct CircularMask
  : Mask<T>
{
    using TupleArg = const std::tuple<std::tuple<double, double>, double>&;

    static void expose_via_pybind11(py::module& m);

    CircularMask(double center_x_, double center_y_, double radius_);
    explicit CircularMask(TupleArg t);

    const std::int32_t center_x, center_y, radius;
};

#include "masks_impl.h"
