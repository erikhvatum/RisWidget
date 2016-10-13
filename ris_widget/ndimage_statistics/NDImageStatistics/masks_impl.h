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

#include "masks.h"

template<typename T>
void Mask<T>::expose_via_pybind11(py::module& m)
{
    std::string s = std::string("_Mask_") + component_type_names[std::type_index(typeid(T))];
    py::class_<Mask<T>, std::shared_ptr<Mask<T>>>(m, s.c_str());
}



template<typename T, BitmapMaskDimensionVsImage T_W, BitmapMaskDimensionVsImage T_H>
void BitmapMask<T, T_W, T_H>::expose_via_pybind11(py::module& m)
{
    std::string s = std::string("_BitmapMask_") + component_type_names[std::type_index(typeid(T))];
    py::class_<BitmapMask<T, T_W, T_H>, std::shared_ptr<BitmapMask<T, T_W, T_H>>, Mask<T>>(m, s.c_str());
}

template<typename T, BitmapMaskDimensionVsImage T_W, BitmapMaskDimensionVsImage T_H>
BitmapMask<T, T_W, T_H>::BitmapMask(PyArrayView&& bitmap_view_)
  : bitmap_view(bitmap_view_)
{
}



template<typename T>
void CircularMask<T>::expose_via_pybind11(py::module& m)
{
    std::string s = std::string("_CircularMask_") + component_type_names[std::type_index(typeid(T))];
    py::class_<CircularMask<T>, std::shared_ptr<CircularMask<T>>, Mask<T>>(m, s.c_str())
        .def_readonly("center_x", &CircularMask::center_x)
        .def_readonly("center_y", &CircularMask::center_y)
        .def_readonly("radius", &CircularMask::radius);
}

template<typename T>
CircularMask<T>::CircularMask(double center_x_, double center_y_, double radius_)
  : center_x(center_x_),
    center_y(center_y_),
    radius(radius_)
{
}

template<typename T>
CircularMask<T>::CircularMask(TupleArg t)
  : center_x(std::get<0>(std::get<0>(t))),
    center_y(std::get<1>(std::get<0>(t))),
    radius(std::get<1>(t))
{
}
