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
    static const std::unordered_map<BitmapMaskDimensionVsImage, std::string> w_names {
        {BitmapMaskDimensionVsImage::Smaller, "narrower"},
        {BitmapMaskDimensionVsImage::Same, "same_width"},
        {BitmapMaskDimensionVsImage::Larger, "wider"},
    };
    static const std::unordered_map<BitmapMaskDimensionVsImage, std::string> h_names {
        {BitmapMaskDimensionVsImage::Smaller, "shorter"},
        {BitmapMaskDimensionVsImage::Same, "same_height"},
        {BitmapMaskDimensionVsImage::Larger, "taller"},
    };
    std::string s = std::string("_BitmapMask_") + component_type_names[std::type_index(typeid(T))];
    s += "_";
    switch(T_W)
    {
    case BitmapMaskDimensionVsImage::Smaller:
        s += "narrower";
        break;
    case BitmapMaskDimensionVsImage::Same:
        s += "same_width";
        break;
    case BitmapMaskDimensionVsImage::Larger:
        s += "wider";
        break;
    }
    s += "_and_";
    switch(T_H)
    {
    case BitmapMaskDimensionVsImage::Smaller:
        s += "shorter";
        break;
    case BitmapMaskDimensionVsImage::Same:
        s += "same_height";
        break;
    case BitmapMaskDimensionVsImage::Larger:
        s += "taller";
        break;
    }
    py::class_<BitmapMask<T, T_W, T_H>, std::shared_ptr<BitmapMask<T, T_W, T_H>>, Mask<T>>(m, s.c_str());
}

template<typename T, BitmapMaskDimensionVsImage T_W, BitmapMaskDimensionVsImage T_H>
BitmapMask<T, T_W, T_H>::BitmapMask(std::unique_ptr<PyArrayView>&& bitmap_view_)
  : bitmap_view(std::move(bitmap_view_))
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
  : center_x(0),
    center_y(0),
    radius(0)
{
    if(unlikely(!is_finite(center_x_) || !is_finite(center_y_) || !is_finite(radius_)))
        throw std::invalid_argument("center_x, center_y, and radius must be finite (neither NaN, nor inf, nor -inf).");
    if(unlikely(radius_ < 0))
        throw std::out_of_range("radius_ must be >= 0.");
    center_x_ = std::round(center_x_);
    center_y_ = std::round(center_y_);
    radius_ = std::round(radius_);
    if ( unlikely(
             center_x_ - radius_ < std::numeric_limits<std::int32_t>::min()
          || center_x_ + radius_ > std::numeric_limits<std::int32_t>::max()
          || center_y_ - radius_ < std::numeric_limits<std::int32_t>::min()
          || center_y_ + radius_ > std::numeric_limits<std::int32_t>::max()) )
    {
        std::ostringstream o;
        o << "After rounding center_x, center_y, and radius to the nearest integer, center_x +/- radius and center_y +/- "
             "radius must be in the interval [";
        o << std::numeric_limits<std::int32_t>::min() << ", " << std::numeric_limits<std::int32_t>::max() << "].";
        throw std::out_of_range(o.str());
    }
    const_cast<std::int32_t&>(center_x) = static_cast<std::int32_t>(center_x_);
    const_cast<std::int32_t&>(center_y) = static_cast<std::int32_t>(center_y_);
    const_cast<std::int32_t&>(radius) = static_cast<std::int32_t>(radius_);
}

template<typename T>
CircularMask<T>::CircularMask(TupleArg t)
  : CircularMask(std::get<0>(std::get<0>(t)), std::get<1>(std::get<0>(t)), std::get<1>(t))
{
}
