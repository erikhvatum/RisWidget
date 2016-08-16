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

#include "NDImageStatistics.h"

Luts luts{200};

const std::unordered_map<std::type_index, std::string> component_type_names {
    {typeid(std::int8_t), "int8"},
    {typeid(std::uint8_t), "uint8"},
    {typeid(std::int16_t), "int16"},
    {typeid(std::uint16_t), "uint16"},
    {typeid(std::int32_t), "int32"},
    {typeid(std::uint32_t), "uint32"},
    {typeid(std::int64_t), "int64"},
    {typeid(std::uint64_t), "uint64"},
    {typeid(float), "float"},
    {typeid(double), "double"},
};

template<>
std::size_t bin_count<std::uint8_t>()
{
    return 256;
}

template<>
std::size_t bin_count<std::int8_t>()
{
    return 256;
}

BitmapMask::BitmapMask(typed_array_t<std::uint8_t>& bitmap_)
  : bitmap(bitmap_),
    bitmap_bi(bitmap.request())
{
}

CircularMask::CircularMask(double center_x_, double center_y_, double radius_)
  : center_x(center_x_),
    center_y(center_y_),
    radius(radius_)
{
}

CircularMask::CircularMask(TupleArg t)
  : center_x(std::get<0>(std::get<0>(t))),
    center_y(std::get<1>(std::get<0>(t))),
    radius(std::get<1>(t))
{
}
