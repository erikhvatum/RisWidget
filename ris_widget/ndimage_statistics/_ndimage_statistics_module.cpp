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

#include <pybind11/pybind11.h>
#include "_ndimage_statistics.h"
#include "resampling_lut.h"

namespace py = pybind11;

// It would be nice to use distinct py::array_t</*element type*/> overload definitions rather than dispatching from a 
// single frontend, but this does not work as py::array_t attempts to cast.  That is, supplied with an numpy array of dtype 
// numpy.uint64, m.def("min_max", [](py::array_t<float>...) would be called with temporary arguments holding inputs 
// converted to float arrays.  This is never something we want. 
void py_min_max(py::buffer im, py::buffer& min_max)
{
    py::buffer_info im_info{im.request()}, min_max_info{min_max.request()};
    if(im_info.ndim != 2)
        throw std::invalid_argument("im argument must be a 2 dimensional buffer object (such as a numpy array).");
    if(min_max_info.ndim != 1)
        throw std::invalid_argument("min_max arugment must be a 1 dimensional buffer object (such as a numpy array).");
    if(min_max_info.shape[0] != 2)
        throw std::invalid_argument("min_max argument must contain exactly 2 elements.");
    if(im_info.format != min_max_info.format)
        throw std::invalid_argument(
            "im and min_max arguments must be the same format (or dtype, in the case where they are numpy arays).");
    if(im_info.format == py::format_descriptor<float>::value())
        ::min_max((float*)im_info.ptr, im_info.shape.data(), im_info.strides.data(), (float*)min_max_info.ptr);
    else if(im_info.format == py::format_descriptor<std::uint8_t>::value())
        ::min_max((std::uint8_t*)im_info.ptr, im_info.shape.data(), im_info.strides.data(), (std::uint8_t*)min_max_info.ptr);
    else if(im_info.format == py::format_descriptor<std::uint16_t>::value())
        ::min_max((std::uint16_t*)im_info.ptr, im_info.shape.data(), im_info.strides.data(), (std::uint16_t*)min_max_info.ptr);
    else if(im_info.format == py::format_descriptor<std::uint32_t>::value())
        ::min_max((std::uint32_t*)im_info.ptr, im_info.shape.data(), im_info.strides.data(), (std::uint32_t*)min_max_info.ptr);
    else if(im_info.format == py::format_descriptor<std::uint64_t>::value())
        ::min_max((std::uint64_t*)im_info.ptr, im_info.shape.data(), im_info.strides.data(), (std::uint64_t*)min_max_info.ptr);
    else if(im_info.format == py::format_descriptor<double>::value())
        ::min_max((double*)im_info.ptr, im_info.shape.data(), im_info.strides.data(), (double*)min_max_info.ptr);
    else
        throw std::invalid_argument("Only uint8, uint16, uint32, uint64, float32, and float64 buffers are supported.");
}

PYBIND11_PLUGIN(_ndimage_statistics)
{
    py::module m("_ndimage_statistics", "ris_widget.ndimage_statistics._ndimage_statistics module");

    m.def("min_max", &py_min_max);

    py::class_<Luts>(m, "Luts")
        .def(py::init<const std::size_t&>())
        .def("getLut", [](Luts& luts, const std::uint32_t& fromSamples, const std::uint32_t& toSamples){
            luts.getLut(fromSamples, toSamples);
        });

    return m.ptr();
}