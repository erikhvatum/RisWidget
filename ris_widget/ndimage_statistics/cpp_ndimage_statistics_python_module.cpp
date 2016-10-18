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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _ndimage_statistics_ARRAY_API
#include <numpy/arrayobject.h>

#include "NDImageStatistics/NDImageStatistics.h"

PYBIND11_PLUGIN(cpp_ndimage_statistics)
{
    import_array();

    py::module m("cpp_ndimage_statistics", "ris_widget.ndimage_statistics.cpp_ndimage_statistics module");

    py::class_<std::vector<std::uint64_t>, std::shared_ptr<std::vector<std::uint64_t>>>(m, "_HistogramBuffer")
        .def_buffer([](std::vector<std::uint64_t>& v) {
            return py::buffer_info(
                v.data(),
                sizeof(std::uint64_t),
                py::format_descriptor<std::uint64_t>::format(),
                1,
                { v.size() },
                { sizeof(std::uint64_t) });
         });

    NDImageStatistics<std::int8_t>  ::expose_via_pybind11(m, "int8");
    NDImageStatistics<std::uint8_t> ::expose_via_pybind11(m, "uint8");
    NDImageStatistics<std::int16_t> ::expose_via_pybind11(m, "int16");
    NDImageStatistics<std::uint16_t>::expose_via_pybind11(m, "uint16");
    NDImageStatistics<std::int32_t> ::expose_via_pybind11(m, "int32");
    NDImageStatistics<std::uint32_t>::expose_via_pybind11(m, "uint32");
    NDImageStatistics<std::int64_t> ::expose_via_pybind11(m, "int64");
    NDImageStatistics<std::uint64_t>::expose_via_pybind11(m, "uint64");
    NDImageStatistics<float>        ::expose_via_pybind11(m, "float32");
    NDImageStatistics<double>       ::expose_via_pybind11(m, "float64");

    return m.ptr();
}