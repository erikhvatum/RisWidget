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

#include "NDImageStatistics.h"

template<typename T>
struct SBase
{
    std::tuple<T, T> get_t(){return t;}
    std::tuple<T, T> t;
    static void sfunc() { std::cout << "SBase::sfunc" << std::endl; }
};

template<typename T>
struct S : SBase<T>, std::enable_shared_from_this<S<T>>
{
    static void sfunc() { std::cout << "S::sfunc" << std::endl; SBase<T>::sfunc(); }
};

template<typename T>
struct SS : S<T>
{
    std::shared_ptr<S<T>> a, b;
    SS() : a(std::make_shared<S<T>>()), b(std::make_shared<S<T>>()){}
    static void sfunc() { std::cout << "SS::sfunc" << std::endl; S<T>::sfunc(); }
};

template<typename T>
void expose(py::module& m)
{
    py::class_<SBase<T>>(m, "SBase");

    py::class_<S<T>, std::shared_ptr<S<T>>>(m, "S", py::base<SBase<T>>())
            .def_readwrite("t", &S<T>::t)
            .def("get_t", &S<T>::get_t);

    py::class_<SS<T>, std::shared_ptr<SS<T>>>(m, "S", py::base<S<T>>())
            .def_readwrite("a", &SS<T>::a)
            .def_readwrite("b", &SS<T>::b);

    m.def("make_SS", []{return new SS<T>();});

    SS<T>::sfunc();
}

PYBIND11_PLUGIN(_ndimage_statistics)
{
    py::module m("_ndimage_statistics", "ris_widget.ndimage_statistics._ndimage_statistics module");

    NDImageStatistics<std::int8_t>  ::expose_via_pybind11(m, "int8");
//    py::bind_vector<std::shared_ptr<ImageStats<std::int8_t>>>(m, "_ImageStats_int8_list");
    NDImageStatistics<std::uint8_t> ::expose_via_pybind11(m, "uint8");
//    py::bind_vector<std::shared_ptr<ImageStats<std::uint8_t>>>(m, "_ImageStats_uint8_list");
//    NDImageStatistics<std::int16_t> ::expose_via_pybind11(m, "int16");
//    NDImageStatistics<std::uint16_t>::expose_via_pybind11(m, "uint16");
//    NDImageStatistics<std::int32_t> ::expose_via_pybind11(m, "int32");
//    NDImageStatistics<std::uint32_t>::expose_via_pybind11(m, "uint32");
//    NDImageStatistics<std::int64_t> ::expose_via_pybind11(m, "int64");
//    NDImageStatistics<std::uint64_t>::expose_via_pybind11(m, "uint64");
//    NDImageStatistics<float        >::expose_via_pybind11(m, "float32");
//    NDImageStatistics<double       >::expose_via_pybind11(m, "float64");

    expose<int>(m);

    std::cout << 

    return m.ptr();
}