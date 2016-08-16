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
#include "NDImageStatistics.h"

template<typename T>
std::size_t bin_count()
{
    return 1024;
}

template<typename T>
void StatsBase<T>::expose_via_pybind11(py::module& m)
{
    auto s = std::string("_StatsBase_") + component_type_names[typeinfo(T)];
    py::class_<StatsBase<T>>(m, s.c_str())
        .def_readonly("extrema", &StatsBase<T>::extrema)
        .def("histogram", [](StatsBase<T>& v){return *v.histogram_py;});
}

template<typename T>
StatsBase<T>::StatsBase()
  : extrema(0, 0),
    histogram_py(new typed_array_t<std::uint32_t>(py::buffer_info(nullptr,
                                                                  sizeof(std::uint32_t),
                                                                  py::format_descriptor<std::uint32_t>::value,
                                                                  1,
                                                                  {bin_count<T>()},
                                                                  {sizeof(std::uint32_t)}))),
    histogram(reinterpret_cast<std::uint32_t*>(histogram_py->request().ptr))
{
}

template<typename T>
StatsBase<T>::~StatsBase()
{}

template<typename T>
ImageStats<T>::ImageStats()
{}

template<typename T>
void NDImageStatistics<T>::expose_via_pybind11(py::module& m, const std::string& s)
{
    std::string name("_StatsBase_");
    name = "_Stats_";
    name += s;
//    py::class_<StatsBase<T>>(m, name.c_str())
//        .
//    name += "_List";
//  py::bind_vector<std::shared_ptr<Stats<T>>(m, std::string("_Stats_") + s + "_List");
//    name = "_ImageStats_";
//    name += s;
//    py::class_<ImageStats<T>, std::shared_ptr<ImageStats<T>>>(m, name.c_str()/*, py::base<Stats<T>>()*/)
//            .def
//        .def_readonly("channel_stats");
//            .def("getfoo", []{return std::tuple<int, int>(23,12);});
//      .def_readonly("extrema", &ImageStats<T>::extrema);
    name = "_NDImageStatistics_";
    name += s;
    py::class_<NDImageStatistics<T>, std::shared_ptr<NDImageStatistics<T>>>(m, s.c_str())
//    .def("getfoo", []{return std::tuple<int, int>(23,12);});
//
//    m.def("make_ndimagestatistics", []{return new NDImageStatistics<T>();});
//      .def_readonly("data", &NDImageStatistics<T, MASK>::data)
        .def_readwrite("image_stats", &NDImageStatistics<T>::image_stats);
    // Add overloaded "constructor" function.  pybind11 does not (yet, at time of writing) support templated class
    // instantiation via overloaded constructor defs, but plain function overloading is supported, and we take
    // advantage of this to present a factory function that is semantically similar.
    m.def("NDImageStatistics",
          [](typed_array_t<T>& a, bool b){ return new NDImageStatistics<T>(a, b); });
//  m.def("NDImageStatistics",
//        [](typed_array_t<T>& a, const std::tuple<std::tuple<double, double>, double>& m, bool b){return new NDImageStatistics<T>(a, m, b);});
//  m.def("NDImageStatistics", [](typed_array_t<T>& a, typed_array_t<std::uint8_t>& m, bool b){return new NDImageStatistics<T>(a, m, b);});
}

template<typename T>
NDImageStatistics<T>::NDImageStatistics(typed_array_t<T>& data_py_,
                                        bool drop_last_channel_from_overall_stats)
  : image_stats(new ImageStats<T>())
{
}

//template<typename T>
//NDImageStatistics<T>::NDImageStatistics(typed_array_t<T>& data_py_,
//                                        typed_array_t<std::uint8_t>& mask_,
//                                        bool drop_last_channel_from_overall_stats)
//{
//}
//
//template<typename T>
//NDImageStatistics<T>::NDImageStatistics(typed_array_t<T>& data_py_,
//                                        CircularMask::TupleArg mask_,
//                                        bool drop_last_channel_from_overall_stats)
//{
//}
//
//template<typename T>
//NDImageStatistics<T>::NDImageStatistics(typed_array_t<T>& data_py_,
//                                        std::shared_ptr<const Mask>&& mask_,
//                                        bool drop_last_channel_from_overall_stats)
//// : data(data_),
////  stats(data_, drop_channel_idxs_from_overall)
//{
//}
//
//template<typename T>
//NDImageStatistics<T>::~NDImageStatistics()
//{
//}
