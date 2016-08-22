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
CursorBase<T>::CursorBase(std::shared_ptr<typed_array_t<T>>& data_py_)
  : data_py(data_py_),
    data_bi(data_py->request()),
    scanline_stride(data_bi.strides[1]),
    pixel_stride(data_bi.strides[0]),
    component_stride(data_bi.strides.back()),
    component(reinterpret_cast<const T*&>(component_raw))
{
}

template<typename T>
void NonPerComponentMaskCursor<T>::advance_component()
{
    assert(this->pixel_valid);
    assert(this->component_valid);
    this->component_raw += this->component_stride;
    this->component_valid = this->component_raw < this->components_raw_end;
}

template<typename T>
void Mask<T>::expose_via_pybind11(py::module& m)
{
    std::string s = std::string("_Mask_") + component_type_names[std::type_index(typeid(T))];
    py::class_<Mask<T>, std::shared_ptr<Mask<T>>>(m, s.c_str());
}

// template<typename T>
// Cursor<T>* Mask<T>::make_cursor(std::shared_ptr<typed_array_t<T>>& data_py)
// {
//  py::gil_scoped_acquire acquire_gil;
//  return new Cursor<T>(data_py);
// }

template<typename T, typename MASK_T>
void Cursor<T, MASK_T>::advance_pixel()
{
}

template<typename T>
void BitmapMask<T>::expose_via_pybind11(py::module& m)
{
    std::string s = std::string("_BitmapMask_") + component_type_names[std::type_index(typeid(T))];
    py::class_<BitmapMask<T>, std::shared_ptr<BitmapMask<T>>>(m, s.c_str(), py::base<Mask<T>>())
        .def_property_readonly("bitmap", [](BitmapMask<T>& v){return *v.bitmap_py;});
}

template<typename T>
BitmapMask<T>::BitmapMask(typed_array_t<std::uint8_t>& bitmap_py_)
{
    py::buffer_info bi(bitmap_py_.request());
    if(bi.ndim != 2) throw std::invalid_argument("bitmap mask must be 2 dimensional.");
    if(bi.strides[0] > bi.strides[1]) throw std::invalid_argument("bitmap mask striding must be (X, Y).");
    bitmap_py.reset(new typed_array_t<std::uint8_t>(bitmap_py_), &safe_py_deleter);
}

template<typename T>
void Cursor<T, BitmapMask<T>>::advance_pixel()
{
}

template<typename T>
void CircularMask<T>::expose_via_pybind11(py::module& m)
{
    std::string s = std::string("_CircularMask_") + component_type_names[std::type_index(typeid(T))];
    py::class_<CircularMask<T>, std::shared_ptr<CircularMask<T>>>(m, s.c_str(), py::base<Mask<T>>())
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

template<typename T>
void Cursor<T, CircularMask<T>>::advance_pixel()
{
}

template<typename T>
void StatsBase<T>::expose_via_pybind11(py::module& m)
{
    std::string s = std::string("_StatsBase_") + component_type_names[std::type_index(typeid(T))];
    py::class_<StatsBase<T>, std::shared_ptr<StatsBase<T>>>(m, s.c_str())
        .def_readonly("extrema", &StatsBase<T>::extrema)
        .def_readonly("max_bin", &StatsBase<T>::max_bin)
        .def_property_readonly("histogram", [](StatsBase<T>& v){return *v.histogram_py;});
}

template<typename T>
StatsBase<T>::StatsBase()
  : extrema(0, 0),
    max_bin(0)
{
    py::gil_scoped_acquire acquire_gil;
    histogram_py.reset(
       new typed_array_t<std::uint64_t>(
          py::buffer_info(
             nullptr,
             sizeof(std::uint64_t),
             py::format_descriptor<std::uint64_t>::value,
             1,
             {bin_count<T>()},
             {sizeof(std::uint64_t)})),
       &safe_py_deleter);
}

template<typename T>
void FloatStatsBase<T>::expose_via_pybind11(py::module& m)
{
    StatsBase<T>::expose_via_pybind11(m);
    std::string s = std::string("_FloatStatsBase_") + component_type_names[std::type_index(typeid(T))];
    py::class_<FloatStatsBase<T>, std::shared_ptr<FloatStatsBase<T>>>(m, s.c_str(), py::base<StatsBase<T>>())
        .def_readonly("NaN_count", &FloatStatsBase<T>::NaN_count)
        .def_readonly("neg_inf_count", &FloatStatsBase<T>::neg_inf_count)
        .def_readonly("pos_inf_count", &FloatStatsBase<T>::pos_inf_count);
}

template<typename T>
FloatStatsBase<T>::FloatStatsBase()
  : NaN_count(0),
    neg_inf_count(0),
    pos_inf_count(0)
{
}

// Note that concrete specializations for T=float and T=double are found in NDImageStatistics.cpp
template<typename T>
void Stats<T>::expose_via_pybind11(py::module& m)
{
    StatsBase<T>::expose_via_pybind11(m);
    std::string s = std::string("_Stats_") + component_type_names[std::type_index(typeid(T))];
    py::class_<Stats<T>, std::shared_ptr<Stats<T>>>(m, s.c_str(), py::base<StatsBase<T>>());
}

template<typename T>
void ImageStats<T>::expose_via_pybind11(py::module& m)
{
    Stats<T>::expose_via_pybind11(m);
    std::string s = std::string("_ImageStats_") + component_type_names[std::type_index(typeid(T))];
    py::class_<ImageStats<T>, std::shared_ptr<ImageStats<T>>>(m, s.c_str(), py::base<Stats<T>>())
        .def_readonly("channel_stats", &ImageStats<T>::channel_stats);
    s = std::string("_Stats_") + component_type_names[std::type_index(typeid(T))] + "_list";
    py::bind_vector<std::shared_ptr<Stats<T>>>(m, s);
}

template<typename T>
void NDImageStatistics<T>::expose_via_pybind11(py::module& m, const std::string& s)
{
    Mask<T>::expose_via_pybind11(m);
    BitmapMask<T>::expose_via_pybind11(m);
    CircularMask<T>::expose_via_pybind11(m);
    ImageStats<T>::expose_via_pybind11(m);
    std::string name = "_NDImageStatistics_";
    name += s;
    py::class_<NDImageStatistics<T>, std::shared_ptr<NDImageStatistics<T>>>(m, name.c_str())
        .def("launch_computation", &NDImageStatistics<T>::launch_computation)
        .def_property_readonly("data", [](NDImageStatistics<T>& v){return *v.data_py.get();})
        .def_readonly("mask", &NDImageStatistics<T>::mask)
        .def_property_readonly("image_stats", &NDImageStatistics<T>::get_image_stats)
        .def_readonly("drop_last_channel_from_overall_stats", &NDImageStatistics<T>::drop_last_channel_from_overall_stats);
    // Add overloaded "constructor" function.  pybind11 does not (yet, at time of writing) support templated class
    // instantiation via overloaded constructor defs, but plain function overloading is supported, and we take
    // advantage of this to present a factory function that is semantically similar.
    m.def("NDImageStatistics",
          [](typed_array_t<T>& a, bool b){return new NDImageStatistics<T>(a, b);});
    m.def("NDImageStatistics",
          [](typed_array_t<T>& a, typename CircularMask<T>::TupleArg m, bool b){return new NDImageStatistics<T>(a, m, b);});
    m.def("NDImageStatistics",
          [](typed_array_t<T>& a, typed_array_t<std::uint8_t>& m, bool b){return new NDImageStatistics<T>(a, m, b);});
}



template<typename T>
NDImageStatistics<T>::NDImageStatistics(typed_array_t<T>& data_py_,
                                        bool drop_last_channel_from_overall_stats_)
  : NDImageStatistics<T>(data_py_,
                         std::make_shared<Mask<T>>(),
                         drop_last_channel_from_overall_stats_,
                         &NDImageStatistics<T>::compute<Mask<T>>)
{
}

template<typename T>
NDImageStatistics<T>::NDImageStatistics(typed_array_t<T>& data_py_,
                                        typed_array_t<std::uint8_t>& mask_,
                                        bool drop_last_channel_from_overall_stats_)
  : NDImageStatistics<T>(data_py_,
                         std::make_shared<BitmapMask<T>>(mask_),
                         drop_last_channel_from_overall_stats_,
                         &NDImageStatistics<T>::compute<BitmapMask<T>>)
{
}

template<typename T>
NDImageStatistics<T>::NDImageStatistics(typed_array_t<T>& data_py_,
                                        typename CircularMask<T>::TupleArg mask_,
                                        bool drop_last_channel_from_overall_stats_)
  : NDImageStatistics<T>(data_py_,
                         std::make_shared<CircularMask<T>>(mask_),
                         drop_last_channel_from_overall_stats_,
                         &NDImageStatistics<T>::compute<CircularMask<T>>)
{
}

template<typename T>
NDImageStatistics<T>::NDImageStatistics(typed_array_t<T>& data_py_,
                                        std::shared_ptr<Mask<T>>&& mask_,
                                        bool drop_last_channel_from_overall_stats_,
                                        std::shared_ptr<ImageStats<T>>(*compute_fn_)(std::weak_ptr<NDImageStatistics<T>>))
  : data_py(std::shared_ptr<typed_array_t<T>>(new typed_array_t<T>(data_py_), &safe_py_deleter)),
    mask(mask_),
    drop_last_channel_from_overall_stats(drop_last_channel_from_overall_stats_),
    compute_fn(compute_fn_)
{
    py::buffer_info bi{data_py_.request()};
    if(bi.ndim < 2 || bi.ndim > 3) throw std::invalid_argument("data argument must be 2 or 3 dimensional.");
    if(bi.strides[0] > bi.strides[1]) throw std::invalid_argument("data argument striding must be (X, Y) or (X, Y, C).");
}

template<typename T>
void NDImageStatistics<T>::launch_computation()
{
    image_stats = std::async(std::launch::async, std::bind(compute_fn, std::weak_ptr<NDImageStatistics<T>>(this->shared_from_this())));
}

template<typename T>
std::shared_ptr<ImageStats<T>> NDImageStatistics<T>::get_image_stats()
{
    // Avoid deadlocking when called from Python just as worker thread is starting (GIL would not be released until we 
    // return, but we can not return until the future is ready, and the future needs the GIL in order to set up data 
    // structures). 
    py::gil_scoped_release release_gil;
    return image_stats.get();
}

template<typename T>
template<typename MASK_T>
std::shared_ptr<ImageStats<T>> NDImageStatistics<T>::compute(std::weak_ptr<NDImageStatistics<T>> this_wp)
{
    std::shared_ptr<ImageStats<T>> stats(new ImageStats<T>());
    std::shared_ptr<NDImageStatistics<T>> this_sp(this_wp.lock());
    if(this_sp)
    {
        std::shared_ptr<typed_array_t<T>> data_py{this_sp->data_py};
        std::shared_ptr<MASK_T> mask{std::dynamic_pointer_cast<MASK_T>(this_sp->mask)};
        bool drop_last_channel_from_overall_stats{this_sp->drop_last_channel_from_overall_stats};
        this_sp.reset();
    }
    return stats;
}
