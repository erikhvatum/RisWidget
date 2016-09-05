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
std::uint16_t max_bin_count()
{
    return 1024;
}

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

template<typename T>
CursorBase<T>::CursorBase(PyArrayView& data_view)
  : scanline_valid(false),
    pixel_valid(false),
    component_valid(false),
    scanline_count(data_view.shape[1]),
    scanline_stride(data_view.strides[1]),
    scanline_origin(reinterpret_cast<const T*>(data_view.buf)),
    scanline_raw(nullptr),
    scanlines_raw_end(reinterpret_cast<std::uint8_t*>(data_view.buf) + scanline_stride * scanline_count),
    scanline_width(data_view.shape[0]),
    pixel_stride(data_view.strides[0]),
    component_count(1),
    component_stride(sizeof(T)),
    component(reinterpret_cast<const T*&>(component_raw))
{
    if(data_view.ndim == 3)
    {
        const_cast<std::size_t&>(component_count) = data_view.shape[2];
        const_cast<std::size_t&>(component_stride) = data_view.strides[2];
    }
}

template<typename T>
void NonPerComponentMaskCursor<T>::seek_front_component_of_pixel()
{
    assert(this->scanline_valid);
    assert(this->pixel_valid);
    this->component_raw = this->pixel_raw;
    this->components_raw_end = this->pixel_raw + this->component_stride * this->component_count;
    this->component_valid = this->component_raw < this->components_raw_end;
}

template<typename T>
void NonPerComponentMaskCursor<T>::advance_component()
{
    assert(this->scanline_valid);
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

template<typename T, typename MASK_T>
Cursor<T, MASK_T>::Cursor(PyArrayView& data_view, MASK_T& /*mask_*/)
  : NonPerComponentMaskCursor<T>(data_view)
{
}

template<typename T, typename MASK_T>
void Cursor<T, MASK_T>::seek_front_scanline()
{
    this->scanline_raw = reinterpret_cast<const std::uint8_t*>(this->scanline_origin);
    this->scanline_valid = this->scanline_raw < this->scanlines_raw_end;
}

template<typename T, typename MASK_T>
void Cursor<T, MASK_T>::advance_scanline()
{
    assert(this->scanline_valid);
    this->scanline_raw += this->scanline_stride;
    this->scanline_valid = this->scanline_raw < this->scanlines_raw_end;
    this->pixel_valid = false;
    this->component_valid = false;
}

template<typename T, typename MASK_T>
void Cursor<T, MASK_T>::seek_front_pixel_of_scanline()
{
    assert(this->scanline_valid);
    this->pixel_raw = this->scanline_raw;
    this->pixels_raw_end = this->pixel_raw + this->scanline_width * this->pixel_stride;
    this->pixel_valid = this->pixel_raw < this->pixels_raw_end;
    this->component_valid = false;
}

template<typename T, typename MASK_T>
void Cursor<T, MASK_T>::advance_pixel()
{
    assert(this->scanline_valid);
    assert(this->pixel_valid);
    this->pixel_raw += this->pixel_stride;
    this->pixel_valid = this->pixel_raw < this->pixels_raw_end;
    this->component_valid = false;
}

template<typename T>
void BitmapMask<T>::expose_via_pybind11(py::module& m)
{
    std::string s = std::string("_BitmapMask_") + component_type_names[std::type_index(typeid(T))];
    py::class_<BitmapMask<T>, std::shared_ptr<BitmapMask<T>>>(m, s.c_str(), py::base<Mask<T>>());
//        .def_property_readonly("bitmap", [](BitmapMask<T>& v){return *v.bitmap_py;});
}

template<typename T>
BitmapMask<T>::BitmapMask(typed_array_t<std::uint8_t>& bitmap_py_)
  : bitmap_view(bitmap_py_)
{
    if(bitmap_view.ndim != 2) throw std::invalid_argument("bitmap mask must be 2 dimensional.");
    if(bitmap_view.strides[0] > bitmap_view.strides[1]) throw std::invalid_argument("bitmap mask striding must be (X, Y).");
}

template<typename T>
Cursor<T, BitmapMask<T>>::Cursor(PyArrayView& data_view, BitmapMask<T>& mask_)
  : NonPerComponentMaskCursor<T>(data_view),
    mask(mask_)
{
}

template<typename T>
void Cursor<T, BitmapMask<T>>::seek_front_scanline()
{
}

template<typename T>
void Cursor<T, BitmapMask<T>>::advance_scanline()
{
}

template<typename T>
void Cursor<T, BitmapMask<T>>::seek_front_pixel_of_scanline()
{
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
Cursor<T, CircularMask<T>>::Cursor(PyArrayView& data_view, CircularMask<T>& mask_)
  : NonPerComponentMaskCursor<T>(data_view),
    mask(mask_)
{
}

template<typename T>
void Cursor<T, CircularMask<T>>::seek_front_scanline()
{
}

template<typename T>
void Cursor<T, CircularMask<T>>::advance_scanline()
{
}

template<typename T>
void Cursor<T, CircularMask<T>>::seek_front_pixel_of_scanline()
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
        .def_readonly("histogram_buff", &StatsBase<T>::histogram)
        .def_property_readonly("histogram", [](StatsBase<T>& v){return v.get_histogram_py();});
}

template<typename T>
StatsBase<T>::StatsBase()
  : extrema(0, 0),
    max_bin(0),
    histogram(new std::vector<std::uint64_t>()),
    histogram_py(nullptr)
{
}

template<typename T>
py::object& StatsBase<T>::get_histogram_py()
{
    if(!histogram_py)
    {
        py::object buffer_obj = py::cast(histogram);
        histogram_py.reset(new py::object(PyArray_FromAny(buffer_obj.ptr(), nullptr, 1, 1, 0, nullptr), false), &safe_py_deleter);
    }
    return *histogram_py;
}

template<typename T>
void StatsBase<T>::set_bin_count(std::size_t bin_count)
{
    histogram->resize(bin_count, 0);
}

template<typename T>
void StatsBase<T>::find_max_bin()
{
    const std::uint64_t* h{histogram->data()};
    max_bin = std::max_element(h, h+histogram->size()) - h;
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
    this->extrema.second = this->extrema.first = std::nan("");
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
void ImageStats<T>::set_bin_count(std::size_t bin_count)
{
    Stats<T>::set_bin_count(bin_count);
    for(std::shared_ptr<Stats<T>>& channel_stat : channel_stats)
        channel_stat->set_bin_count(bin_count);
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
//        .def_property_readonly("data", [](NDImageStatistics<T>& v){return *v.data_py.get();})
        .def_readonly("range", &NDImageStatistics<T>::range)
        .def_readonly("mask", &NDImageStatistics<T>::mask)
        .def_property_readonly("image_stats", &NDImageStatistics<T>::get_image_stats)
        .def_readonly("drop_last_channel_from_overall_stats", &NDImageStatistics<T>::drop_last_channel_from_overall_stats);
    // Add overloaded "constructor" function.  pybind11 does not (yet, at time of writing) support templated class
    // instantiation via overloaded constructor defs, but plain function overloading is supported, and we take
    // advantage of this to present a factory function that is semantically similar.
    m.def("NDImageStatistics",
          [](typed_array_t<T>& a, const std::pair<T, T>& b, bool c){return new NDImageStatistics<T>(a, b, c);});
    m.def("NDImageStatistics",
          [](typed_array_t<T>& a, const std::pair<T, T>& b, typename CircularMask<T>::TupleArg m, bool c){return new NDImageStatistics<T>(a, b, m, c);});
    m.def("NDImageStatistics",
          [](typed_array_t<T>& a, const std::pair<T, T>& b, typed_array_t<std::uint8_t>& m, bool c){return new NDImageStatistics<T>(a, b, m, c);});
}

template<typename T>
NDImageStatistics<T>::NDImageStatistics(typed_array_t<T>& data_py,
                                        const std::pair<T, T>& range_,
                                        bool drop_last_channel_from_overall_stats_)
  : NDImageStatistics<T>(data_py,
                         range_,
                         std::make_shared<Mask<T>>(),
                         drop_last_channel_from_overall_stats_,
                         &NDImageStatistics<T>::compute<Mask<T>>)
{
}

template<typename T>
NDImageStatistics<T>::NDImageStatistics(typed_array_t<T>& data_py,
                                        const std::pair<T, T>& range_,
                                        typed_array_t<std::uint8_t>& mask_,
                                        bool drop_last_channel_from_overall_stats_)
  : NDImageStatistics<T>(data_py,
                         range_,
                         std::make_shared<BitmapMask<T>>(mask_),
                         drop_last_channel_from_overall_stats_,
                         &NDImageStatistics<T>::compute<BitmapMask<T>>)
{
}

template<typename T>
NDImageStatistics<T>::NDImageStatistics(typed_array_t<T>& data_py,
                                        const std::pair<T, T>& range_,
                                        typename CircularMask<T>::TupleArg mask_,
                                        bool drop_last_channel_from_overall_stats_)
  : NDImageStatistics<T>(data_py,
                         range_,
                         std::make_shared<CircularMask<T>>(mask_),
                         drop_last_channel_from_overall_stats_,
                         &NDImageStatistics<T>::compute<CircularMask<T>>)
{
}

template<typename T>
NDImageStatistics<T>::NDImageStatistics(typed_array_t<T>& data_py,
                                        const std::pair<T, T>& range_,
                                        std::shared_ptr<Mask<T>>&& mask_,
                                        bool drop_last_channel_from_overall_stats_,
                                        std::shared_ptr<ImageStats<T>>(*compute_fn_)(std::weak_ptr<NDImageStatistics<T>>))
  : data_view(new PyArrayView(data_py)),
    range(range_),
    mask(mask_),
    drop_last_channel_from_overall_stats(drop_last_channel_from_overall_stats_),
    compute_fn(compute_fn_)
{
    if(data_view->ndim < 2 || data_view->ndim > 3) throw std::invalid_argument("data argument must be 2 or 3 dimensional.");
    if(data_view->strides[0] > data_view->strides[1]) throw std::invalid_argument("data argument striding must be (X, Y) or (X, Y, C).");
}

template<typename T>
void NDImageStatistics<T>::launch_computation()
{
    image_stats = std::async(std::launch::async, std::bind(compute_fn, std::weak_ptr<NDImageStatistics<T>>(this->shared_from_this())));
}

template<typename T>
std::shared_ptr<ImageStats<T>> NDImageStatistics<T>::get_image_stats()
{
    return image_stats.get();
}

template<typename T>
template<typename MASK_T>
NDImageStatistics<T>::ComputeContext<MASK_T>::ComputeContext(std::shared_ptr<NDImageStatistics<T>>& ndis_sp, ImageStats<T>& stats_)
  : range(ndis_sp->range),
    data_view(ndis_sp->data_view),
    mask(std::dynamic_pointer_cast<MASK_T>(ndis_sp->mask)),
    drop_last_channel_from_overall_stats(ndis_sp->drop_last_channel_from_overall_stats),
    ndis_wp(ndis_sp),
    stats(stats_),
    bin_count(max_bin_count<T>())
{
}

template<typename T>
template<typename MASK_T>
std::shared_ptr<ImageStats<T>> NDImageStatistics<T>::compute(std::weak_ptr<NDImageStatistics<T>> this_wp)
{
    std::shared_ptr<ImageStats<T>> stats(new ImageStats<T>());
    std::shared_ptr<NDImageStatistics<T>> this_sp(this_wp.lock());
    // (bool)this_sp evaluates to false if the NDImageStatistics instance that asynchronously invoked compute(..) has
    // already been deleted. If/when it is deleted, we take this to mean that nobody is interested in the result of the
    // current invocation, so this is the mechanism (detecting that our invoking NDImageStatistics instance has been
    // deleted) is our early-out cue (later, we simply check if the weak pointer is expired rather than acquiring a
    // shared pointer to it via the lock method because checking if it is expired at that point is all we require).
    if(this_sp)
    {
        // Copy or get reference counted vars to all of the invoking NDImageStatistics instance's data that we will
        // need.
        ComputeContext<MASK_T> cc(this_sp, *stats);

        // We made copies of or reference-counted vars to all of the invoking NDImageStatistics instance's data that we
        // need. Now, we drop our reference to that NDImageStatistics instance. It is possible that ours became the last
        // reference to it; this would be OK and would simply indicate that some Python thread dropped the last existing
        // reference other than ours between the evaluation of the condition of the enclosing if statement and this line
        // of code. If that happened, we early-out before beginning processing of the first image scanline.
        this_sp.reset();

        // Dispatch to computation code path templated by image and paremeter (range) characteristics 
        dispatch_tagged_compute(cc);
    }
    return stats;
}
#include<iostream>
template<typename T>
template<typename MASK_T>
void NDImageStatistics<T>::dispatch_tagged_compute(ComputeContext<MASK_T>& cc, char (*)[std::is_integral<T>::value])
{
    if(std::is_unsigned<T>::value)
    {
        if(cc.range.first == std::numeric_limits<T>::min() && cc.range.second == std::numeric_limits<T>::max())
        {
            std::uint16_t bin_shift = sizeof(T)*8 - power_of_two(cc.bin_count).second;
            tagged_compute(cc, MaxRangeUnsignedIntegerComputeTag(bin_shift));
        }
        else
        {
            // NB: The above if statement handles the one case where range.second - range.first + 1 would overflow T
            T range_quanta_count = cc.range.second - cc.range.first; ++range_quanta_count;
            if(range_quanta_count < cc.bin_count)
                cc.bin_count = range_quanta_count;
            bool is_power2;
            std::int16_t power2;
            std::tie(is_power2, power2) = power_of_two(range_quanta_count);
            if(is_power2)
            {
                std::uint16_t bin_shift = power2 - power_of_two(cc.bin_count).second;
                tagged_compute(cc, Power2RangeUnsignedIntegerComputeTag(bin_shift));
            }
            else
            {
                tagged_compute(cc, UnsignedIntegerComputeTag());
            }
        }
    }
    else
    {
        // float is used to avoid overflowing in the case of broad range with signed value; the if statement need only
        // be triggered if this is small enough that adding one has an effect, in which case the conditional block will
        // also not overflow (given the relatively small bin counts we use)
        if((static_cast<float>(cc.range.second) - cc.range.first) + 1 < cc.bin_count)
            cc.bin_count = (cc.range.second - cc.range.first) + 1;
        tagged_compute(cc, IntegerComputeTag());
    }
}

template<typename T>
template<typename MASK_T>
void NDImageStatistics<T>::dispatch_tagged_compute(ComputeContext<MASK_T>& cc, char (*)[std::is_floating_point<T>::value])
{
    // TODO: if bin count exceeds floating point quanta in found or specified range, constrain bin count
    if(std::isnan(cc.range.first) || std::isnan(cc.range.second))
        tagged_compute(cc, UnrangedFloatComputeTag());
    else
        tagged_compute(cc, FloatComputeTag());
}

template<typename T>
template<typename MASK_T, typename COMPUTE_TAG>
void NDImageStatistics<T>::tagged_compute(ComputeContext<MASK_T>& cc, const COMPUTE_TAG& tag)
{
    cc.stats.channel_stats.resize(cc.data_view->ndim == 3 ? cc.data_view->shape[2] : 1);
    std::generate(cc.stats.channel_stats.begin(), cc.stats.channel_stats.end(), std::make_shared<Stats<T>>);
    init_extrema<MASK_T>(cc, tag);
    cc.stats.set_bin_count(cc.bin_count);
    scan_image<MASK_T>(cc, tag);
    gather_overall<MASK_T>(cc, tag);
}

template<typename T>
template<typename MASK_T>
void NDImageStatistics<T>::init_extrema(ComputeContext<MASK_T>& cc, const IntegerComputeTag& tag)
{
    Cursor<T, MASK_T> cursor(*cc.data_view, *cc.mask);
    std::shared_ptr<Stats<T>>* channel_stat_p;
    cursor.seek_front_scanline();
    if(cursor.scanline_valid)
    {
        cursor.seek_front_pixel_of_scanline();
        if(cursor.pixel_valid)
        {
            channel_stat_p = cc.stats.channel_stats.data();
            for(cursor.seek_front_component_of_pixel(); cursor.component_valid; cursor.advance_component(), ++channel_stat_p)
            {
                Stats<T>& channel_stat{**channel_stat_p};
                channel_stat.extrema.first = channel_stat.extrema.second = *cursor.component;
            }
        }
    }
}

template<typename T>
template<typename MASK_T>
void NDImageStatistics<T>::init_extrema(ComputeContext<MASK_T>& cc, const FloatComputeTag& tag)
{
    bool complete;
    Cursor<T, MASK_T> cursor(*cc.data_view, *cc.mask);
    std::shared_ptr<Stats<T>>* channel_stat_p;
    for(cursor.seek_front_scanline(); cursor.scanline_valid && !cc.ndis_wp.expired(); cursor.advance_scanline())
    {
        for(cursor.seek_front_pixel_of_scanline(); cursor.pixel_valid; cursor.advance_pixel())
        {
            channel_stat_p = cc.stats.channel_stats.data();
            complete = true;
            for(cursor.seek_front_component_of_pixel(); cursor.component_valid; cursor.advance_component(), ++channel_stat_p)
            {
                Stats<T>& channel_stat{**channel_stat_p};
                if(std::isnan(channel_stat.extrema.first))
                {
                    if(is_finite(*cursor.component))
                        channel_stat.extrema.first = channel_stat.extrema.second = *cursor.component;
                    else
                        complete = false;
                }
            }
            if(complete)
                return;
        }
    }
}

template<typename T>
template<typename MASK_T>
void NDImageStatistics<T>::init_extrema(ComputeContext<MASK_T>& cc, const UnrangedFloatComputeTag& tag)
{
    Cursor<T, MASK_T> cursor(*cc.data_view, *cc.mask);
    std::shared_ptr<Stats<T>>* channel_stat_p;
    for(cursor.seek_front_scanline(); cursor.scanline_valid && !cc.ndis_wp.expired(); cursor.advance_scanline())
    {
        for(cursor.seek_front_pixel_of_scanline(); cursor.pixel_valid; cursor.advance_pixel())
        {
            channel_stat_p = cc.stats.channel_stats.data();
            for(cursor.seek_front_component_of_pixel(); cursor.component_valid; cursor.advance_component(), ++channel_stat_p)
            {
                Stats<T>& channel_stat{**channel_stat_p};
                if(!std::isnan(channel_stat.extrema.first))
                {
                    if(*cursor.component < channel_stat.extrema.first)
                        channel_stat.extrema.first = *cursor.component;
                    else if(*cursor.component > channel_stat.extrema.second)
                        channel_stat.extrema.second = *cursor.component;
                }
                else
                {
                    if(is_finite(*cursor.component))
                        channel_stat.extrema.first = channel_stat.extrema.second = *cursor.component;
                }
            }
        }
    }
    // NB: For the purpose of finding range, drop_last_channel_from_overall_stats is ignored
    for(std::shared_ptr<Stats<T>>& channel_stat : cc.stats.channel_stats)
    {
        if(std::isnan(cc.range.first) || channel_stat->extrema.first < cc.range.first)
            cc.range.first = channel_stat->extrema.first;
        if(std::isnan(cc.range.second) || channel_stat->extrema.second > cc.range.second)
            cc.range.second = channel_stat->extrema.second;
    }
}

template<typename T>
template<typename MASK_T, typename COMPUTE_TAG>
void NDImageStatistics<T>::scan_image(ComputeContext<MASK_T>& cc, const COMPUTE_TAG& tag)
{
    Cursor<T, MASK_T> cursor(*cc.data_view, *cc.mask);
    std::shared_ptr<Stats<T>>* channel_stat_p;
    for(cursor.seek_front_scanline(); cursor.scanline_valid && !cc.ndis_wp.expired(); cursor.advance_scanline())
    {
        for(cursor.seek_front_pixel_of_scanline(); cursor.pixel_valid; cursor.advance_pixel())
        {
            channel_stat_p = cc.stats.channel_stats.data();
            for(cursor.seek_front_component_of_pixel(); cursor.component_valid; cursor.advance_component(), ++channel_stat_p)
            {
                process_component<MASK_T>(cc, **channel_stat_p, *cursor.component, tag);
            }
        }
    }
    for(std::shared_ptr<Stats<T>>& channel_stat : cc.stats.channel_stats)
        channel_stat->find_max_bin();
}

template<typename T>
template<typename MASK_T>
void NDImageStatistics<T>::process_component(ComputeContext<MASK_T>& cc,
                                             Stats<T>& component_stats,
                                             const T& component,
                                             const ComputeTag& tag)
{
}

template<typename T>
template<typename MASK_T>
void NDImageStatistics<T>::process_component(ComputeContext<MASK_T>& cc,
                                             Stats<T>& component_stats,
                                             const T& component,
                                             const MaxRangeUnsignedIntegerComputeTag& tag)
{
    std::uint16_t bin = component >> tag.bin_shift;
    ++(*component_stats.histogram)[bin];
    if(component < component_stats.extrema.first)
        component_stats.extrema.first = component;
    else if(component > component_stats.extrema.second)
        component_stats.extrema.second = component;
}

template<typename T>
template<typename MASK_T>
void NDImageStatistics<T>::process_component(ComputeContext<MASK_T>& cc,
                                             Stats<T>& component_stats,
                                             const T& component,
                                             const FloatComputeTag& tag)
{
    switch(std::fpclassify(component))
    {
    case FP_INFINITE:
        if(component == -INFINITY)
            ++component_stats.neg_inf_count;
        else
            ++component_stats.pos_inf_count;
        break;
    case FP_NAN:
        ++component_stats.NaN_count;
        break;
    default:
        // It would be more efficient to have a NonFiniteOnlyFloatComputeTag and specialization in order to factor out 
        // this if statement; doing so may improve float image throughput somewhat.
        if(!std::isnan(cc.range.first))
            process_component(cc, component_stats, component, ComputeTag());
        break;
    }
}

template<typename T>
template<typename MASK_T>
void NDImageStatistics<T>::gather_overall(ComputeContext<MASK_T>& cc, const IntegerComputeTag&)
{
    const std::ptrdiff_t last_overall_component_idx{(std::ptrdiff_t)(cc.data_view->ndim == 3 ? cc.data_view->shape[2] : 1) - 1 - int(cc.drop_last_channel_from_overall_stats)};
    std::ptrdiff_t component_idx;
    if(last_overall_component_idx == 0)
    {
        cc.stats.max_bin = cc.stats.channel_stats[0]->max_bin;
        cc.stats.extrema = cc.stats.channel_stats[0]->extrema;
        cc.stats.histogram = cc.stats.channel_stats[0]->histogram;
    }
    else
    {
        cc.stats.extrema = cc.stats.channel_stats[0]->extrema;
        *cc.stats.histogram = *cc.stats.channel_stats[0]->histogram;

        std::uint64_t *mhit, *chit, *mhit_end;
        for(std::ptrdiff_t component_idx=1; component_idx <= last_overall_component_idx; ++component_idx)
        {
            if(cc.stats.channel_stats[component_idx]->extrema.first < cc.stats.extrema.first)
                cc.stats.extrema.first = cc.stats.channel_stats[component_idx]->extrema.first;
            if(cc.stats.channel_stats[component_idx]->extrema.second > cc.stats.extrema.second)
                cc.stats.extrema.second = cc.stats.channel_stats[component_idx]->extrema.second;
            mhit = cc.stats.histogram->data();
            mhit_end = mhit + cc.bin_count;
            chit = cc.stats.channel_stats[component_idx]->histogram->data();
            for(; mhit != mhit_end; ++mhit, ++chit)
                *mhit += *chit;
        }

        cc.stats.find_max_bin();
    }
}

template<typename T>
template<typename MASK_T>
void NDImageStatistics<T>::gather_overall(ComputeContext<MASK_T>& cc, const FloatComputeTag&)
{
    const std::ptrdiff_t last_overall_component_idx{(std::ptrdiff_t)(cc.data_view->ndim == 3 ? cc.data_view->shape[2] : 1) - 1 - int(cc.drop_last_channel_from_overall_stats)};
    std::ptrdiff_t component_idx;
    if(last_overall_component_idx == 0)
    {
        cc.stats.max_bin = cc.stats.channel_stats[0]->max_bin;
        cc.stats.extrema = cc.stats.channel_stats[0]->extrema;
        cc.stats.histogram = cc.stats.channel_stats[0]->histogram;
        cc.stats.NaN_count = cc.stats.channel_stats[0]->NaN_count;
        cc.stats.neg_inf_count = cc.stats.channel_stats[0]->neg_inf_count;
        cc.stats.pos_inf_count = cc.stats.channel_stats[0]->pos_inf_count;
    }
    else
    {
        cc.stats.extrema = cc.stats.channel_stats[0]->extrema;
        *cc.stats.histogram = *cc.stats.channel_stats[0]->histogram;
        cc.stats.NaN_count = cc.stats.channel_stats[0]->NaN_count;
        cc.stats.neg_inf_count = cc.stats.channel_stats[0]->neg_inf_count;
        cc.stats.pos_inf_count = cc.stats.channel_stats[0]->pos_inf_count;

        std::uint64_t *mhit, *chit, *mhit_end;
        for(std::ptrdiff_t component_idx=1; component_idx <= last_overall_component_idx; ++component_idx)
        {
            if(std::isnan(cc.stats.extrema.first) || cc.stats.channel_stats[component_idx]->extrema.first < cc.stats.extrema.first)
                cc.stats.extrema.first = cc.stats.channel_stats[component_idx]->extrema.first;
            if(std::isnan(cc.stats.extrema.second) || cc.stats.channel_stats[component_idx]->extrema.second > cc.stats.extrema.second)
                cc.stats.extrema.second = cc.stats.channel_stats[component_idx]->extrema.second;
            mhit = cc.stats.histogram->data();
            mhit_end = mhit + cc.bin_count;
            chit = cc.stats.channel_stats[component_idx]->histogram->data();
            for(; mhit != mhit_end; ++mhit, ++chit)
                *mhit += *chit;
            cc.stats.NaN_count += cc.stats.channel_stats[component_idx]->NaN_count;
            cc.stats.neg_inf_count += cc.stats.channel_stats[component_idx]->neg_inf_count;
            cc.stats.pos_inf_count += cc.stats.channel_stats[component_idx]->pos_inf_count;
        }

        cc.stats.find_max_bin();
    }
}
