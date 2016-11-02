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
void NDImageStatistics<T>::expose_via_pybind11(py::module& m, const std::string& s)
{
    Mask<T>::expose_via_pybind11(m);
    BitmapMask<T, BitmapMaskDimensionVsImage::Smaller, BitmapMaskDimensionVsImage::Smaller>::expose_via_pybind11(m);
    BitmapMask<T, BitmapMaskDimensionVsImage::Smaller, BitmapMaskDimensionVsImage::Same>::expose_via_pybind11(m);
    BitmapMask<T, BitmapMaskDimensionVsImage::Smaller, BitmapMaskDimensionVsImage::Larger>::expose_via_pybind11(m);
    BitmapMask<T, BitmapMaskDimensionVsImage::Same, BitmapMaskDimensionVsImage::Smaller>::expose_via_pybind11(m);
    BitmapMask<T, BitmapMaskDimensionVsImage::Same, BitmapMaskDimensionVsImage::Same>::expose_via_pybind11(m);
    BitmapMask<T, BitmapMaskDimensionVsImage::Same, BitmapMaskDimensionVsImage::Larger>::expose_via_pybind11(m);
    BitmapMask<T, BitmapMaskDimensionVsImage::Larger, BitmapMaskDimensionVsImage::Smaller>::expose_via_pybind11(m);
    BitmapMask<T, BitmapMaskDimensionVsImage::Larger, BitmapMaskDimensionVsImage::Same>::expose_via_pybind11(m);
    BitmapMask<T, BitmapMaskDimensionVsImage::Larger, BitmapMaskDimensionVsImage::Larger>::expose_via_pybind11(m);
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
          [](typed_array_t<T>& a, const std::pair<T, T>& b, bool c){return new NDImageStatistics<T>(a, b, c);},
          py::arg("data"), py::arg("range_"), py::arg("drop_last_channel_from_overall_stats"));
    m.def("NDImageStatistics",
          [](typed_array_t<T>& a, const std::pair<T, T>& b, typename CircularMask<T>::TupleArg m, bool c){return new NDImageStatistics<T>(a, b, m, c);},
          py::arg("data"), py::arg("range_"), py::arg("roi_mask_tuple"), py::arg("drop_last_channel_from_overall_stats"));
    m.def("NDImageStatistics",
          [](typed_array_t<T>& a, const std::pair<T, T>& b, typed_array_t<std::uint8_t>& m, bool c){return new NDImageStatistics<T>(a, b, m, c);},
          py::arg("data"), py::arg("range_"), py::arg("bitmap_mask_data"), py::arg("drop_last_channel_from_overall_stats"));
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
template<BitmapMaskDimensionVsImage T_W, BitmapMaskDimensionVsImage T_H>
void NDImageStatistics<T>::bitmap_mask_constructor_helper(std::unique_ptr<PyArrayView>&& mask_view)
{
    using BM_T = BitmapMask<T, T_W, T_H>;
    compute_fn = &NDImageStatistics<T>::compute<BM_T>;
    mask = std::make_shared<BM_T>(std::move(mask_view));
}

template<typename T>
NDImageStatistics<T>::NDImageStatistics(typed_array_t<T>& data_py,
                                        const std::pair<T, T>& range_,
                                        typed_array_t<std::uint8_t>& mask_,
                                        bool drop_last_channel_from_overall_stats_)
  : data_view(new PyArrayView(data_py)),
    range(range_),
    drop_last_channel_from_overall_stats(drop_last_channel_from_overall_stats_)
{
    if(data_view->ndim < 2 || data_view->ndim > 3) throw std::invalid_argument("data argument must be 2 or 3 dimensional.");
    if(data_view->strides[0] > data_view->strides[1]) throw std::invalid_argument("data argument striding must be (X, Y) or (X, Y, C).");

    std::unique_ptr<PyArrayView> mask_view{new PyArrayView(mask_)};
    if(mask_view->ndim != 2) throw std::invalid_argument("bitmap mask must be 2 dimensional.");
    if(mask_view->strides[0] > mask_view->strides[1]) throw std::invalid_argument("bitmap mask striding must be (X, Y).");

    if(mask_view->shape[0] < data_view->shape[0])
    {
        if(mask_view->shape[1] < data_view->shape[1])
            bitmap_mask_constructor_helper<BitmapMaskDimensionVsImage::Smaller, BitmapMaskDimensionVsImage::Smaller>(std::move(mask_view));
        else if(mask_view->shape[1] == data_view->shape[1])
            bitmap_mask_constructor_helper<BitmapMaskDimensionVsImage::Smaller, BitmapMaskDimensionVsImage::Same>(std::move(mask_view));
        else
            bitmap_mask_constructor_helper<BitmapMaskDimensionVsImage::Smaller, BitmapMaskDimensionVsImage::Larger>(std::move(mask_view));
    }
    else if(mask_view->shape[0] == data_view->shape[0])
    {
        if(mask_view->shape[1] < data_view->shape[1])
            bitmap_mask_constructor_helper<BitmapMaskDimensionVsImage::Same, BitmapMaskDimensionVsImage::Smaller>(std::move(mask_view));
        else if(mask_view->shape[1] == data_view->shape[1])
            bitmap_mask_constructor_helper<BitmapMaskDimensionVsImage::Same, BitmapMaskDimensionVsImage::Same>(std::move(mask_view));
        else
            bitmap_mask_constructor_helper<BitmapMaskDimensionVsImage::Same, BitmapMaskDimensionVsImage::Larger>(std::move(mask_view));
    }
    else
    {
        if(mask_view->shape[1] < data_view->shape[1])
            bitmap_mask_constructor_helper<BitmapMaskDimensionVsImage::Larger, BitmapMaskDimensionVsImage::Smaller>(std::move(mask_view));
        else if(mask_view->shape[1] == data_view->shape[1])
            bitmap_mask_constructor_helper<BitmapMaskDimensionVsImage::Larger, BitmapMaskDimensionVsImage::Same>(std::move(mask_view));
        else
            bitmap_mask_constructor_helper<BitmapMaskDimensionVsImage::Larger, BitmapMaskDimensionVsImage::Larger>(std::move(mask_view));
    }
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

template<typename T>
template<typename MASK_T, bool IS_INTEGRAL>
void NDImageStatistics<T>::dispatch_tagged_compute(ComputeContext<MASK_T>& cc, char (*)[IS_INTEGRAL])
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
template<typename MASK_T, bool IS_INTEGRAL>
void NDImageStatistics<T>::dispatch_tagged_compute(ComputeContext<MASK_T>& cc, char (*)[!IS_INTEGRAL])
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
void NDImageStatistics<T>::init_extrema(ComputeContext<MASK_T>& cc, const ComputeTag& tag)
{
    std::shared_ptr<Stats<T>>* channel_stat_p{cc.stats.channel_stats.data()};
    std::shared_ptr<Stats<T>>*const channel_stat_p_end{channel_stat_p + cc.stats.channel_stats.size()};
    for(; channel_stat_p != channel_stat_p_end; ++channel_stat_p)
    {
        std::pair<T, T>& extrema = (*channel_stat_p)->extrema;
        extrema.first = std::numeric_limits<T>::max();
        extrema.second = std::numeric_limits<T>::lowest();
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
//              std::cout << "*process_component\n";
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
    const T range_width = cc.range.second - cc.range.first;
    if(range_width == 0)
        // TODO: bail out in caller rather than here
        return;
    // TODO: cache this
    const double bin_factor = static_cast<double>(cc.bin_count) / range_width;
    if(component >= cc.range.first && component < cc.range.second)
        ++component_stats.histogram->data()[static_cast<std::ptrdiff_t>(bin_factor * (component - cc.range.first))];
    else if(component == cc.range.second)
        ++component_stats.histogram->back();
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
