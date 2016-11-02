// The MIT License (MIT)
//
// Copyright (c) 2015-2016 WUSTL ZPLAB
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
#include "common.h"
#include "cursors.h"
#include "masks.h"
#include "statses.h"

template<typename T>
class NDImageStatistics
  : public std::enable_shared_from_this<NDImageStatistics<T>>
{
    friend struct ImageStats<T>;
public:
    static void expose_via_pybind11(py::module& m, const std::string& s);

    // No mask
    NDImageStatistics(typed_array_t<T>& data_py,
                      const std::pair<T, T>& range_,
                      bool drop_last_channel_from_overall_stats_);
    // Bitmap mask
    NDImageStatistics(typed_array_t<T>& data_py,
                      const std::pair<T, T>& range_,
                      typed_array_t<std::uint8_t>& mask_,
                      bool drop_last_channel_from_overall_stats_);
    // Circular mask
    NDImageStatistics(typed_array_t<T>& data_py,
                      const std::pair<T, T>& range_,
                      typename CircularMask<T>::TupleArg mask_,
                      bool drop_last_channel_from_overall_stats_);
    // Not constructable with no parameters
    NDImageStatistics() = delete;
    // Not copyable via constructor
    NDImageStatistics(const NDImageStatistics&) = delete;
    // Not copyable via assignment
    NDImageStatistics& operator = (const NDImageStatistics&) = delete;

    void launch_computation();
    std::shared_ptr<ImageStats<T>> get_image_stats();

protected:
    std::shared_ptr<PyArrayView> data_view;
    std::pair<T, T> range;
    std::shared_ptr<Mask<T>> mask;
    std::shared_future<std::shared_ptr<ImageStats<T>>> image_stats;
    const bool drop_last_channel_from_overall_stats;
    std::shared_ptr<ImageStats<T>>(*compute_fn)(std::weak_ptr<NDImageStatistics<T>>);

    NDImageStatistics(typed_array_t<T>& data_py,
                      const std::pair<T, T>& range_,
                      std::shared_ptr<Mask<T>>&& mask_,
                      bool drop_last_channel_from_overall_stats_,
                      std::shared_ptr<ImageStats<T>>(*compute_fn_)(std::weak_ptr<NDImageStatistics<T>>));

    template<BitmapMaskDimensionVsImage T_W, BitmapMaskDimensionVsImage T_H>
    void bitmap_mask_constructor_helper(std::unique_ptr<PyArrayView>&& mask_view);

    template<typename MASK_T>
    struct ComputeContext
    {
        ComputeContext(std::shared_ptr<NDImageStatistics<T>>& ndis_sp, ImageStats<T>& stats_);

        std::pair<T, T> range;
        std::shared_ptr<PyArrayView> data_view;
        std::shared_ptr<MASK_T> mask;
        bool drop_last_channel_from_overall_stats;
        std::weak_ptr<NDImageStatistics<T>> ndis_wp;
        ImageStats<T>& stats;
        std::uint16_t bin_count;
    };

    template<typename MASK_T>
    static std::shared_ptr<ImageStats<T>> compute(std::weak_ptr<NDImageStatistics<T>> this_wp);

    struct ComputeTag {};
    struct FloatComputeTag : ComputeTag {};
    struct UnrangedFloatComputeTag : FloatComputeTag {};
    struct IntegerComputeTag : ComputeTag {};
    struct UnsignedIntegerComputeTag : IntegerComputeTag {};
    struct Power2RangeUnsignedIntegerComputeTag : UnsignedIntegerComputeTag {
        std::uint16_t bin_shift;
        explicit Power2RangeUnsignedIntegerComputeTag(std::uint16_t bin_shift_) : bin_shift(bin_shift_) {}
    };
    struct MaxRangeUnsignedIntegerComputeTag : Power2RangeUnsignedIntegerComputeTag {
        using Power2RangeUnsignedIntegerComputeTag::Power2RangeUnsignedIntegerComputeTag;
    };

    // Dispatch for integral T. The second parameter is a character array with length of 0 or 1, depending on the value
    // of std::is_integral<T>::value. A zero length array is invalid in C++. Therefore, this definition is valid C++
    // only when T is integer, causing the method to be hidden entirely when T is floating point, allowing the integral
    // and floating point versions to make calls that are only valid for their respective types. The specific C++
    // facility exploited by this trick is known as SFINAE.
    template<typename MASK_T, bool IS_INTEGRAL=std::is_integral<T>::value>
    static void dispatch_tagged_compute(ComputeContext<MASK_T>& cc, char (*)[IS_INTEGRAL]=0);

    // Dispatch for floating point T.
    template<typename MASK_T, bool IS_INTEGRAL=std::is_integral<T>::value>
    static void dispatch_tagged_compute(ComputeContext<MASK_T>& cc, char (*)[!IS_INTEGRAL]=0);

    template<typename MASK_T, typename COMPUTE_TAG>
    static void tagged_compute(ComputeContext<MASK_T>& cc, const COMPUTE_TAG& tag);

    // Sets (min, max) to (largest for type T, smallest and possibly negative value for type T)
    template<typename MASK_T>
    static void init_extrema(ComputeContext<MASK_T>& cc, const ComputeTag&);

    // Finds extrema across entire image in order to establish range
    template<typename MASK_T>
    static void init_extrema(ComputeContext<MASK_T>& cc, const UnrangedFloatComputeTag&);

    template<typename MASK_T, typename COMPUTE_TAG>
    static void scan_image(ComputeContext<MASK_T>& cc, const COMPUTE_TAG& tag);

    template<typename MASK_T>
    static inline void process_component(ComputeContext<MASK_T>& cc,
                                         Stats<T>& component_stats,
                                         const T& component,
                                         const ComputeTag& tag);

    template<typename MASK_T>
    static inline void process_component(ComputeContext<MASK_T>& cc,
                                         Stats<T>& component_stats,
                                         const T& component,
                                         const MaxRangeUnsignedIntegerComputeTag& tag);

    template<typename MASK_T>
    static inline void process_component(ComputeContext<MASK_T>& cc,
                                         Stats<T>& component_stats,
                                         const T& component,
                                         const FloatComputeTag&);

    template<typename MASK_T>
    static void gather_overall(ComputeContext<MASK_T>& cc, const IntegerComputeTag&);

    template<typename MASK_T>
    static void gather_overall(ComputeContext<MASK_T>& cc, const FloatComputeTag&);
};

#include "NDImageStatistics_impl.h"