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
#include <Python.h>
#include <numpy/npy_common.h>
#define _USE_MATH_DEFINES
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "Luts.h"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(_T_, std::shared_ptr<_T_>);

// The second template parameter of py::array_t<..> is ExtraFlags, an enum value that defaults to py::array::forecast.
// py::array::forecast causes array typecasting/conversion in order to satisfy an overloaded c++ function argument when
// we instead want the overload not requiring typecasting to be called - or a ValueError exception if there is no
// matching overload. Specifying 0 for this argument seems to work, even for zp-strided RGB images, although
// the enum definition suggests 0 is actually equivalent to c_contig.
template<typename T> using typed_array_t = py::array_t<T, 0>;
using mask_tuple_t = std::tuple<std::tuple<double, double>, double>;

extern Luts luts;

template<typename T>
std::size_t bin_count();

template<>
std::size_t bin_count<std::uint8_t>();

template<>
std::size_t bin_count<std::int8_t>();

struct Mask
{
    virtual ~Mask() = default;
};

struct BitmapMask
  : Mask
{
    explicit BitmapMask(typed_array_t<std::uint8_t>& bitmap_);

    typed_array_t<std::uint8_t> bitmap;
    py::buffer_info bitmap_bi;
};

struct CircularMask
  : Mask
{
    CircularMask(double center_x_, double center_y_, double radius_);

    double center_x, center_y, radius;
};

template<typename T>
struct StatsBase
{
    StatsBase();
    StatsBase(const StatsBase&) = delete;
    StatsBase& operator = (const StatsBase&) = delete;
    virtual ~StatsBase();

    std::tuple<T, T> extrema;
    typed_array_t<std::uint32_t>* histogram_py;
    std::uint32_t* histogram;
};

template<typename T>
struct Stats
  : StatsBase<T>
{
};

template<typename T>
struct FloatStatsBase
  : StatsBase<T>
{
    std::uint32_t NaN_count=0, neg_inf_count=0, pos_inf_count=0;
};

template<>
struct Stats<float>
  : FloatStatsBase<float>
{
};

template<>
struct Stats<double>
  : FloatStatsBase<double>
{
};

// This is neat: we only need to provide specializations for Stats; ImageStats automatically inherits the correct 
// specialization and therefore gets the extra overall fields (NaN count, neg_inf_count, pos_inf_count) without futher 
// ado.
template<typename T>
struct ImageStats
  : Stats<T>
{
    ImageStats();
    ImageStats(const ImageStats&) = delete;
    ImageStats& operator = (const ImageStats&) = delete;

    std::vector<std::shared_ptr<Stats<T>>> channel_stats;
};

// template<typename T, typename MASK_T>
// struct StatComputer
// {
//     StatComputer(const StatComputer&) = delete;
//     StatComputer& operator = (const StatComputer&) = delete;
//     virtual ~StatComputer() = default;
// 
//     std::atomic_bool m_cancelled;
// };

template<typename T>
class NDImageStatistics
{
public:
    static void expose_via_pybind11(py::module& m, const std::string& s);

    // No mask
    NDImageStatistics(typed_array_t<T>& data_py_,
                      bool drop_last_channel_from_overall_stats);
    // Bitmap mask
    NDImageStatistics(typed_array_t<T>& data_py_,
                      typed_array_t<std::uint8_t>& mask_,
                      bool drop_last_channel_from_overall_stats);
    // Circular mask
    NDImageStatistics(typed_array_t<T>& data_py_,
                      const std::tuple<std::tuple<double, double>, double>& circular_mask_parameters,
                      bool drop_last_channel_from_overall_stats);
    // Not copyable via constructor
    NDImageStatistics(const NDImageStatistics&) = delete;
    // Not copyable via assignment
    NDImageStatistics& operator = (const NDImageStatistics&) = delete;

    ~NDImageStatistics();

protected:
    // A shared pointer with a GIL-aware deleter is kept to avoid the need to acquire the gil whenever the C++ reference 
    // count to the array changes. In other words, we keep our own fast, atomic reference count in the form of a 
    // shared_ptr and only bother acquiring the GIL and decrementing the Python reference count when ours has dropped to 
    // zero. 
    std::shared_ptr<typed_array_t<T>> data_py;
//  const T* const data;
    std::shared_ptr<const Mask> mask;
    ImageStats<T> stats;

    NDImageStatistics(typed_array_t<T>& data_py_,
                      std::shared_ptr<const Mask>&& mask_,
                      bool drop_last_channel_from_overall_stats);
};

#include "NDImageStatistics_impl.h"