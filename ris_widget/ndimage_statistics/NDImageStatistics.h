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
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _ndimage_statistics_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/npy_common.h>
#include <numpy/arrayobject.h>
#define _USE_MATH_DEFINES
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>
#include "Luts.h"
#include "PyArrayView.h"

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

extern std::unordered_map<std::type_index, std::string> component_type_names;

template<typename T>
std::size_t bin_count();

template<>
std::size_t bin_count<std::uint8_t>();

template<>
std::size_t bin_count<std::int8_t>();

// GIL-aware deleter for Python objects likely to be released and refcount-decremented on another (non-Python) thread
void safe_py_deleter(py::object* py_obj);

// SimpleBufferInfo is like py::buffer_info, differing in that SimpleBufferInfo does not retain keep a Python reference 
// to the buffer whose info is represented. with only the information we need and without 
struct SimpleBufferInfo
{
    std::size_t ndim;

};

template<typename T>
struct Mask;

template<typename T>
struct CursorBase
{
    explicit CursorBase(PyArrayView& data_view);
    CursorBase(const CursorBase&) = delete;
    CursorBase& operator = (const CursorBase&) = delete;
    virtual ~CursorBase() = default;

    virtual void seek_to_front();

    bool pixel_valid, component_valid;

    const std::size_t scanline_count;
    const std::size_t scanline_stride;
    const std::uint8_t* scanline_raw;
    const std::uint8_t*const scanlines_raw_end;

    const std::size_t scanline_width;
    const std::size_t pixel_stride;
    const std::uint8_t* pixel_raw;
    const std::uint8_t* pixels_raw_end;

    const std::size_t component_count;
    const std::size_t component_stride;
    const std::uint8_t* component_raw;
    const std::uint8_t* components_raw_end;
    const T*& component;
};

template<typename T>
struct NonPerComponentMaskCursor
  : CursorBase<T>
{
    using CursorBase<T>::CursorBase;

    void advance_component();
};

template<typename T>
struct Mask
{
    static void expose_via_pybind11(py::module& m);

    Mask() = default;
    Mask(const Mask&) = delete;
    Mask& operator = (const Mask&) = delete;
    virtual ~Mask() = default;
};

template<typename T, typename MASK_T>
struct Cursor
  : NonPerComponentMaskCursor<T>
{
    Cursor(PyArrayView& data_view, MASK_T& mask_);

    void advance_pixel();
};

template<typename T>
struct BitmapMask
  : Mask<T>
{
    static void expose_via_pybind11(py::module& m);

    explicit BitmapMask(typed_array_t<std::uint8_t>& bitmap_py_);

    PyArrayView bitmap_view;
};

template<typename T>
struct Cursor<T, BitmapMask<T>>
  : NonPerComponentMaskCursor<T>
{
    Cursor(PyArrayView& data_view, BitmapMask<T>& mask_);

    void advance_pixel();

    BitmapMask<T>& mask;
};

template<typename T>
struct CircularMask
  : Mask<T>
{
    using TupleArg = const std::tuple<std::tuple<double, double>, double>&;

    static void expose_via_pybind11(py::module& m);

    CircularMask(double center_x_, double center_y_, double radius_);
    explicit CircularMask(TupleArg t);

    double center_x, center_y, radius;
};

template<typename T>
struct Cursor<T, CircularMask<T>>
  : NonPerComponentMaskCursor<T>
{
    Cursor(PyArrayView& data_view, CircularMask<T>& mask_);

    void advance_pixel();

    CircularMask<T>& mask;
};

template<typename T>
struct StatsBase
{
    static void expose_via_pybind11(py::module& m);

    StatsBase();
    StatsBase(const StatsBase&) = delete;
    StatsBase& operator = (const StatsBase&) = delete;
    virtual ~StatsBase() = default;

    std::tuple<T, T> extrema;
    std::size_t max_bin;

    std::shared_ptr<std::vector<std::uint64_t>> histogram;
    // A numpy array that is a read-only view of histogram. Lazily created in response to get_histogram_py calls.
    std::shared_ptr<py::object> histogram_py;

    py::object& get_histogram_py();
};

template<typename T>
struct Stats
  : StatsBase<T>
{
    static void expose_via_pybind11(py::module& m);
};

template<typename T>
struct FloatStatsBase
  : StatsBase<T>
{
    static void expose_via_pybind11(py::module& m);

    FloatStatsBase();

    std::uint64_t NaN_count, neg_inf_count, pos_inf_count;
};

template<>
struct Stats<float>
  : FloatStatsBase<float>
{
    static void expose_via_pybind11(py::module& m);
};

template<>
struct Stats<double>
  : FloatStatsBase<double>
{
    static void expose_via_pybind11(py::module& m);
};

template<typename T>
class NDImageStatistics;

// This is neat: we only need to provide specializations for Stats; ImageStats automatically inherits the correct 
// specialization and therefore gets the extra overall fields (NaN count, neg_inf_count, pos_inf_count) without futher 
// ado.
template<typename T>
struct ImageStats
  : std::enable_shared_from_this<ImageStats<T>>,
    Stats<T>
{
    static void expose_via_pybind11(py::module& m);
    std::vector<std::shared_ptr<Stats<T>>> channel_stats;
};

template<typename T>
class NDImageStatistics
  : public std::enable_shared_from_this<NDImageStatistics<T>>
{
    friend class ImageStats<T>;
public:
    static void expose_via_pybind11(py::module& m, const std::string& s);

    // No mask
    NDImageStatistics(typed_array_t<T>& data_py_,
                      const std::pair<T, T>& range_,
                      bool drop_last_channel_from_overall_stats_);
    // Bitmap mask
    NDImageStatistics(typed_array_t<T>& data_py_,
                      const std::pair<T, T>& range_,
                      typed_array_t<std::uint8_t>& mask_,
                      bool drop_last_channel_from_overall_stats_);
    // Circular mask
    NDImageStatistics(typed_array_t<T>& data_py_,
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

    NDImageStatistics(typed_array_t<T>& data_py_,
                      const std::pair<T, T>& range_,
                      std::shared_ptr<Mask<T>>&& mask_,
                      bool drop_last_channel_from_overall_stats_,
                      std::shared_ptr<ImageStats<T>>(*compute_fn_)(std::weak_ptr<NDImageStatistics<T>>));

    template<typename MASK_T>
    static std::shared_ptr<ImageStats<T>> compute(std::weak_ptr<NDImageStatistics<T>> this_wp);
};

#include "NDImageStatistics_impl.h"