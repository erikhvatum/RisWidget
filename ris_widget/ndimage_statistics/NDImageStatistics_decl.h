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

// Copies u_shape to o_shape and u_strides to o_strides, reversing the elements of each if u_strides[0] < u_strides[1]
void reorder_to_inner_outer(const std::size_t* u_shape, const std::size_t* u_strides,
                            std::size_t* o_shape,       std::size_t* o_strides);

// Copies u_shape to o_shape and u_strides to o_strides, reversing the elements of each if u_strides[0] < u_strides[1].
// Additionally, u_slave_shape is copied to o_slave_shape and u_slave_strides is copied to o_slave_strides, reversing
// the elements of each if u_strides[0] < u_strides[1].
void reorder_to_inner_outer(const std::size_t* u_shape,       const std::size_t* u_strides,
                            std::size_t* o_shape,             std::size_t* o_strides,
                            const std::size_t* u_slave_shape, const std::size_t* u_slave_strides,
                            std::size_t* o_slave_shape,       std::size_t* o_slave_strides);

void reorder_to_inner_outer(const std::size_t* u_shape, const std::size_t* u_strides,
                            std::size_t* o_shape,       std::size_t* o_strides,
                            const float& u_coord_0,     const float& u_coord_1,
                            float& o_coord_0,           float& o_coord_1);

template<typename T>
std::size_t bin_count();

template<>
std::size_t bin_count<std::uint8_t>();

template<>
std::size_t bin_count<std::int8_t>();

template<typename T>
struct Stats
  : public std::enable_shared_from_this<Stats<T>>
{
    Stats();
    py::array data;
    std::tuple<T, T> extrema;
    py::array histogram_py_array;
    std::uint32_t* histogram;
};

template<typename T>
struct ImageStats
  : Stats<T>
{
    ImageStats(typed_array_t<T>& data_, const std::vector<std::uint8_t>& drop_channel_idxs);
    std::vector<std::shared_ptr<Stats<T>>> channels;
};

//template<typename T, typename MASK, std::uint8_t CHANNEL_COUNT>
//struct StatComputer
//  : Stats<T, CHANNEL_COUNT>
//{
//    StatComputer(const StatComputer&) = delete;
//    StatComputer& operator = (const StatComputer&) = delete;
//
//    std::atomic_bool m_cancelled;
//};

template<typename T, typename MASK>
class NDImageStatistics
  : public std::enable_shared_from_this<NDImageStatistics<T, MASK>>
{
public:
    static void expose_via_pybind11(py::module& m, const std::string& s);

    explicit NDImageStatistics(typed_array_t<T>& data_, const std::vector<std::uint8_t>& drop_channel_idxs={});
    NDImageStatistics(const NDImageStatistics&) = delete;
    NDImageStatistics& operator = (const NDImageStatistics&) = delete;
    virtual ~NDImageStatistics();

protected:
    typed_array_t<T> data;
    ImageStats<T> stats;
};