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
#include "common.h"

template<typename T>
struct StatsBase
{
    static void expose_via_pybind11(py::module& m);

    StatsBase();
    StatsBase(const StatsBase&) = delete;
    StatsBase& operator = (const StatsBase&) = delete;
    virtual ~StatsBase() = default;

    std::pair<T, T> extrema;
    std::size_t max_bin;

    std::shared_ptr<std::vector<std::uint64_t>> histogram;
    // A numpy array that is a read-only view of histogram. Lazily created in response to get_histogram_py calls.
    std::shared_ptr<py::object> histogram_py;

    explicit virtual operator std::string () const;
    py::object& get_histogram_py();
    virtual void set_bin_count(std::size_t bin_count);
    void find_max_bin();
    virtual void aggregate(const StatsBase& from);
};

template<typename T, bool = std::is_integral<T>::value>
struct Stats;

template<typename T>
struct Stats<T, true>
  : StatsBase<T>
{
    static void expose_via_pybind11(py::module& m);
};

template<typename T>
struct Stats<T, false>
  : StatsBase<T>
{
    static void expose_via_pybind11(py::module& m);

    Stats();

    std::uint64_t NaN_count, neg_inf_count, pos_inf_count;

    explicit operator std::string () const override;
    void aggregate(const StatsBase<T>& from) override;
};

// This is neat: we only need to provide specializations for Stats; ImageStats automatically inherits the correct
// specialization and therefore gets the extra overall fields (NaN count, neg_inf_count, pos_inf_count) without futher
// ado.
template<typename T>
struct ImageStats
  : std::enable_shared_from_this<ImageStats<T>>,
    Stats<T>
{
    static void expose_via_pybind11(py::module& m);
    explicit operator std::string () const override;
    std::vector<std::shared_ptr<Stats<T>>> channel_stats;
    void set_bin_count(std::size_t bin_count) override;
};

#include "statses_impl.h"
