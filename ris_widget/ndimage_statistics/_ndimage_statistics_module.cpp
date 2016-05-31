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

#include <stdexcept>
#include <sstream>
#include <pybind11/pybind11.h>
#include "_ndimage_statistics.h"
#include "resampling_lut.h"

namespace py = pybind11;

template<typename C>
static inline bool min_max_C(py::buffer_info& im_info, py::buffer_info& min_max_info)
{
    bool ret{false};
    if(im_info.format == py::format_descriptor<C>::value)
    {
        py::gil_scoped_release releaseGil;
        min_max(
           reinterpret_cast<C*>(im_info.ptr),
           im_info.shape.data(),
           im_info.strides.data(),
           reinterpret_cast<C*>(min_max_info.ptr),
           min_max_info.strides[0]);
        ret = true;
    }
    return ret;
}

// It would be nice to use distinct py_min_max(py::array_t</*element type*/>...) overload definitions rather than 
// dispatching from a single frontend, but this does not work as py::array_t attempts to cast.  That is, supplied with 
// an numpy array of dtype numpy.uint64, m.def("min_max", [](py::array_t<float>...) would be called with temporary 
// arguments holding inputs converted to float arrays.  This is never something we want. 
void py_min_max(py::buffer im, py::buffer min_max)
{
    py::buffer_info im_info{im.request()}, min_max_info{min_max.request()};
    if(im_info.ndim != 2)
        throw std::invalid_argument("im argument must be a 2 dimensional buffer object (such as a numpy array).");
    if(min_max_info.ndim != 1)
        throw std::invalid_argument("min_max arugment must be a 1 dimensional buffer object (such as a numpy array).");
    if(min_max_info.shape[0] != 2)
        throw std::invalid_argument("min_max argument must contain exactly 2 elements.");
    if(im_info.format != min_max_info.format)
        throw std::invalid_argument(
            "im and min_max arguments must be the same format (or dtype, in the case where they are numpy arays).");
    if ( !(
        min_max_C<float>(im_info, min_max_info) ||
        min_max_C<std::uint8_t>(im_info, min_max_info) ||
        min_max_C<std::uint16_t>(im_info, min_max_info) ||
        min_max_C<std::uint32_t>(im_info, min_max_info) ||
        min_max_C<std::uint64_t>(im_info, min_max_info) ||
        min_max_C<double>(im_info, min_max_info)) )
    {
       throw std::invalid_argument("Only uint8, uint16, uint32, uint64, float32, and float64 buffers are supported.");
    }
}

template<typename C>
static inline bool masked_min_max_C(py::buffer_info& im_info, py::buffer_info& mask_info, py::buffer_info& min_max_info)
{
    bool ret{false};
    if(im_info.format == py::format_descriptor<C>::value)
    {
        py::gil_scoped_release releaseGil;
        masked_min_max(
           reinterpret_cast<C*>(im_info.ptr),
           im_info.shape.data(),
           im_info.strides.data(),
           reinterpret_cast<std::uint8_t*>(mask_info.ptr),
           mask_info.shape.data(),
           mask_info.strides.data(),
           reinterpret_cast<C*>(min_max_info.ptr),
           min_max_info.strides[0]);
        ret = true;
    }
    return ret;
}

void py_masked_min_max(py::buffer im, py::buffer mask, py::buffer min_max)
{
    py::buffer_info im_info{im.request()}, min_max_info{min_max.request()}, mask_info{mask.request()};
    if(im_info.ndim != 2)
        throw std::invalid_argument("im argument must be a 2 dimensional buffer object (such as a numpy array).");
    if(mask_info.ndim != 2)
        throw std::invalid_argument("mask argument must be a 2 dimensionsal buffer object (such as a numpy array).");
    if(mask_info.format != py::format_descriptor<std::uint8_t>::value && mask_info.format != py::format_descriptor<bool>::value)
        throw std::invalid_argument("mask argument format must be uint8 or bool.");
    if(min_max_info.ndim != 1)
        throw std::invalid_argument("min_max arugment must be a 1 dimensional buffer object (such as a numpy array).");
    if(min_max_info.shape[0] != 2)
        throw std::invalid_argument("min_max argument must contain exactly 2 elements.");
    if(im_info.format != min_max_info.format)
        throw std::invalid_argument(
            "im and min_max arguments must be the same format (or dtype, in the case where they are numpy arays).");
    if ( !(
        masked_min_max_C<float>(im_info, mask_info, min_max_info) ||
        masked_min_max_C<std::uint8_t>(im_info, mask_info, min_max_info) ||
        masked_min_max_C<std::uint16_t>(im_info, mask_info, min_max_info) ||
        masked_min_max_C<std::uint32_t>(im_info, mask_info, min_max_info) ||
        masked_min_max_C<std::uint64_t>(im_info, mask_info, min_max_info) ||
        masked_min_max_C<double>(im_info, mask_info, min_max_info)) )
    {
        throw std::invalid_argument("Only uint8, uint16, uint32, uint64, float32, and float64 im and min_max buffers are supported.");
    }
}

template<typename C, bool with_overflow_bins>
static inline bool ranged_hist_C(py::buffer_info& im_info, py::buffer_info& range_info, py::buffer_info& hist_info)
{
    bool ret{false};
    if(im_info.format == py::format_descriptor<C>::value)
    {
        py::gil_scoped_release releaseGil;
        ranged_hist<C, with_overflow_bins>(
           reinterpret_cast<C*>(im_info.ptr),
           im_info.shape.data(),
           im_info.strides.data(),
           reinterpret_cast<C*>(range_info.ptr),
           range_info.strides[0],
           hist_info.shape[0],
           reinterpret_cast<std::uint32_t*>(hist_info.ptr),
           hist_info.strides[0]);
        ret = true;
    }
    return ret;
}

void py_ranged_hist(py::buffer im, py::buffer range, py::buffer hist, bool with_overflow_bins)
{
    py::buffer_info im_info{im.request()}, range_info{range.request()}, hist_info{hist.request()};
    if(im_info.ndim != 2)
        throw std::invalid_argument("im argument must be a 2 dimensional buffer object (such as a numpy array).");
    if(im_info.format != range_info.format)
        throw std::invalid_argument(
            "im and range arguments must be the same format (or dtype, in the case where they are numpy arays).");
    if(hist_info.ndim != 1)
        throw std::invalid_argument("hist argument must be a 1 dimensional buffer object (such as a numpy array).");
    if(hist_info.format != py::format_descriptor<std::uint32_t>::value)
        throw std::invalid_argument("hist argument format must be uint32.");
    if
    (
        !(
            (
                with_overflow_bins
                &&
                (
                    ranged_hist_C<float, true>(im_info, range_info, hist_info) ||
                    ranged_hist_C<std::uint8_t, true>(im_info, range_info, hist_info) ||
                    ranged_hist_C<std::uint16_t, true>(im_info, range_info, hist_info) ||
                    ranged_hist_C<std::uint32_t, true>(im_info, range_info, hist_info) ||
                    ranged_hist_C<std::uint64_t, true>(im_info, range_info, hist_info) ||
                    ranged_hist_C<double, true>(im_info, range_info, hist_info)
                )
            )
            ||
            (
                !with_overflow_bins
                &&
                (
                    ranged_hist_C<float, false>(im_info, range_info, hist_info) ||
                    ranged_hist_C<std::uint8_t, false>(im_info, range_info, hist_info) ||
                    ranged_hist_C<std::uint16_t, false>(im_info, range_info, hist_info) ||
                    ranged_hist_C<std::uint32_t, false>(im_info, range_info, hist_info) ||
                    ranged_hist_C<std::uint64_t, false>(im_info, range_info, hist_info) ||
                    ranged_hist_C<double, false>(im_info, range_info, hist_info)
                )
            )
        )
    )
    {
        throw std::invalid_argument("Only uint8, uint16, uint32, uint64, float32, and float64 im and range buffers are supported.");
    }
}

template<typename C, bool with_overflow_bins>
static inline bool masked_ranged_hist_C(py::buffer_info& im_info, py::buffer_info& mask_info, py::buffer_info& range_info, py::buffer_info& hist_info)
{
    bool ret{false};
    if(im_info.format == py::format_descriptor<C>::value)
    {
        py::gil_scoped_release releaseGil;
        masked_ranged_hist<C, with_overflow_bins>(
           reinterpret_cast<C*>(im_info.ptr),
           im_info.shape.data(),
           im_info.strides.data(),
           reinterpret_cast<std::uint8_t*>(mask_info.ptr),
           mask_info.shape.data(),
           mask_info.strides.data(),
           reinterpret_cast<C*>(range_info.ptr),
           range_info.strides[0],
           hist_info.shape[0],
           reinterpret_cast<std::uint32_t*>(hist_info.ptr),
           hist_info.strides[0]);
        ret = true;
    }
    return ret;
}

void py_masked_ranged_hist(py::buffer im, py::buffer mask, py::buffer range, py::buffer hist, bool with_overflow_bins)
{
    py::buffer_info im_info{im.request()}, mask_info{mask.request()}, range_info{range.request()}, hist_info{hist.request()};
    if(im_info.ndim != 2)
        throw std::invalid_argument("im argument must be a 2 dimensional buffer object (such as a numpy array).");
    if(mask_info.ndim != 2)
        throw std::invalid_argument("mask argument must be a 2 dimensionsal buffer object (such as a numpy array).");
    if(mask_info.format != py::format_descriptor<std::uint8_t>::value && mask_info.format != py::format_descriptor<bool>::value)
        throw std::invalid_argument("mask argument format must be uint8 or bool.");
    if(im_info.format != range_info.format)
        throw std::invalid_argument(
            "im and range arguments must be the same format (or dtype, in the case where they are numpy arays).");
    if(hist_info.ndim != 1)
        throw std::invalid_argument("hist argument must be a 1 dimensional buffer object (such as a numpy array).");
    if(hist_info.format != py::format_descriptor<std::uint32_t>::value)
        throw std::invalid_argument("hist argument format must be uint32.");
    if
    (
        !(
            (
                with_overflow_bins
                &&
                (
                    masked_ranged_hist_C<float, true>(im_info, mask_info, range_info, hist_info) ||
                    masked_ranged_hist_C<std::uint8_t, true>(im_info, mask_info, range_info, hist_info) ||
                    masked_ranged_hist_C<std::uint16_t, true>(im_info, mask_info, range_info, hist_info) ||
                    masked_ranged_hist_C<std::uint32_t, true>(im_info, mask_info, range_info, hist_info) ||
                    masked_ranged_hist_C<std::uint64_t, true>(im_info, mask_info, range_info, hist_info) ||
                    masked_ranged_hist_C<double, true>(im_info, mask_info, range_info, hist_info)
                )
            )
            ||
            (
                !with_overflow_bins
                &&
                (
                    masked_ranged_hist_C<float, false>(im_info, mask_info, range_info, hist_info) ||
                    masked_ranged_hist_C<std::uint8_t, false>(im_info, mask_info, range_info, hist_info) ||
                    masked_ranged_hist_C<std::uint16_t, false>(im_info, mask_info, range_info, hist_info) ||
                    masked_ranged_hist_C<std::uint32_t, false>(im_info, mask_info, range_info, hist_info) ||
                    masked_ranged_hist_C<std::uint64_t, false>(im_info, mask_info, range_info, hist_info) ||
                    masked_ranged_hist_C<double, false>(im_info, mask_info, range_info, hist_info)
                )
            )
        )
    )
    {
        throw std::invalid_argument("Only uint8, uint16, uint32, uint64, float32, and float64 im and range buffers are supported.");
    }
}

template<typename C, bool is_twelve_bit>
static inline bool hist_min_max_C(py::buffer_info& im_info, py::buffer_info& hist_info, py::buffer_info& min_max_info)
{
    bool ret{false};
    if(im_info.format == py::format_descriptor<C>::value)
    {
        if(hist_info.shape[0] != bin_count<C>())
        {
            std::ostringstream o;
            o << "hist argument must contain " << bin_count<C>() << " elements for " << py::format_descriptor<C>::value << " im.";
            throw std::invalid_argument(o.str());
        }
        py::gil_scoped_release releaseGil;
        hist_min_max<C, is_twelve_bit>(
           reinterpret_cast<C*>(im_info.ptr),
           im_info.shape.data(),
           im_info.strides.data(),
           reinterpret_cast<std::uint32_t*>(hist_info.ptr),
           hist_info.strides[0],
           reinterpret_cast<C*>(min_max_info.ptr),
           min_max_info.strides[0]);
        ret = true;
    }
    return ret;
}

void py_hist_min_max(py::buffer im, py::buffer hist, py::buffer min_max, bool is_twelve_bit)
{
    py::buffer_info im_info{im.request()}, hist_info{hist.request()}, min_max_info{min_max.request()};
    if(im_info.ndim != 2)
        throw std::invalid_argument("im argument must be a 2 dimensional buffer object (such as a numpy array).");
    if(hist_info.ndim != 1)
        throw std::invalid_argument("hist argument must be a 1 dimensional buffer object (such as a numpy array).");
    if(hist_info.format != py::format_descriptor<std::uint32_t>::value)
        throw std::invalid_argument("hist argument format must be uint32.");
    if(min_max_info.ndim != 1)
        throw std::invalid_argument("min_max arugment must be a 1 dimensional buffer object (such as a numpy array).");
    if(min_max_info.shape[0] != 2)
        throw std::invalid_argument("min_max argument must contain exactly 2 elements.");
    if(im_info.format != min_max_info.format)
        throw std::invalid_argument(
            "im and min_max arguments must be the same format (or dtype, in the case where they are numpy arays).");
    if(is_twelve_bit)
    {
        if(!hist_min_max_C<std::uint16_t, true>(im_info, hist_info, min_max_info))
        {
            throw std::invalid_argument("is_twelve_bit may be True only if im is uint16.");
        }
    }
    else
    {
        if ( !(
            hist_min_max_C<std::uint8_t, false>(im_info, hist_info, min_max_info) || 
            hist_min_max_C<std::uint16_t, false>(im_info, hist_info, min_max_info) || 
            hist_min_max_C<std::uint32_t, false>(im_info, hist_info, min_max_info) || 
            hist_min_max_C<std::uint64_t, false>(im_info, hist_info, min_max_info)) )
        {
            throw std::invalid_argument("Only uint8, uint16, uint32, and uint64 im buffers are supported.");
        }
    }
}

template<typename C, bool is_twelve_bit>
static inline bool masked_hist_min_max_C(py::buffer_info& im_info, py::buffer_info& mask_info, py::buffer_info& hist_info, py::buffer_info& min_max_info, bool use_open_mp)
{
    bool ret{false};
    if(im_info.format == py::format_descriptor<C>::value)
    {
        if(hist_info.shape[0] != bin_count<C>())
        {
            std::ostringstream o;
            o << "hist argument must contain " << bin_count<C>() << " elements for " << py::format_descriptor<C>::value << " im.";
            throw std::invalid_argument(o.str());
        }
        py::gil_scoped_release releaseGil;
        #ifdef _OPENMP
        if(use_open_mp)
        {
            masked_hist_min_max_omp<C, is_twelve_bit>(
               reinterpret_cast<C*>(im_info.ptr),
               im_info.shape.data(),
               im_info.strides.data(),
               reinterpret_cast<std::uint8_t*>(mask_info.ptr),
               mask_info.shape.data(),
               mask_info.strides.data(),
               reinterpret_cast<std::uint32_t*>(hist_info.ptr),
               hist_info.strides[0],
               reinterpret_cast<C*>(min_max_info.ptr),
               min_max_info.strides[0]);
        }
        else
        {
            masked_hist_min_max<C, is_twelve_bit>(
               reinterpret_cast<C*>(im_info.ptr),
               im_info.shape.data(),
               im_info.strides.data(),
               reinterpret_cast<std::uint8_t*>(mask_info.ptr),
               mask_info.shape.data(),
               mask_info.strides.data(),
               reinterpret_cast<std::uint32_t*>(hist_info.ptr),
               hist_info.strides[0],
               reinterpret_cast<C*>(min_max_info.ptr),
               min_max_info.strides[0]);
        }
        #else
        masked_hist_min_max<C, is_twelve_bit>(
           reinterpret_cast<C*>(im_info.ptr),
           im_info.shape.data(),
           im_info.strides.data(),
           reinterpret_cast<std::uint8_t*>(mask_info.ptr),
           mask_info.shape.data(),
           mask_info.strides.data(),
           reinterpret_cast<std::uint32_t*>(hist_info.ptr),
           hist_info.strides[0],
           reinterpret_cast<C*>(min_max_info.ptr),
           min_max_info.strides[0]);
        #endif
        ret = true;
    }
    return ret;
}

void py_masked_hist_min_max(py::buffer im, py::buffer mask, py::buffer hist, py::buffer min_max, bool is_twelve_bit, bool use_open_mp)
{
    py::buffer_info im_info{im.request()}, mask_info{mask.request()}, hist_info{hist.request()}, min_max_info{min_max.request()};
    if(im_info.ndim != 2)
        throw std::invalid_argument("im argument must be a 2 dimensional buffer object (such as a numpy array).");
    if(mask_info.ndim != 2)
        throw std::invalid_argument("mask argument must be a 2 dimensionsal buffer object (such as a numpy array).");
    if(mask_info.format != py::format_descriptor<std::uint8_t>::value && mask_info.format != py::format_descriptor<bool>::value)
        throw std::invalid_argument("mask argument format must be uint8 or bool.");
    if(hist_info.ndim != 1)
        throw std::invalid_argument("hist argument must be a 1 dimensional buffer object (such as a numpy array).");
    if(hist_info.format != py::format_descriptor<std::uint32_t>::value)
        throw std::invalid_argument("hist argument format must be uint32.");
    if(min_max_info.ndim != 1)
        throw std::invalid_argument("min_max arugment must be a 1 dimensional buffer object (such as a numpy array).");
    if(min_max_info.shape[0] != 2)
        throw std::invalid_argument("min_max argument must contain exactly 2 elements.");
    if(im_info.format != min_max_info.format)
        throw std::invalid_argument(
            "im and min_max arguments must be the same format (or dtype, in the case where they are numpy arays).");
    if(is_twelve_bit)
    {
        if(!masked_hist_min_max_C<std::uint16_t, true>(im_info, mask_info, hist_info, min_max_info, use_open_mp))
        {
            throw std::invalid_argument("is_twelve_bit may be True only if im is uint16.");
        }
    }
    else
    {
        if ( !(
            masked_hist_min_max_C<std::uint8_t, false>(im_info, mask_info, hist_info, min_max_info, use_open_mp) ||
            masked_hist_min_max_C<std::uint16_t, false>(im_info, mask_info, hist_info, min_max_info, use_open_mp) ||
            masked_hist_min_max_C<std::uint32_t, false>(im_info, mask_info, hist_info, min_max_info, use_open_mp) ||
            masked_hist_min_max_C<std::uint64_t, false>(im_info, mask_info, hist_info, min_max_info, use_open_mp)) )
        {
            throw std::invalid_argument("Only uint8, uint16, uint32, and uint64 im buffers are supported.");
        }
    }
}

PYBIND11_PLUGIN(_ndimage_statistics)
{
    py::module m("_ndimage_statistics", "ris_widget.ndimage_statistics._ndimage_statistics module");

    m.def("min_max", &py_min_max);
    m.def("masked_min_max", &py_masked_min_max);
    m.def("ranged_hist", &py_ranged_hist);
    m.def("masked_ranged_hist", &py_masked_ranged_hist);
    m.def("hist_min_max", &py_hist_min_max);
    m.def("masked_hist_min_max", &py_masked_hist_min_max);

//  py::class_<Luts>(m, "Luts")
//      .def(py::init<const std::size_t&>())
//      .def("getLut", [](Luts& luts, const std::uint32_t& fromSamples, const std::uint32_t& toSamples){
//          luts.getLut(fromSamples, toSamples);
//      });

    return m.ptr();
}