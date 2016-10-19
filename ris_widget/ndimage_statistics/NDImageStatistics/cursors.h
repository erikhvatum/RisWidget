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
#include "masks.h"

// Cursors (ie, CursorBase derivatives) are built-to-purpose matrix iterators. For an example of usage, see the
// definition of NDImageStatistics<T>::scan_image(ComputeContext<MASK_T>&, const COMPUTE_TAG&) in
// NDImageStatistics_impl.h.

template<typename T>
struct CursorBase
{
    explicit CursorBase(PyArrayView& data_view);
    CursorBase(const CursorBase&) = delete;
    CursorBase& operator = (const CursorBase&) = delete;
    virtual ~CursorBase() = default;

    volatile bool scanline_valid, pixel_valid, component_valid;

    const std::size_t scanline_count;
    const std::size_t scanline_stride;
    const T*const scanlines_origin;
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
    // The component member variable is a typecasting reference to component_raw. It acts as a toll-free
    // bridge.
    const T*& component = reinterpret_cast<const T*&>(component_raw);
    // TODO: verify toll-freeness
};

template<typename T>
struct NonPerComponentMaskCursor
  : CursorBase<T>
{
    using CursorBase<T>::CursorBase;

    inline void seek_front_component_of_pixel();
    inline void advance_component();
};

// Primary Cursor template; used concretely in the case where MASK_T is of type Mask (eg, a null mask)
template<typename T, typename MASK_T>
struct Cursor
  : NonPerComponentMaskCursor<T>
{
    Cursor(PyArrayView& data_view, MASK_T& mask_);

    inline void seek_front_scanline();
    inline void advance_scanline();
    inline void seek_front_pixel_of_scanline();
    inline void advance_pixel();
};

// Specialized aspects of the Cursor<T, BitmapMask> specialization, BitmapMaskCursorXxxxxAdvanceAspects are mixins that
// use the curiously recurring template pattern to gain access to members of the composed class
// https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
#include "bitmap_mask_cursor_aspects.h"

// Cursor specialization for BitmapMask
template<typename T, BitmapMaskDimensionVsImage T_W, BitmapMaskDimensionVsImage T_H>
struct Cursor<T, BitmapMask<T, T_W, T_H>>
  : NonPerComponentMaskCursor<T>,
    BitmapMaskCursorScanlineAdvanceAspect<Cursor<T, BitmapMask<T, T_W, T_H>>, T, T_W, T_H>,
    BitmapMaskCursorPixelAdvanceAspect   <Cursor<T, BitmapMask<T, T_W, T_H>>, T, T_W, T_H>
{
    Cursor(PyArrayView& data_view, BitmapMask<T, T_W, T_H>& mask_);

    std::ptrdiff_t scanline_idx;
    std::ptrdiff_t pixel_idx;

    const std::size_t mask_scanline_count;
    const std::size_t mask_scanline_stride;
    const std::uint8_t*const mask_scanlines_origin;
    const std::uint8_t* mask_scanline;

    const std::size_t mask_scanline_width;
    const std::size_t mask_element_stride;
    const std::uint8_t* mask_element;

    BitmapMask<T, T_W, T_H>& mask;

    inline void seek_front_scanline();
    inline void seek_front_pixel_of_scanline();
};

// Cursor specialization for CircularMask
template<typename T>
struct Cursor<T, CircularMask<T>>
  : NonPerComponentMaskCursor<T>
{
    Cursor(PyArrayView& data_view, CircularMask<T>& mask_);

    inline void seek_front_scanline();
    inline void advance_scanline();
    inline void seek_front_pixel_of_scanline();
    inline void advance_pixel();

    CircularMask<T>& mask;
    PeroneCircleLutPtr bounds_lut;
    const std::int32_t* bound;
};

#include "cursors_impl.h"
