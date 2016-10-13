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

    bool scanline_valid, pixel_valid, component_valid;

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

    void seek_front_component_of_pixel();
    void advance_component();
};

// Primary Cursor template; used concretely in the case where MASK_T is of type Mask (eg, a null mask)
template<typename T, typename MASK_T>
struct Cursor
  : NonPerComponentMaskCursor<T>
{
    Cursor(PyArrayView& data_view, MASK_T& mask_);

    void seek_front_scanline();
    void advance_scanline();
    void seek_front_pixel_of_scanline();
    void advance_pixel();
};

// Cursor specialization for BitmapMask
template<typename T, BitmapMaskDimensionVsImage T_W, BitmapMaskDimensionVsImage T_H>
struct Cursor<T, BitmapMask<T, T_W, T_H>>
  : NonPerComponentMaskCursor<T>
{
    Cursor(PyArrayView& data_view, BitmapMask<T>& mask_);

    std::ptrdiff_t scanline_idx;
    std::ptrdiff_t pixel_idx;

    const std::size_t mask_scanline_count;
    const std::size_t mask_scanline_stride;
    const std::uint8_t*const mask_scanlines_origin;
    const std::uint8_t* mask_scanline;
    bool at_unmasked_front_of_scanline;

    const std::size_t mask_scanline_width;
    const std::size_t mask_element_stride;
    const std::uint8_t* mask_element;

    const LutPtr im_to_mask_scanline_idx_lut;
    const LutPtr im_to_mask_pixel_idx_lut;
    // mask_to_im LUTs are used to map from mask index to image index - an operation required only for dimensions where
    // mask is smaller than image. For dimensions where this is not the case, the mask_to_im LutPtrs are null.
    const LutPtr mask_to_im_scanline_idx_lut;
    const LutPtr mask_to_im_pixel_idx_lut;

    BitmapMask<T>& mask;

    void seek_front_scanline();

    // SFINAE is used to hide all but the desired pair of scanline and pixel advance functions. There are other
    // approaches to implementing advance_scanline/pixel static polymorphism, but the most natural, simply implementing
    // specializations for advance_scanline templated to with default value of T_W, is not possible:
    //
    // C++11 14.7.3/16: In an explicit specialization declaration for a member of a class template or a member template
    // that appears in namespace scope, the member template and some of its enclosing class templates may remain
    // unspecialized, except that the declaration shall not explicitly specialize a class member template if its
    // enclosing class templates are not explicitly specialized as well.

    template<bool MASK_IS_SHORTER=T_H==BitmapMaskDimensionVsImage::Smaller>
    void advance_scanline(char(*)[MASK_IS_SHORTER]=0);
    template<bool MASK_IS_SAME_HEIGHT=T_H==BitmapMaskDimensionVsImage::Same>
    void advance_scanline(char(*)[MASK_IS_SAME_HEIGHT]=0);
    template<bool MASK_IS_TALLER=T_H==BitmapMaskDimensionVsImage::Larger>
    void advance_scanline(char(*)[MASK_IS_TALLER]=0);

    void seek_front_pixel_of_scanline();

    template<bool MASK_IS_NARROWER=T_W==BitmapMaskDimensionVsImage::Smaller>
    void advance_pixel(char(*)[MASK_IS_NARROWER]=0);
    template<bool MASK_IS_SAME_WIDTH=T_W==BitmapMaskDimensionVsImage::Same>
    void advance_pixel(char(*)[MASK_IS_SAME_WIDTH]=0);
    template<bool MASK_IS_WIDER=T_W==BitmapMaskDimensionVsImage::Larger>
    void advance_pixel(char(*)[MASK_IS_WIDER]=0);
};

// Cursor specialization for CircularMask
template<typename T>
struct Cursor<T, CircularMask<T>>
  : NonPerComponentMaskCursor<T>
{
    Cursor(PyArrayView& data_view, CircularMask<T>& mask_);

    void seek_front_scanline();
    void advance_scanline();
    void seek_front_pixel_of_scanline();
    void advance_pixel();

    CircularMask<T>& mask;
};

#include "cursors_impl.h"
