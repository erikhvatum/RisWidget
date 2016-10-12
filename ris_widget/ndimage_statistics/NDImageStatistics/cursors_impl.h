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

#include "cursors.h"

template<typename T>
CursorBase<T>::CursorBase(PyArrayView& data_view)
  : scanline_valid(false),
    pixel_valid(false),
    component_valid(false),
    scanline_count(data_view.shape[1]),
    scanline_stride(data_view.strides[1]),
    scanlines_origin(reinterpret_cast<const T*>(data_view.buf)),
    scanlines_raw_end(reinterpret_cast<std::uint8_t*>(data_view.buf) + scanline_stride * scanline_count),
    scanline_width(data_view.shape[0]),
    pixel_stride(data_view.strides[0]),
    component_count(1),
    component_stride(sizeof(T))
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



template<typename T, typename MASK_T>
Cursor<T, MASK_T>::Cursor(PyArrayView& data_view, MASK_T& /*mask_*/)
  : NonPerComponentMaskCursor<T>(data_view)
{
}

template<typename T, typename MASK_T>
void Cursor<T, MASK_T>::seek_front_scanline()
{
    this->scanline_raw = reinterpret_cast<const std::uint8_t*>(this->scanlines_origin);
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
Cursor<T, BitmapMask<T>>::Cursor(PyArrayView& data_view, BitmapMask<T>& mask_)
  : NonPerComponentMaskCursor<T>(data_view),
    mask_scanline_count(mask_.bitmap_view.shape[1]),
    mask_scanline_stride(mask_.bitmap_view.strides[1]),
    mask_scanlines_origin(reinterpret_cast<const std::uint8_t*const>(mask_.bitmap_view.buf)),
    mask_scanline_width(mask_.bitmap_view.shape[0]),
    mask_element_stride(mask_.bitmap_view.strides[0]),
    im_to_mask_scanline_idx_lut(Luts::getLut(this->scanline_count, mask_scanline_count)),
    im_to_mask_pixel_idx_lut(Luts::getLut(this->scanline_width, mask_scanline_width)),
    mask(mask_)
{
    if(mask_scanline_count < this->scanline_count)
        const_cast<LutPtr&>(mask_to_im_scanline_idx_lut) = Luts::getLut(mask_scanline_count, this->scanline_count);
    if(mask_scanline_width < this->scanline_width)
        const_cast<LutPtr&>(mask_to_im_pixel_idx_lut) = Luts::getLut(mask_scanline_width, this->scanline_width);
}

template<typename T>
void Cursor<T, BitmapMask<T>>::seek_front_scanline()
{
    if(mask_scanline_count < this->scanline_count && mask_scanline_width < this->scanline_width)
    {
        std::size_t mask_scanline_idx=0, mask_element_idx;
        for ( const std::uint8_t* mask_scanline = mask_scanlines_origin;
              mask_scanline_idx < mask_scanline_count;
              ++mask_scanline_idx, mask_scanline += mask_scanline_stride )
        {
            for ( mask_element_idx = 0, mask_element = mask_scanline;
                  mask_element_idx < mask_scanline_width;
                  mask_element += mask_element_stride, ++mask_element_idx )
            {
                if(*mask_element != 0)
                {
                    scanline_idx = mask_to_im_scanline_idx_lut->m_data[mask_scanline_idx];
                    pixel_idx = mask_to_im_pixel_idx_lut->m_data[mask_element_idx];
                    this->scanline_raw =
                        reinterpret_cast<const std::uint8_t*>(this->scanlines_origin) + scanline_idx * this->scanline_stride;
                    this->pixels_raw_end = this->scanline_raw + this->scanline_width * this->pixel_stride;
                    this->pixel_raw = this->scanline_raw + pixel_idx * this->pixel_stride;
                    this->scanline_valid = true;
                    this->pixel_valid = true;
                    this->component_valid = false;
                    at_unmasked_front_of_scanline = true;
                    return;
                }
            }
        }
    }
    else
    {
        this->scanline_raw = reinterpret_cast<const std::uint8_t*>(this->scanlines_origin);
        for(scanline_idx = 0; this->scanline_raw < this->scanlines_raw_end; this->scanline_raw += this->scanline_stride, ++scanline_idx)
        {
            mask_scanline = mask_scanlines_origin + im_to_mask_
            mask_scanline = reinterpret_cast<std::uint8_t*>(mask.bitmap_view.buf) +
                static_cast<std::ptrdiff_t>(scanline_idx * im_mask_h_ratio) * mask.bitmap_view.strides[1];
            for ( pixel_idx = 0, this->pixel_raw = this->scanline_raw, this->pixels_raw_end = this->scanline_raw + this->scanline_width * this->pixel_stride;
                  this->pixel_raw < this->pixels_raw_end;
                  this->pixel_raw += this->pixel_stride, ++pixel_idx )
            {
                mask_element = mask_scanline + static_cast<std::ptrdiff_t>(pixel_idx * im_mask_w_ratio) * mask.bitmap_view.strides[0];
                if(*mask_element != 0)
                {
                    this->scanline_valid = true;
                    this->pixel_valid = true;
                    this->component_valid = false;
                    at_unmasked_front_of_scanline = true;
                    return;
                }
            }
        }
    }
    this->scanline_valid = false;
    this->pixel_valid = false;
    this->component_valid = false;
    at_unmasked_front_of_scanline = false;
}

template<typename T>
void Cursor<T, BitmapMask<T>>::advance_scanline()
{
    assert(this->scanline_valid);
    this->scanline_raw += this->scanline_stride;
    this->scanline_valid = this->scanline_raw < this->scanlines_raw_end;
    this->pixel_valid = false;
    this->component_valid = false;
    at_unmasked_front_of_scanline = false;
    ++scanline_idx;
}

template<typename T>
void Cursor<T, BitmapMask<T>>::seek_front_pixel_of_scanline()
{
    assert(this->scanline_valid);
    if(!at_unmasked_front_of_scanline)
    {
        if(im_mask_w_ratio > 1)
        {
            mask_element = reinterpret_cast<std::uint8_t*>(mask.bitmap_view.buf) +
                static_cast<std::ptrdiff_t>(scanline_idx * im_mask_h_ratio) * mask.bitmap_view.strides[1];
            std::ptrdiff_t mask_element_idx = 0;
            const std::uint8_t*const mask_elements_end = mask_element + mask.bitmap_view.shape[0] * mask.bitmap_view.strides[0];
            for(; mask_element < mask_elements_end; mask_element += mask.bitmap_view.strides[0], ++mask_element_idx)
            {
                if(*mask_element != 0)
                {
                    pixel_idx = mask_element_idx / im_mask_w_ratio;
                    this->pixel_raw = this->scanline_raw + pixel_idx * this->pixel_stride;
                    this->pixels_raw_end = this->pixel_raw + this->scanline_width * this->pixel_stride;
                    this->pixel_valid = this->pixel_raw < this->pixels_raw_end;
                    this->component_valid = false;
                    at_unmasked_front_of_scanline = true;
                    return;
                }
            }
        }
        else
        {
            const std::uint8_t*const mask_scanline = reinterpret_cast<std::uint8_t*>(mask.bitmap_view.buf) +
                static_cast<std::ptrdiff_t>(scanline_idx * im_mask_h_ratio) * mask.bitmap_view.strides[1];
            for ( pixel_idx = 0, this->pixel_raw = this->scanline_raw, this->pixels_raw_end = this->scanline_raw + this->scanline_width * this->pixel_stride;
                  this->pixel_raw < this->pixels_raw_end;
                  this->pixel_raw += this->pixel_stride, ++pixel_idx )
            {
                mask_element = mask_scanline + static_cast<std::ptrdiff_t>(pixel_idx * im_mask_w_ratio) * mask.bitmap_view.strides[0];
                if(*mask_element != 0)
                {
                    this->pixel_valid = true;
                    this->component_valid = false;
                    at_unmasked_front_of_scanline = true;
                    return;
                }
            }
        }
        this->pixel_valid = false;
        this->component_valid = false;
        at_unmasked_front_of_scanline = false;
    }
}

template<typename T>
void Cursor<T, BitmapMask<T>>::advance_pixel()
{
    assert(this->scanline_valid);
    assert(this->pixel_valid);
    at_unmasked_front_of_scanline = false;
    this->component_valid = false;
    const std::uint8_t*const mask_scanline = reinterpret_cast<std::uint8_t*>(mask.bitmap_view.buf) +
        static_cast<std::ptrdiff_t>(scanline_idx * im_mask_h_ratio) * mask.bitmap_view.strides[1];
    this->pixel_raw += this->pixel_stride;
    ++pixel_idx;
    for(; this->pixel_raw < this->pixels_raw_end; this->pixel_raw += this->pixel_stride, ++pixel_idx)
    {
        mask_element = mask_scanline + static_cast<std::ptrdiff_t>(pixel_idx * im_mask_w_ratio) * mask.bitmap_view.strides[0];
        if(*mask_element != 0)
        {
            this->pixel_valid = true;
            return;
        }
    }
    this->pixel_valid = false;
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