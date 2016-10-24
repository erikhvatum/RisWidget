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

template<typename BitmapMaskCursor, typename T, BitmapMaskDimensionVsImage T_W, BitmapMaskDimensionVsImage T_H>
struct BitmapMaskCursorScanlineAdvanceAspect;

template<typename BitmapMaskCursor, typename T, BitmapMaskDimensionVsImage T_W>
struct BitmapMaskCursorScanlineAdvanceAspect<BitmapMaskCursor, T, T_W, BitmapMaskDimensionVsImage::Smaller>
{
    const std::uint8_t*const mask_scanlines_end;
    const SampleLutPtr im_to_mask_scanline_idx_lut;
//  const std::uint64_t* im_to_mask_scanline_lut_element;
    const SampleLutPtr mask_to_im_scanline_idx_lut;
    const std::uint64_t* mask_to_im_scanline_lut_element;

    BitmapMaskCursorScanlineAdvanceAspect(PyArrayView& data_view, BitmapMask<T, T_W, BitmapMaskDimensionVsImage::Smaller>& mask_)
      : mask_scanlines_end(static_cast<std::uint8_t*>(mask_.bitmap_view.buf) + mask_.bitmap_view.shape[1] * mask_.bitmap_view.strides[1]),
        im_to_mask_scanline_idx_lut(sampleLuts.getLut(data_view.shape[1], mask_.bitmap_view.shape[1])),
        mask_to_im_scanline_idx_lut(sampleLuts.getLut(mask_.bitmap_view.shape[1], data_view.shape[1])) {}

    inline void update_scanline()
    {
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        assert(S.mask_scanline_valid);
        S.scanline_raw = reinterpret_cast<const std::uint8_t*>(S.scanlines_origin) + *mask_to_im_scanline_lut_element * S.scanline_stride;
    }

    inline void seek_front_scanline()
    {
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        S.mask_scanline = S.mask_scanlines_origin;
        S.mask_scanline_valid = S.mask_scanline < S.mask_scanlines_end;
        if(unlikely(!S.mask_scanline_valid)) return;
        mask_to_im_scanline_lut_element = mask_to_im_scanline_idx_lut->m_data.data();
        S.seek_front_element_of_mask_scanline();
        if(S.mask_element_valid)
            update_scanline();
        else
            advance_mask_scanline();
    }

    inline void advance_mask_scanline()
    {
    }

//  void quick_advance_scanline()
//  {
//      mask_scanline_lut_element = im_to_mask_scanline_idx_lut->m_data.data();
//      im_scanline_lut_element = mask_to_im_scanline_idx_lut->m_data.data();
//      static_cast<BitmapMaskCursor*>(this)->mask_scanline = static_cast<BitmapMaskCursor*>(this)->mask_scanlines_origin[*mask_scanline_lut_element];
//  }

    inline void advance_scanline()
    {
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        assert(S.scanline_valid);
        S.scanline_valid = S.mask_scanline_valid;
        if(likely(S.scanline_valid))
            update_scanline();
    }
};

template<typename BitmapMaskCursor, typename T, BitmapMaskDimensionVsImage T_W>
struct BitmapMaskCursorScanlineAdvanceAspect<BitmapMaskCursor, T, T_W, BitmapMaskDimensionVsImage::Same>
{
    BitmapMaskCursorScanlineAdvanceAspect(PyArrayView& data_view, BitmapMask<T, T_W, BitmapMaskDimensionVsImage::Same>& mask_) {}
    inline void seek_front_scanline() {}
    inline void update_scanline()
    {
    }
    inline void advance_mask_scanline()
    {
    }
    inline void advance_scanline() {}
};

template<typename BitmapMaskCursor, typename T, BitmapMaskDimensionVsImage T_W>
struct BitmapMaskCursorScanlineAdvanceAspect<BitmapMaskCursor, T, T_W, BitmapMaskDimensionVsImage::Larger>
{
    const SampleLutPtr im_to_mask_scanline_idx_lut;
    const std::uint64_t* lut_element;

    BitmapMaskCursorScanlineAdvanceAspect(PyArrayView& data_view, BitmapMask<T, T_W, BitmapMaskDimensionVsImage::Larger>& mask_)
      : im_to_mask_scanline_idx_lut(sampleLuts.getLut(data_view.shape[1], mask_.bitmap_view.shape[1])),
        lut_element(im_to_mask_scanline_idx_lut->m_data.data()) {}
    
    inline void seek_front_scanline() {}
    inline void update_scanline()
    {
    }
    inline void advance_mask_scanline()
    {
    }
    inline void advance_scanline() {}
};


template<typename BitmapMaskCursor, typename T, BitmapMaskDimensionVsImage T_W, BitmapMaskDimensionVsImage T_H>
struct BitmapMaskCursorPixelAdvanceAspect;

template<typename BitmapMaskCursor, typename T, BitmapMaskDimensionVsImage T_H>
struct BitmapMaskCursorPixelAdvanceAspect<BitmapMaskCursor, T, BitmapMaskDimensionVsImage::Smaller, T_H>
{
    const std::uint8_t* mask_elements_end;
    const SampleLutPtr im_to_mask_pixel_idx_lut;
    const std::uint64_t* im_to_mask_pixel_lut_element;
    std::uint64_t prev_im_to_mask_pixel_lut_element_value;
    const std::uint64_t*const im_to_mask_pixel_lut_elements_end;
    const SampleLutPtr mask_to_im_pixel_idx_lut;
    const std::uint64_t* mask_to_im_pixel_lut_element;
    const std::uint64_t*const mask_to_im_pixel_lut_elements_end;

    BitmapMaskCursorPixelAdvanceAspect(PyArrayView& data_view, BitmapMask<T, BitmapMaskDimensionVsImage::Smaller, T_H>& mask_)
      : im_to_mask_pixel_idx_lut(sampleLuts.getLut(data_view.shape[0], mask_.bitmap_view.shape[0])),
        im_to_mask_pixel_lut_elements_end(im_to_mask_pixel_idx_lut->m_data.data() + data_view.shape[0]),
        mask_to_im_pixel_idx_lut(sampleLuts.getLut(mask_.bitmap_view.shape[0], data_view.shape[0])),
        mask_to_im_pixel_lut_elements_end(mask_to_im_pixel_idx_lut->m_data.data() + mask_.bitmap_view.shape[0]) {}

    inline void advance_mask_element()
    {
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        assert(S.mask_element_valid);
        for(;;)
        {
            S.mask_element += S.mask_element_stride;
            S.mask_element_valid = S.mask_element < mask_elements_end;
            ++mask_to_im_pixel_lut_element;
            if(!S.mask_element_valid || *S.mask_element != 0)
                break;
        }
    }

    inline void seek_front_element_of_mask_scanline()
    {
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        assert(S.mask_scanline_valid);
        mask_to_im_pixel_lut_element = mask_to_im_pixel_idx_lut->m_data.data();
        prev_im_to_mask_pixel_lut_element_value = 0;
        S.mask_element = S.mask_scanline;
        S.mask_element_valid = true;
        mask_elements_end = S.mask_element + S.mask_element_stride * S.mask_scanline_width;
        if(*S.mask_element == 0)
            advance_mask_element();
    }

    inline void update_pixel()
    {
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        assert(S.scanline_valid && S.mask_element_valid && *S.mask_element != 0);
        S.pixel_raw = S.scanline_raw + *mask_to_im_pixel_lut_element * S.pixel_stride;
        S.pixel_valid = true;
        im_to_mask_pixel_lut_element = im_to_mask_pixel_idx_lut->m_data.data() + *mask_to_im_pixel_lut_element;
    }

    inline void seek_front_pixel_of_scanline()
    {
        // Precondition: either seek_front_scanline or advance_scanline is the BitmapMaskCursor member function most 
        // recently called. Both leave us in the state where the mask element pointer and lut element pointer are 
        // correct, but the pixel pointer is not. Calling update_pixel fixes that. 
        update_pixel();
    }

    inline void advance_pixel()
    {
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        assert(S.pixel_valid && S.mask_element_valid);
        ++im_to_mask_pixel_lut_element;
        if(im_to_mask_pixel_lut_element > im_to_mask_pixel_lut_elements_end)
        {
            S.pixel_valid = false;
            S.mask_element_valid = false;
            S.component_valid = false;
            return;
        }
        if(prev_im_to_mask_pixel_lut_element_value != *im_to_mask_pixel_lut_element)
        {
            advance_mask_element();
            S.pixel_valid = S.mask_element_valid;
            if(likely(S.pixel_valid))
                update_pixel();
        }
        else
        {
//          normal stuff
        }
    }
};

template<typename BitmapMaskCursor, typename T, BitmapMaskDimensionVsImage T_H>
struct BitmapMaskCursorPixelAdvanceAspect<BitmapMaskCursor, T, BitmapMaskDimensionVsImage::Same, T_H>
{
    BitmapMaskCursorPixelAdvanceAspect(PyArrayView& data_view, BitmapMask<T, BitmapMaskDimensionVsImage::Same, T_H>& mask_) {}

    inline void seek_front_element_of_mask_scanline()
    {
    }
    inline void seek_front_pixel_of_scanline() {}
    inline void advance_mask_element()
    {
    }
    inline void advance_pixel() {}
};

template<typename BitmapMaskCursor, typename T, BitmapMaskDimensionVsImage T_H>
struct BitmapMaskCursorPixelAdvanceAspect<BitmapMaskCursor, T, BitmapMaskDimensionVsImage::Larger, T_H>
{
    const SampleLutPtr im_to_mask_pixel_idx_lut;
    const std::uint64_t* lut_element;

    BitmapMaskCursorPixelAdvanceAspect(PyArrayView& data_view, BitmapMask<T, BitmapMaskDimensionVsImage::Larger, T_H>& mask_)
      : im_to_mask_pixel_idx_lut(sampleLuts.getLut(data_view.shape[0], mask_.bitmap_view.shape[0])) {}

    inline void seek_front_element_of_mask_scanline()
    {
    }
    inline void seek_front_pixel_of_scanline() {}
    inline void advance_mask_element()
    {
    }
    inline void advance_pixel() {}
};
