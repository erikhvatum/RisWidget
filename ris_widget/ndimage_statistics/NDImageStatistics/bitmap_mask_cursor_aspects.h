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
// #include <iostream>
template<typename BitmapMaskCursor, typename T, BitmapMaskDimensionVsImage T_W>
struct BitmapMaskCursorScanlineAdvanceAspect<BitmapMaskCursor, T, T_W, BitmapMaskDimensionVsImage::Smaller>
{
    const SampleLutPtr im_to_mask_scanline_idx_lut;
    const std::uint64_t* im_to_mask_scanline_lut_element;
    const std::uint64_t*const im_to_mask_scanline_lut_elements_end;
    std::uint64_t prev_mask_scanline_idx;
    const std::uint8_t* prev_mask_scanline_front;
    const SampleLutPtr mask_to_im_scanline_idx_lut;
    const std::uint64_t* mask_to_im_scanline_lut_element;

    BitmapMaskCursorScanlineAdvanceAspect(PyArrayView& data_view, BitmapMask<T, T_W, BitmapMaskDimensionVsImage::Smaller>& mask_)
      : im_to_mask_scanline_idx_lut(sampleLuts.getLut(data_view.shape[1], mask_.bitmap_view->shape[1])),
        im_to_mask_scanline_lut_elements_end(im_to_mask_scanline_idx_lut->m_data.data() + data_view.shape[1]),
        mask_to_im_scanline_idx_lut(sampleLuts.getLut(mask_.bitmap_view->shape[1], data_view.shape[1])) {}

    inline void advance_mask_scanline()
    {
//      std::cout << "advance_mask_scanline" << std::endl;
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        assert(S.mask_scanline_valid);
        for(;;)
        {
            S.mask_scanline += S.mask_scanline_stride;
            S.mask_scanline_valid = S.mask_scanline < S.mask_scanlines_end;
            if(!S.mask_scanline_valid)
                break;
            ++mask_to_im_scanline_lut_element;
            S.seek_front_element_of_mask_scanline();
            if(S.mask_element_valid)
                break;
        }
    }

    inline void update_scanline()
    {
//      std::cout << "update_scanline" << std::endl;
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        assert(S.mask_scanline_valid);
        S.scanline_raw = reinterpret_cast<const std::uint8_t*>(S.scanlines_origin) + *mask_to_im_scanline_lut_element * S.scanline_stride;
        S.scanline_valid = true;
        im_to_mask_scanline_lut_element = im_to_mask_scanline_idx_lut->m_data.data() + *mask_to_im_scanline_lut_element;
        prev_mask_scanline_idx = *im_to_mask_scanline_lut_element;
        S.update_pixel();
    }

    inline void seek_front_scanline()
    {
//      std::cout << "seek_front_scanline" << std::endl;
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        mask_to_im_scanline_lut_element = mask_to_im_scanline_idx_lut->m_data.data();
        S.mask_scanline = S.mask_scanlines_origin;
        S.mask_scanline_valid = true;
        prev_mask_scanline_idx = 0;
        S.seek_front_element_of_mask_scanline();
        if(S.mask_element_valid)
        {
            update_scanline();
            prev_mask_scanline_front = S.mask_element;
        }
        else
        {
            advance_mask_scanline();
            if(S.mask_element_valid)
            {
                update_scanline();
                prev_mask_scanline_front = S.mask_element;
            }
        }
    }

    inline void advance_scanline()
    {
//      std::cout << "advance_scanline" << std::endl;
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        assert(S.scanline_valid && S.mask_scanline_valid);
        ++im_to_mask_scanline_lut_element;
        if(im_to_mask_scanline_lut_element < im_to_mask_scanline_lut_elements_end)
        {
            if(prev_mask_scanline_idx == *im_to_mask_scanline_lut_element)
            {
                S.scanline_raw += S.scanline_stride;
                assert(S.scanline_raw < S.scanlines_raw_end);
                S.mask_element = prev_mask_scanline_front;
                S.mask_element_valid = true;
            }
            else
            {
                advance_mask_scanline();
                S.scanline_valid = S.mask_scanline_valid;
                if(likely(S.scanline_valid)) // We are likely to see another scanline with a true value before the end of the image
                {
                    update_scanline();
                }
            }
        }
        else
        {
            S.mask_scanline_valid = false;
            S.mask_element_valid = false;
        }
        S.pixel_valid = false;
        S.component_valid = false;
    }
};

template<typename BitmapMaskCursor, typename T, BitmapMaskDimensionVsImage T_W>
struct BitmapMaskCursorScanlineAdvanceAspect<BitmapMaskCursor, T, T_W, BitmapMaskDimensionVsImage::Same>
{
    BitmapMaskCursorScanlineAdvanceAspect(PyArrayView& data_view, BitmapMask<T, T_W, BitmapMaskDimensionVsImage::Same>& mask_) {}

    inline void seek_front_scanline()
    {
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        S.mask_scanline = S.mask_scanlines_origin;
        S.mask_scanline_valid = true;
        S.seek_front_element_of_mask_scanline();
        S.scanline_raw = reinterpret_cast<const std::uint8_t*>(S.scanlines_origin);
        S.scanline_valid = true;
        if(!S.mask_element_valid)
            advance_scanline();
    }

    inline void advance_scanline()
    {
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        for(;;)
        {
            S.mask_scanline += S.mask_scanline_stride;
            S.scanline_valid = S.mask_scanline_valid = S.mask_scanline < S.mask_scanlines_end;
            if(likely(S.scanline_valid))
            {
                S.scanline_raw += S.scanline_stride;
                S.seek_front_element_of_mask_scanline();
                if(S.mask_element_valid)
                    break;
            }
            else
            {
                break;
            }
        }
        S.pixel_valid = false;
        S.component_valid = false;
    }
};

template<typename BitmapMaskCursor, typename T, BitmapMaskDimensionVsImage T_W>
struct BitmapMaskCursorScanlineAdvanceAspect<BitmapMaskCursor, T, T_W, BitmapMaskDimensionVsImage::Larger>
{
    const SampleLutPtr im_to_mask_scanline_idx_lut;
    const std::uint64_t* lut_element;

    BitmapMaskCursorScanlineAdvanceAspect(PyArrayView& data_view, BitmapMask<T, T_W, BitmapMaskDimensionVsImage::Larger>& mask_)
      : im_to_mask_scanline_idx_lut(sampleLuts.getLut(data_view.shape[1], mask_.bitmap_view->shape[1])),
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
    const SampleLutPtr im_to_mask_pixel_idx_lut;
    const std::uint64_t* im_to_mask_pixel_lut_element;
    const std::uint64_t*const im_to_mask_pixel_lut_elements_end;
    std::uint64_t prev_mask_element_idx;
    const SampleLutPtr mask_to_im_pixel_idx_lut;
    const std::uint64_t* mask_to_im_pixel_lut_element;
    const std::uint64_t*const mask_to_im_pixel_lut_elements_end;

    BitmapMaskCursorPixelAdvanceAspect(PyArrayView& data_view, BitmapMask<T, BitmapMaskDimensionVsImage::Smaller, T_H>& mask_)
      : im_to_mask_pixel_idx_lut(sampleLuts.getLut(data_view.shape[0], mask_.bitmap_view->shape[0])),
        im_to_mask_pixel_lut_elements_end(im_to_mask_pixel_idx_lut->m_data.data() + data_view.shape[0]),
        mask_to_im_pixel_idx_lut(sampleLuts.getLut(mask_.bitmap_view->shape[0], data_view.shape[0])),
        mask_to_im_pixel_lut_elements_end(mask_to_im_pixel_idx_lut->m_data.data() + mask_.bitmap_view->shape[0]) {}

    inline void advance_mask_element()
    {
//      std::cout << "advance_mask_element" << std::endl;
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        assert(S.mask_element_valid);
        for(;;)
        {
            S.mask_element += S.mask_element_stride;
            S.mask_element_valid = S.mask_element < S.mask_elements_end;
            ++mask_to_im_pixel_lut_element;
            if(!S.mask_element_valid || *S.mask_element != 0)
                break;
            assert(mask_to_im_pixel_lut_element < mask_to_im_pixel_lut_elements_end);
        }
    }

    // There is no corresponding BitmapCursorScanlineAdvanceAspect::seek_front_mask_scanline method, owing to the fact
    // that BitmapCursorScanlineAdvanceAspect handles scanning through mask scanlines consisting entirely of false
    // values, requiring seek_front_element_of_mask_scanline, whereas BitmapMaskCursorPixelAdvanceAspect never has a
    // need to scan to the first non-zero scanline.
    inline void seek_front_element_of_mask_scanline()
    {
//      std::cout << "seek_front_element_of_mask_scanline" << std::endl;
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        assert(S.mask_scanline_valid);
        mask_to_im_pixel_lut_element = mask_to_im_pixel_idx_lut->m_data.data();
        prev_mask_element_idx = 0;
        S.mask_element = S.mask_scanline;
        S.mask_element_valid = true;
        S.mask_elements_end = S.mask_element + S.mask_element_stride * S.mask_scanline_width;
        if(*S.mask_element == 0)
            advance_mask_element();
    }

    inline void update_pixel()
    {
//      std::cout << "update_pixel" << std::endl;
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
//      std::cout << S.scanline_valid << " : " << S.mask_element_valid << " : " << (*S.mask_element == 0 ? "false" : "true") << std::endl;
        assert(S.scanline_valid && S.mask_element_valid && *S.mask_element != 0);
        S.pixel_raw = S.scanline_raw + *mask_to_im_pixel_lut_element * S.pixel_stride;
        S.pixel_valid = true;
        assert(mask_to_im_pixel_lut_element < mask_to_im_pixel_lut_elements_end);
        im_to_mask_pixel_lut_element = im_to_mask_pixel_idx_lut->m_data.data() + *mask_to_im_pixel_lut_element;
        prev_mask_element_idx = *im_to_mask_pixel_lut_element;
    }

    inline void seek_front_pixel_of_scanline()
    {
//      std::cout << "seek_front_pixel_of_scanline" << std::endl;
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        S.pixels_raw_end = S.scanline_raw + S.pixel_stride * S.scanline_width;
        // Precondition: either seek_front_scanline or advance_scanline is the BitmapMaskCursor member function most
        // recently called. Both leave us in the state where the mask element pointer and lut element pointer are
        // correct, but the pixel pointer is not. Calling update_pixel fixes that.
        update_pixel();            
    }

    inline void advance_pixel()
    {
//      std::cout << "advance_pixel" << std::endl;
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        // Our being called can only mean we are looking at the current mask element because it is true; if the mask
        // element were false, the preceeding advance_pixel or seek_front_element_of_mask_scanline call would have left
        // pixel_valid false and caused the calling loop to terminate before calling advance_pixel.
        assert(S.pixel_valid && S.mask_element_valid && *S.mask_element != 0);
        ++im_to_mask_pixel_lut_element;
        if(im_to_mask_pixel_lut_element < im_to_mask_pixel_lut_elements_end)
        {
            if(prev_mask_element_idx == *im_to_mask_pixel_lut_element)
            {
                S.pixel_raw += S.pixel_stride;
                assert(S.pixel_raw < S.pixels_raw_end);
            }
            else
            {
                advance_mask_element();
                S.pixel_valid = S.mask_element_valid;
                if(likely(S.pixel_valid)) // We are likely to see another true mask value before the end of the scanline
                    update_pixel();
            }
        }
        else
        {
            S.pixel_valid = false;
            S.mask_element_valid = false;
        }
        S.component_valid = false;
    }
};

template<typename BitmapMaskCursor, typename T, BitmapMaskDimensionVsImage T_H>
struct BitmapMaskCursorPixelAdvanceAspect<BitmapMaskCursor, T, BitmapMaskDimensionVsImage::Same, T_H>
{
    std::uint64_t mask_element_idx;

    BitmapMaskCursorPixelAdvanceAspect(PyArrayView& data_view, BitmapMask<T, BitmapMaskDimensionVsImage::Same, T_H>& mask_) {}

    inline void advance_mask_element()
    {
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        assert(S.mask_element_valid);
        for(;;)
        {
            S.mask_element += S.mask_element_stride;
            S.mask_element_valid = S.mask_element < S.mask_elements_end;
            ++mask_element_idx;
            if(unlikely(!S.mask_element_valid) || *S.mask_element != 0)
                break;
        }
    }

    inline void seek_front_element_of_mask_scanline()
    {
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        assert(S.mask_scanline_valid);
        mask_element_idx = 0;
        S.mask_element = S.mask_scanline;
        S.mask_element_valid = true;
        S.mask_elements_end = S.mask_scanline + S.mask_scanline_width * S.mask_element_stride;
        if(*S.mask_element == 0)
            advance_mask_element();
    }

    inline void update_pixel()
    {
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        assert(S.mask_element_valid);
        S.pixel_raw = S.scanline_raw + mask_element_idx * S.pixel_stride;
        S.pixel_valid = true;
    }

    inline void seek_front_pixel_of_scanline()
    {
        update_pixel();
    }

    inline void advance_pixel()
    {
        BitmapMaskCursor& S = *static_cast<BitmapMaskCursor*>(this);
        assert(S.pixel_valid && S.mask_element_valid && *S.mask_element != 0);
        advance_mask_element();
        S.pixel_valid = S.mask_element_valid;
        if(likely(S.pixel_valid))
            update_pixel();
        S.component_valid = false;
    }
};

template<typename BitmapMaskCursor, typename T, BitmapMaskDimensionVsImage T_H>
struct BitmapMaskCursorPixelAdvanceAspect<BitmapMaskCursor, T, BitmapMaskDimensionVsImage::Larger, T_H>
{
    const SampleLutPtr im_to_mask_pixel_idx_lut;
    const std::uint64_t* lut_element;

    BitmapMaskCursorPixelAdvanceAspect(PyArrayView& data_view, BitmapMask<T, BitmapMaskDimensionVsImage::Larger, T_H>& mask_)
      : im_to_mask_pixel_idx_lut(sampleLuts.getLut(data_view.shape[0], mask_.bitmap_view->shape[0])) {}

    inline void seek_front_element_of_mask_scanline()
    {
    }
    inline void update_pixel() {}
    inline void seek_front_pixel_of_scanline() {}
    inline void advance_mask_element()
    {
    }
    inline void advance_pixel() {}
};
