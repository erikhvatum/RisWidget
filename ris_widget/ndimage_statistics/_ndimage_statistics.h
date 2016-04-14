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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

#include "resampling_lut.h"

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

template<typename C>
void min_max(const C* im, const std::size_t* im_shape, const std::size_t* im_strides,
             C* min_max, const std::size_t& min_max_stride)
{
    std::size_t shape[2], strides[2];
    reorder_to_inner_outer(im_shape, im_strides, shape, strides);

    C& min{min_max[0]};
    C& max{*reinterpret_cast<C*>(reinterpret_cast<std::uint8_t*>(min_max) + min_max_stride)};
    max = min = im[0];

    const std::uint8_t* outer = reinterpret_cast<const std::uint8_t*>(im);
    const std::uint8_t*const outer_end = outer + shape[0] * strides[0];
    const std::uint8_t* inner;
    const std::ptrdiff_t inner_end_offset = shape[1] * strides[1];
    const std::uint8_t* inner_end;
    for(; outer != outer_end; outer += strides[0])
    {
        inner = outer;
        inner_end = inner + inner_end_offset;
        for(; inner != inner_end; inner += strides[1])
        {
            const C& v = *reinterpret_cast<const C*>(inner);
            if(v < min)
            {
                min = v;
            }
            else if(v > max)
            {
                max = v;
            }
        }
    }
}

template<typename C>
void masked_min_max(const C* im, const std::size_t* im_shape, const std::size_t* im_strides,
                    const std::uint8_t* mask, const std::size_t* mask_shape, const std::size_t* mask_strides,
                    C* min_max, const std::size_t& min_max_stride)
{
    std::size_t shape[2], strides[2], mshape[2], mstrides[2];
    reorder_to_inner_outer(im_shape, im_strides, shape, strides,
                           mask_shape, mask_strides, mshape, mstrides);

    C& min{min_max[0]};
    C& max{*reinterpret_cast<C*>(reinterpret_cast<std::uint8_t*>(min_max) + min_max_stride)};

    bool seen_unmasked = false;
    const std::uint8_t* outer = reinterpret_cast<const std::uint8_t*>(im);
    const std::uint8_t*const outer_end = outer + shape[0] * strides[0];
    const std::uint8_t* inner;
    const std::ptrdiff_t inner_end_offset = shape[1] * strides[1];
    const std::uint8_t* inner_end;
    const std::uint8_t* mouter = mask;
    const std::uint8_t* minner;
    if(im_shape[0] == mask_shape[0] && im_shape[1] == mask_shape[1])
    {
        for(; outer != outer_end; outer += strides[0], mouter += mstrides[0])
        {
            inner = outer;
            inner_end = inner + inner_end_offset;
            minner = mouter;
            for(; inner != inner_end; inner += strides[1], minner += mstrides[1])
            {
                const C& v = *reinterpret_cast<const C*>(inner);
                if(*minner != 0)
                {
                    if(seen_unmasked)
                    {
                        if(v < min)
                        {
                            min = v;
                        }
                        else if(v > max)
                        {
                            max = v;
                        }
                    }
                    else
                    {
                        seen_unmasked = true;
                        min = max = v;
                    }
                }
            }
        }
    }
    else
    {
        LutPtr mouter_lut_obj{luts.getLut(shape[0], mshape[0])};
        const std::uint32_t* mouter_lut{mouter_lut_obj->m_data.data()};
        std::uint32_t mouter_lut_val, mouter_prev_lut_val = 0;
        LutPtr minner_lut_obj{luts.getLut(shape[1], mshape[1])};
        const std::uint32_t* minner_lut;
        std::uint32_t minner_lut_val, minner_prev_lut_val;
        for(; outer != outer_end; outer += strides[0], ++mouter_lut)
        {
            mouter_lut_val = *mouter_lut;
            mouter += mstrides[0] * (mouter_lut_val - mouter_prev_lut_val);
            mouter_prev_lut_val = mouter_lut_val;
            inner = outer;
            inner_end = inner + inner_end_offset;
            minner = mouter;
            minner_lut = minner_lut_obj->m_data.data();
            minner_prev_lut_val = 0;
            for(; inner != inner_end; inner += strides[1], ++minner_lut)
            {
                minner_lut_val = *minner_lut;
                minner += mstrides[1] * (minner_lut_val - minner_prev_lut_val);
                minner_prev_lut_val = minner_lut_val;
                const C& v = *reinterpret_cast<const C*>(inner);
                if(*minner != 0)
                {
                    if(seen_unmasked)
                    {
                        if(v < min)
                        {
                            min = v;
                        }
                        else if(v > max)
                        {
                            max = v;
                        }
                    }
                    else
                    {
                        seen_unmasked = true;
                        min = max = v;
                    }
                }
            }
        }
    }
}

template<typename C, bool with_overflow_bins>
void ranged_hist(const C* im, const std::size_t* im_shape, const std::size_t* im_strides,
                 const C* range, const std::size_t& range_stride,
                 const std::size_t& bin_count,
                 std::uint32_t* hist, const std::size_t& hist_stride)
{
    std::size_t shape[2], strides[2];
    reorder_to_inner_outer(im_shape, im_strides, shape, strides);

    std::uint8_t* hist8{reinterpret_cast<std::uint8_t*>(hist)};
    for(std::uint8_t *hist8It{hist8}, *const hist8EndIt{hist8 + bin_count * hist_stride}; hist8It != hist8EndIt; hist8It += hist_stride)
    {
        *reinterpret_cast<std::uint32_t*>(hist8It) = 0;
    }

    const C range_min{range[0]};
    const C range_max{*reinterpret_cast<const C*>(reinterpret_cast<const std::uint8_t*>(range) + range_stride)};

    const C range_width = range_max - range_min;
    const std::size_t non_overflow_bin_count = with_overflow_bins ? bin_count - 2 : bin_count;
    const double bin_factor = static_cast<double>(non_overflow_bin_count - 1) / range_width;
    std::uint32_t*const last_bin = reinterpret_cast<std::uint32_t*>(hist8 + bin_count * hist_stride);
    const std::uint8_t* outer = reinterpret_cast<const std::uint8_t*>(im);
    const std::uint8_t*const outer_end = outer + shape[0] * strides[0];
    const std::uint8_t* inner;
    const std::ptrdiff_t inner_end_offset = shape[1] * strides[1];
    const std::uint8_t* inner_end;
    for(; outer != outer_end; outer += strides[0])
    {
        inner = outer;
        inner_end = inner + inner_end_offset;
        for(; inner != inner_end; inner += strides[1])
        {
            const C& v = *reinterpret_cast<const C*>(inner);
            if(with_overflow_bins)
            {
                if(v < range_min)
                {
                    ++*hist;
                }
                else if(v > range_max)
                {
                    ++*last_bin;
                }
                else
                {
                    ++*reinterpret_cast<C*>(hist8 + (1 + static_cast<std::ptrdiff_t>( bin_factor * (v - range_min) ))*hist_stride);
                }
            }
            else
            {
                if(v >= range_min && v <= range_max)
                {
                    ++hist[static_cast<std::ptrdiff_t>( bin_factor * (v - range_min) )];
                }
            }
        }
    }
}

template<typename C, bool with_overflow_bins>
void masked_ranged_hist(const C* im, const std::size_t* im_shape, const std::size_t* im_strides,
                        const std::uint8_t* mask, const std::size_t* mask_shape, const std::size_t* mask_strides,
                        const C* range, const std::size_t& range_stride,
                        const std::size_t& bin_count,
                        std::uint32_t* hist, const std::size_t& hist_stride)
{
//  std::size_t shape[2], strides[2], mshape[2], mstrides[2];
//  reorder_to_inner_outer(im_shape, im_strides, shape, strides,
//                         mask_shape, mask_strides, mshape, mstrides);
// 
//  std::uint8_t* hist8{reinterpret_cast<std::uint8_t*>(hist)};
//  for(std::uint8_t *hist8It{hist8}, *const hist8EndIt{hist8 + bin_count * hist_stride}; hist8It != hist8EndIt; hist8It += hist_stride)
//  {
//      *reinterpret_cast<std::uint32_t*>(hist8It) = 0;
//  }
// 
//  const C range_width = range_max - range_min;
//  const std::size_t non_overflow_bin_count = with_overflow_bins ? bin_count - 2 : bin_count;
//  const float bin_factor = static_cast<float>(non_overflow_bin_count - 1) / range_width;
//  std::uint32_t*const last_bin = reinterpret_cast<std::uint32_t*>(hist8 + bin_count * hist_stride);
//  const std::uint8_t* outer = reinterpret_cast<const std::uint8_t*>(im);
//  const std::uint8_t* mouter = mask;
//  const std::uint8_t*const outer_end = outer + shape[0] * strides[0];
//  const std::uint8_t* inner;
//  const std::uint8_t* minner;
//  const std::ptrdiff_t inner_end_offset = shape[1] * strides[1];
//  const std::uint8_t* inner_end;
//  if(im_shape[0] == mask_shape[0] && im_shape[1] == mask_shape[1])
//  {
//      for(; outer != outer_end; outer += strides[0], mouter += mstrides[0])
//      {
//          inner = outer;
//          inner_end = inner + inner_end_offset;
//          minner = mouter;
//          for(; inner != inner_end; inner += strides[1], minner += mstrides[1])
//          {
//              if(*minner != 0)
//              {
//                  const C& v = *reinterpret_cast<const C*>(inner);
//                  if(with_overflow_bins)
//                  {
//                      if(v < range_min)
//                      {
//                          ++*hist;
//                      }
//                      else if(v > range_max)
//                      {
//                          ++*last_bin;
//                      }
//                      else
//                      {
//                          ++hist[1 + static_cast<std::ptrdiff_t>( bin_factor * (v - range_min) )];
//                      }
//                  }
//                  else
//                  {
//                      if(v >= range_min && v <= range_max)
//                      {
//                          ++hist[static_cast<std::ptrdiff_t>( bin_factor * (v - range_min) )];
//                      }
//                  }
//              }
//          }
//      }
//  }
}

#ifdef _WIN32
// Inside of the Microsoft ghetto, we must do without constexpr, the sun never shines, the year is forever 1998,
// and the only sure thing in life is that Microsoft will take your money.
template<typename C>
const std::size_t bin_count();

template<>
const std::size_t bin_count<std::uint8_t>();

template<typename C, bool is_twelve_bit>
const std::ptrdiff_t bin_shift()
{
    return (is_twelve_bit ? 12 : sizeof(C)*8) - static_cast<std::ptrdiff_t>( std::log2(static_cast<double>(bin_count<C>())) );
}
#else
// Outside of the Microsoft ghetto, humanity has transcended all limitations and explores the stars at warp speed,
// sharing the gifts of justice and knowledge without expectation of reward.
template<typename C>
constexpr std::size_t bin_count()
{
    return 1024;
}

template<>
constexpr std::size_t bin_count<std::uint8_t>()
{
    return 256;
}

template<typename C, bool is_twelve_bit>
constexpr const std::ptrdiff_t bin_shift()
{
    return (is_twelve_bit ? 12 : sizeof(C)*8) - static_cast<std::ptrdiff_t>( std::log2(static_cast<double>(bin_count<C>())) );
}
#endif

template<typename C, bool is_twelve_bit>
const C apply_bin_shift(const C& v)
{
    return v >> bin_shift<C, is_twelve_bit>();
}

// This specialization guards against the case where a putative uint12-in-uint16 image has at least one element with a non-zero
// high nibble.
template<>
const std::uint16_t apply_bin_shift<std::uint16_t, true>(const std::uint16_t& v);

template<typename C, bool is_twelve_bit>
void hist_min_max(const C* im, const std::size_t* im_shape, const std::size_t* im_strides,
                  std::uint32_t* hist, const std::size_t& hist_stride,
                  C* min_max, const std::size_t& min_max_stride)
{
    std::size_t shape[2], strides[2];
    reorder_to_inner_outer(im_shape, im_strides, shape, strides);

    std::uint8_t* hist8{reinterpret_cast<std::uint8_t*>(hist)};
    for(std::uint8_t *hist8It{hist8}, *const hist8EndIt{hist8 + bin_count<C>() * hist_stride}; hist8It != hist8EndIt; hist8It += hist_stride)
    {
        *reinterpret_cast<std::uint32_t*>(hist8It) = 0;
    }

    C& min{min_max[0]};
    C& max{*reinterpret_cast<C*>(reinterpret_cast<std::uint8_t*>(min_max) + min_max_stride)};
    max = min = im[0];

    const std::uint8_t* outer = reinterpret_cast<const std::uint8_t*>(im);
    const std::uint8_t*const outer_end = outer + shape[0] * strides[0];
    const std::uint8_t* inner;
    const std::ptrdiff_t inner_end_offset = shape[1] * strides[1];
    const std::uint8_t* inner_end;
    for(; outer != outer_end; outer += strides[0])
    {
        inner = outer;
        inner_end = inner + inner_end_offset;
        for(; inner != inner_end; inner += strides[1])
        {
            const C& v = *reinterpret_cast<const C*>(inner);
            ++*reinterpret_cast<std::uint32_t*>(hist8 + hist_stride*apply_bin_shift<C, is_twelve_bit>(v));
            if(v < min)
            {
                min = v;
            }
            else if(v > max)
            {
                max = v;
            }
        }
    }
}

template<typename C, bool is_twelve_bit>
void masked_hist_min_max(const C* im, const std::size_t* im_shape, const std::size_t* im_strides,
                         const std::uint8_t* mask, const std::size_t* mask_shape, const std::size_t* mask_strides,
                         std::uint32_t* hist, const std::size_t& hist_stride,
                         C* min_max, const std::size_t& min_max_stride)
{
//  std::size_t shape[2], strides[2], mshape[2], mstrides[2];
//  reorder_to_inner_outer(im_shape, im_strides, shape, strides,
//                         mask_shape, mask_strides, mshape, mstrides);
//  // At this point, it should be true that shape == mshape.  Our caller is expected to have verified
//  // that this is the case.
// 
//  memset(hist, 0, bin_count<C>() * sizeof(std::uint32_t));
//  min_max[0] = min_max[1] = 0;
//  bool seen_unmasked = false;
// 
//  const std::uint8_t* outer = reinterpret_cast<const std::uint8_t*>(im);
//  const std::uint8_t* mouter = mask;
//  const std::uint8_t*const outer_end = outer + shape[0] * strides[0];
//  const std::uint8_t* inner;
//  const std::uint8_t* minner;
//  const std::ptrdiff_t inner_end_offset = shape[1] * strides[1];
//  const std::uint8_t* inner_end;
//  for(; outer != outer_end; outer += strides[0], mouter += mstrides[0])
//  {
//      inner = outer;
//      inner_end = inner + inner_end_offset;
//      minner = mouter;
//      for(; inner != inner_end; inner += strides[1], minner += mstrides[1])
//      {
//          if(*minner != 0)
//          {
//              const C& v = *reinterpret_cast<const C*>(inner);
//              ++hist[apply_bin_shift<C, is_twelve_bit>(v)];
//              if(seen_unmasked)
//              {
//                  if(v < min_max[0])
//                  {
//                      min_max[0] = v;
//                  }
//                  else if(v > min_max[1])
//                  {
//                      min_max[1] = v;
//                  }
//              }
//              else
//              {
//                  seen_unmasked = true;
//                  min_max[1] = min_max[0] = v;
//              }
//          }
//      }
//  }
}
