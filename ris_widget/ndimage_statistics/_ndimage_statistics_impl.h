// The MIT License (MIT)
//
// Copyright (c) 2015 WUSTL ZPLAB
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

// Copies u_shape to o_shape and u_strides to o_strides, reversing the elements of each if u_strides[0] < u_strides[1] 
void reorder_to_inner_outer(const Py_ssize_t* u_shape, const Py_ssize_t* u_strides,
                                  Py_ssize_t* o_shape,       Py_ssize_t* o_strides);

// Copies u_shape to o_shape and u_strides to o_strides, reversing the elements of each if u_strides[0] < u_strides[1]. 
// Additionally, u_slave_shape is copied to o_slave_shape and u_slave_strides is copied to o_slave_strides, reversing 
// the elements of each if u_strides[0] < u_strides[1]. 
// 
// The u_strides[0] < u_strides[1] comparison controlling slave shape and striding reversal is not a typo: slave
// striding and shape are reversed if non-slave striding and shape are reversed. 
void reorder_to_inner_outer(const Py_ssize_t* u_shape,       const Py_ssize_t* u_strides,
                                  Py_ssize_t* o_shape,             Py_ssize_t* o_strides,
                            const Py_ssize_t* u_slave_shape, const Py_ssize_t* u_slave_strides,
                                  Py_ssize_t* o_slave_shape,       Py_ssize_t* o_slave_strides);

template<typename C>
constexpr std::size_t bin_count();

template<>
constexpr std::size_t bin_count<npy_uint8>()
{
    return 256;
}

template<>
constexpr std::size_t bin_count<npy_uint16>()
{
    return 1024;
}

template<typename C, bool is_twelve_bit>
constexpr const std::ptrdiff_t bin_shift()
{
    return (is_twelve_bit ? 12 : sizeof(C)*8) - static_cast<std::ptrdiff_t>( std::log2(static_cast<double>(bin_count<C>())) );
}

template<typename C, bool is_twelve_bit>
const C apply_bin_shift(const C& v)
{
    return v >> bin_shift<C, is_twelve_bit>();
}

template<>
const npy_uint16 apply_bin_shift<npy_uint16, true>(const npy_uint16& v);

template<typename C, bool is_twelve_bit>
void _hist_min_max(const C* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
                   npy_uint32* hist, C* min_max)
{
    Py_ssize_t shape[2], strides[2];
    reorder_to_inner_outer(im_shape, im_strides, shape, strides);

    memset(hist, 0, bin_count<C>() * sizeof(npy_uint32));
    min_max[0] = min_max[1] = im[0];

    const npy_uint8* outer = reinterpret_cast<const npy_uint8*>(im);
    const npy_uint8*const outer_end = outer + shape[0] * strides[0];
    const npy_uint8* inner;
    const std::ptrdiff_t inner_end_offset = shape[1] * strides[1];
    const npy_uint8* inner_end;
    for(; outer != outer_end; outer += strides[0])
    {
        inner = outer;
        inner_end = inner + inner_end_offset;
        for(; inner != inner_end; inner += strides[1])
        {
            const C& v = *reinterpret_cast<const C*>(inner);
            ++hist[apply_bin_shift<C, is_twelve_bit>(v)];
            if(v < min_max[0])
            {
                min_max[0] = v;
            }
            else if(v > min_max[1])
            {
                min_max[1] = v;
            }
        }
    }
}

template<typename C, bool is_twelve_bit>
void _masked_hist_min_max(const C* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
                          const npy_uint8* mask, const Py_ssize_t* mask_shape, const Py_ssize_t* mask_strides,
                          npy_uint32* hist, C* min_max)
{
    Py_ssize_t shape[2], strides[2], mshape[2], mstrides[2];
    reorder_to_inner_outer(im_shape, im_strides, shape, strides,
                           mask_shape, mask_strides, mshape, mstrides);
    // At this point, it should be true that shape == mshape.  Our caller is expected to have verified 
    // that this would be the case (ie, the caller of this function  

    memset(hist, 0, bin_count<C>() * sizeof(npy_uint32));
    min_max[0] = min_max[1] = im[0];

    const npy_uint8* outer = reinterpret_cast<const npy_uint8*>(im);
    const npy_uint8* mouter = mask;
    const npy_uint8*const outer_end = outer + shape[0] * strides[0];
    const npy_uint8* inner;
    const npy_uint8* minner;
    const std::ptrdiff_t inner_end_offset = shape[1] * strides[1];
    const npy_uint8* inner_end;
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
                if(v < min_max[0])
                {
                    min_max[0] = v;
                }
                else if(v > min_max[1])
                {
                    min_max[1] = v;
                }
            }
        }
    }
}
