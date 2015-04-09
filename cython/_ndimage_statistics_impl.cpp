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

#include "_ndimage_statistics_impl.h"
#include <cstddef>
#include <string.h>

// Copies u_shape to o_shape and u_strides to o_strides, reversing the elements of each 
// if u_strides[0] < u_strides[1]
static void reorder_to_inner_outer(const Py_ssize_t* u_shape, const Py_ssize_t* u_strides,
                                   Py_ssize_t* o_shape, Py_ssize_t* o_strides)
{
    if(u_strides[0] >= u_strides[1])
    {
        o_strides[0] = u_strides[0]; o_strides[1] = u_strides[1];
        o_shape[0] = u_shape[0]; o_shape[1] = u_shape[1];
    }
    else
    {
        o_strides[0] = u_strides[1]; o_strides[1] = u_strides[0];
        o_shape[0] = u_shape[1]; o_shape[1] = u_shape[0];
    }
}

void _hist_min_max_uint16(const uint16_t* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
                          uint32_t* hist, uint16_t* min_max)
{
    Py_ssize_t shape[2], strides[2];
    reorder_to_inner_outer(im_shape, im_strides, shape, strides);

    memset(hist, 0, 1024 * sizeof(uint32_t));
    min_max[0] = min_max[1] = im[0];

    const uint8_t* outer = reinterpret_cast<const uint8_t*>(im);
    const uint8_t*const outer_end = outer + shape[0] * strides[0];
    const uint8_t* inner;
    const std::ptrdiff_t inner_end_offset = shape[1] * strides[1];
    const uint8_t* inner_end;
    for(; outer != outer_end; outer += strides[0])
    {
        inner = outer;
        inner_end = inner + inner_end_offset;
        for(; inner != inner_end; inner += strides[1])
        {
            const uint16_t& v = *reinterpret_cast<const uint16_t*>(inner);
            ++hist[v >> 6];
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

void _hist_min_max_uint12(const uint16_t* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
                          uint32_t* hist, uint16_t* min_max)
{
    Py_ssize_t shape[2], strides[2];
    reorder_to_inner_outer(im_shape, im_strides, shape, strides);

    memset(hist, 0, 1024 * sizeof(uint32_t));
    min_max[0] = min_max[1] = im[0];

    const uint8_t* outer = reinterpret_cast<const uint8_t*>(im);
    const uint8_t*const outer_end = outer + shape[0] * strides[0];
    const uint8_t* inner;
    const std::ptrdiff_t inner_end_offset = shape[1] * strides[1];
    const uint8_t* inner_end;
    for(; outer != outer_end; outer += strides[0])
    {
        inner = outer;
        inner_end = inner + inner_end_offset;
        for(; inner != inner_end; inner += strides[1])
        {
            const uint16_t& v = *reinterpret_cast<const uint16_t*>(inner);
            ++hist[v >> 2];
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

void _hist_min_max_uint8(const uint8_t* im, const Py_ssize_t* im_shape, const Py_ssize_t* im_strides,
                          uint32_t* hist, uint8_t* min_max)
{
    Py_ssize_t shape[2], strides[2];
    reorder_to_inner_outer(im_shape, im_strides, shape, strides);

    memset(hist, 0, 256 * sizeof(uint32_t));
    min_max[0] = min_max[1] = im[0];

    const uint8_t* outer = im;
    const uint8_t*const outer_end = outer + shape[0] * strides[0];
    const uint8_t* inner;
    const std::ptrdiff_t inner_end_offset = shape[1] * strides[1];
    const uint8_t* inner_end;
    for(; outer != outer_end; outer += strides[0])
    {
        inner = outer;
        inner_end = inner + inner_end_offset;
        for(; inner != inner_end; inner += strides[1])
        {
            const uint8_t& v = *inner;
            ++hist[v];
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
