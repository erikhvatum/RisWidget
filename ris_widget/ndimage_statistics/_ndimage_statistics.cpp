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

#include "_ndimage_statistics.h"
#include <string.h>

Luts luts{200};

void reorder_to_inner_outer(const std::size_t* u_shape, const std::size_t* u_strides,
                                  std::size_t* o_shape,       std::size_t* o_strides)
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

void reorder_to_inner_outer(const std::size_t* u_shape,       const std::size_t* u_strides,
                                  std::size_t* o_shape,             std::size_t* o_strides,
                            const std::size_t* u_slave_shape, const std::size_t* u_slave_strides,
                                  std::size_t* o_slave_shape,       std::size_t* o_slave_strides)
{
    // The u_strides[0] < u_strides[1] comparison controlling slave shape and striding reversal is not a typo: slave
    // striding and shape are reversed if non-slave striding and shape are reversed. 
    if(u_strides[0] >= u_strides[1])
    {
        o_strides[0] = u_strides[0]; o_strides[1] = u_strides[1];
        o_shape[0] = u_shape[0]; o_shape[1] = u_shape[1];
        o_slave_strides[0] = u_slave_strides[0]; o_slave_strides[1] = u_slave_strides[1];
        o_slave_shape[0] = u_slave_shape[0]; o_slave_shape[1] = u_slave_shape[1];
    }
    else
    {
        o_strides[0] = u_strides[1]; o_strides[1] = u_strides[0];
        o_shape[0] = u_shape[1]; o_shape[1] = u_shape[0];
        o_slave_strides[0] = u_slave_strides[1]; o_slave_strides[1] = u_slave_strides[0];
        o_slave_shape[0] = u_slave_shape[1]; o_slave_shape[1] = u_slave_shape[0];
    }
}

void reorder_to_inner_outer(const std::size_t* u_shape, const std::size_t* u_strides,
                                  std::size_t* o_shape,       std::size_t* o_strides,
                            const float& u_coord_0,     const float& u_coord_1,
                                  float& o_coord_0,           float& o_coord_1)
{
    if(u_strides[0] >= u_strides[1])
    {
        o_strides[0] = u_strides[0]; o_strides[1] = u_strides[1];
        o_shape[0] = u_shape[0]; o_shape[1] = u_shape[1];
        o_coord_0 = u_coord_0;
        o_coord_1 = u_coord_1;
    }
    else
    {
        o_strides[0] = u_strides[1]; o_strides[1] = u_strides[0];
        o_shape[0] = u_shape[1]; o_shape[1] = u_shape[0];
        o_coord_0 = u_coord_1;
        o_coord_1 = u_coord_0;
    }
}

#ifdef min
 #undef min
#endif

#include <algorithm>

#ifdef _WIN32
template<typename C>
const std::size_t bin_count()
{
    return 1024;
}
template<>
const std::size_t bin_count<std::uint8_t>()
{
    return 256;
}
#endif

template<>
const std::uint16_t apply_bin_shift<std::uint16_t, true>(const std::uint16_t& v)
{
    return std::min(v, static_cast<const std::uint16_t>(4095)) >> bin_shift<std::uint16_t, true>();
}
