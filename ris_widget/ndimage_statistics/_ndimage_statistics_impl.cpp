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

void reorder_to_inner_outer(const Py_ssize_t* u_shape, const Py_ssize_t* u_strides,
                                  Py_ssize_t* o_shape,       Py_ssize_t* o_strides)
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

void reorder_to_inner_outer(const Py_ssize_t* u_shape,       const Py_ssize_t* u_strides,
                                  Py_ssize_t* o_shape,             Py_ssize_t* o_strides,
                            const Py_ssize_t* u_slave_shape, const Py_ssize_t* u_slave_strides,
                                  Py_ssize_t* o_slave_shape,       Py_ssize_t* o_slave_strides)
{
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
