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
//
// Created by ehvatum on 10/11/16.
// 

#include "common.h"

Luts luts{256};

std::unordered_map<std::type_index, std::string> component_type_names {
    {std::type_index(typeid(std::int8_t)), "int8"},
    {std::type_index(typeid(std::uint8_t)), "uint8"},
    {std::type_index(typeid(std::int16_t)), "int16"},
    {std::type_index(typeid(std::uint16_t)), "uint16"},
    {std::type_index(typeid(std::int32_t)), "int32"},
    {std::type_index(typeid(std::uint32_t)), "uint32"},
    {std::type_index(typeid(std::int64_t)), "int64"},
    {std::type_index(typeid(std::uint64_t)), "uint64"},
    {std::type_index(typeid(float)), "float"},
    {std::type_index(typeid(double)), "double"},
};

template<>
std::uint16_t max_bin_count<std::uint8_t>()
{
    return 256;
}

template<>
std::uint16_t max_bin_count<std::int8_t>()
{
    return 256;
}

void safe_py_deleter(py::object* py_obj)
{
    py::gil_scoped_acquire acquire_gil;
    delete py_obj;
}
