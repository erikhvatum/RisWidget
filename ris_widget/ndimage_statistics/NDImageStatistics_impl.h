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
#include "NDImageStatistics_decl.h"

template<typename T, typename MASK>
void NDImageStatistics<T, MASK>::expose_via_pybind11(py::module& m, const std::string& s)
{
    std::string name("NDImageStatistics");
    name += "_";
    name += s;
    py::class_<NDImageStatistics<T, MASK>, std::shared_ptr<NDImageStatistics<T, MASK>>>(m, name.c_str())
            .def(py::init<typed_array_t<T>&>())
            .def_readonly("data", &NDImageStatistics<T, MASK>::m_a);
    // Add overloaded "constructor" function.  pybind11 does not (yet, at time of writing) support templated class
    // instantiation via overloaded constructor defs, but plain function overloading is supported, and we take
    // advantage of this to present a factory function that is semantically similar.
    m.def("NDImageStatistics", [](typed_array_t<T>& a){return new NDImageStatistics<T, MASK>(a);});
}

template<typename T, typename MASK>
NDImageStatistics<T, MASK>::NDImageStatistics(typed_array_t<T>& a)
    : m_a(a)
{
}

template<typename T, typename MASK>
NDImageStatistics<T, MASK>::~NDImageStatistics()
{
}