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

#include "_cpp_image.h"

PYBIND11_PLUGIN(_cpp_image)
{
    py::module m("_cpp_image", "ris_widget.image_impl._cpp_image module");

    py::class_<_CppImage, std::shared_ptr<_CppImage>>(m, "_CppImage")
        .def(py::init<const char*>());

    py::enum_<ImageStatus>(m, "ImageStatus")
        .value("Empty", ImageStatus::Empty)
        .value("AsyncLoading", ImageStatus::AsyncLoading)
        .value("AsyncLoadingFailed", ImageStatus::AsyncLoadingFailed)
        .value("Valid", ImageStatus::Valid);

    return m.ptr();
}
