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

#include <limits>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _CppImage_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "_CppImage.h"
#include "Gil.h"

static void noop_deleter(void*) {}
using get_default_deleter = std::default_delete<std::uint8_t[]>;

volatile std::atomic<quint64> _CppImage::sm_next_serial{0};

quint64 _CppImage::generate_serial()
{
    return sm_next_serial++;
}

_CppImage::_CppImage(const QString& title, QObject* parent)
  : QObject(parent),
    m_status(Empty),
    m_data_serial(generate_serial()),
    m_mask_serial(generate_serial())
{
    setObjectName(title);
    connect(this, &QObject::objectNameChanged, this, [&](const QString&){title_changed(this);});
}

_CppImage::~_CppImage() {}

QString _CppImage::get_title() const
{
    return objectName();
}

void _CppImage::set_title(const QString& title)
{
    setObjectName(title);
}

const _CppImage::Status& _CppImage::get_status() const
{
    return m_status;
}

bool _CppImage::get_is_valid() const
{
    return m_status == Valid;
}

const std::uint64_t& _CppImage::get_data_serial() const
{
    return m_data_serial;
}

const std::uint64_t& _CppImage::get_mask_serial() const
{
    return m_mask_serial;
}