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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _CppImage_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "_CppImage.h"
#include "Gil.h"
#include <FreeImage.h>
#include <iostream>

static void noop_deleter(void*) {}
static void freeimage_deleter(void* d)
{
    FreeImage_Unload((FIBITMAP*)d);
}
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
    m_mask_serial(generate_serial()),
    m_data(nullptr),
    m_mask(nullptr),
    m_dtype(DTypeNull),
    m_components(ComponentsNull)
{
    setObjectName(title);
    connect(this, &QObject::objectNameChanged, this, [&](const QString&){title_changed(this);});
}

//_CppImage::_CppImage(const QString& fpath, bool async, QObject* parent)
//  : QObject(parent),
//    m_status(Empty),

_CppImage::~_CppImage() {}

void _CppImage::read(const QString& fpath, bool async)
{
    // TODO: async
    QByteArray fpath_{fpath.toUtf8()};
    FREE_IMAGE_FORMAT fif{FreeImage_GetFileType(fpath_.data(), 0)};
    if(fif == FIF_UNKNOWN)
    {
        fif = FreeImage_GetFIFFromFilename(fpath_.data());
    }
    if(fif == FIF_UNKNOWN)
    {
        throw std::runtime_error("failed to open file or file type not recognized");
    }
    FIBITMAP* fibmp{FreeImage_Load(fif, fpath_.data())};
    if(!fibmp)
    {
        throw std::runtime_error("failed to read file");
    }
    if(!FreeImage_HasPixels(fibmp))
    {
        throw std::runtime_error("no pixel data");
    }
    FREE_IMAGE_TYPE fit{FreeImage_GetImageType(fibmp)};
    QString d;
    switch(fit)
    {
    case FIT_UNKNOWN:
        d = "FIT_UNKNOWN";
        break;
    case FIT_BITMAP:
        d = "FIT_BITMAP";
        break;
    case FIT_UINT16:
        d = "FIT_UINT16";
        break;
    case FIT_INT16:
        d = "FIT_INT16";
        break;
    case FIT_UINT32:
        d = "FIT_UINT32";
        break;
    case FIT_INT32:
        d = "FIT_INT32";
        break;
    case FIT_FLOAT:
        d = "FIT_FLOAT";
        break;
    case FIT_DOUBLE:
        d = "FIT_DOUBLE";
        break;
    case FIT_COMPLEX:
        d = "FIT_COMPLEX";
        break;
    case FIT_RGB16:
        d = "FIT_RGB16";
        break;
    case FIT_RGBA16:
        d = "FIT_RGBA16";
        break;
    case FIT_RGBF:
        d = "FIT_RGBF";
        break;
    case FIT_RGBAF:
        d = "FIT_RGBAF";
        break;
    default:
        d = "***ERRONEOUS***";
        break;
    }
    qDebug() << d;
    if(fit == FIT_UNKNOWN)
    {
        throw std::runtime_error("unsupported component data type or arrangement");
    }
    
    // Todo: store strides 
    // If the image data format and channel arrangement happen to be natively supported, 
    // a buffer copy is avoided.  This is accomplished by providing our new data smart 
    // pointer with a deleter that calls FreeImage_Unload(..) on the associated FIBITMAP 
    // pointer.  NB: That FIBITMAP pointer is copied into the deleter's lambda closure. 
//    RawData d()
    QSize new_size((int)FreeImage_GetWidth(fibmp), (int)FreeImage_GetHeight(fibmp));
}

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

const quint64& _CppImage::get_data_serial() const
{
    return m_data_serial;
}

const quint64& _CppImage::get_mask_serial() const
{
    return m_mask_serial;
}

PyObject* _CppImage::get_data()
{
    if(m_data) return m_data;
    Py_RETURN_NONE;
}

PyObject* _CppImage::get_mask()
{
    if(m_mask) return m_mask;
    Py_RETURN_NONE;
}

const _CppImage::DType& _CppImage::get_dtype() const
{
    return m_dtype;
}

const _CppImage::Components& _CppImage::get_components() const
{
    return m_components;
}

const QSize& _CppImage::get_size() const
{
    return m_size;
}

const QList<int>& _CppImage::get_strides() const
{
    return m_strides;
}