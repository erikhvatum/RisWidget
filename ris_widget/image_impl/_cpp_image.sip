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
// Authors: Erik Hvatum <ice.rikh@gmail.com

%Module _cpp_image

%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip
%Import QtOpenGL/QtOpenGLmod.sip
%Import QtWidgets/QtWidgetsmod.sip

%ModuleHeaderCode
    #include "_CppImage.h"
    #include <FreeImage.h>
    #include <algorithm>
    #include <QHash>
    #include <QString>
    extern "C" void freeimage_error_callback(FREE_IMAGE_FORMAT fif, const char* msg);
    void* do_import_array();
    extern QHash<PyObject*, _CppImage::DType> objectCppImageDTypes;
    // Contents match _CppImage::DType
    extern PyObject* cppImageDTypeObjects[_CppImage::DTYPE_COUNT];
    extern PyObject** cppImageDTypeObjectsEnd;
    extern QHash<QString, _CppImage::Components> stringCppImageComponents;
    // Contents match _CppImage::Components
    extern QString cppImageComponentStrings[_CppImage::COMPONENTS_COUNT];
    extern QString* cppImageComponentStringsEnd;
    extern QString cppImageStatusStrings[_CppImage::STATUS_COUNT];
%End

%ModuleCode
    #ifdef _DEBUG
     #undef _DEBUG
     #include <Python.h>
     #define _DEBUG
    #else
     #include <Python.h>
    #endif

    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    #define PY_ARRAY_UNIQUE_SYMBOL _CppImage_ARRAY_API
    #include <numpy/arrayobject.h>
    #include <iostream>

    void* do_import_array()
    {
        // import_array() is actually a macro that returns NULL if it fails, so it has to be wrapped in order to be called
        // from a constructor which necessarily does not return anything
        import_array();
        return reinterpret_cast<void*>(1);
    }

    extern "C" void freeimage_error_callback(FREE_IMAGE_FORMAT fif, const char* msg)
    {
        std::cerr << msg << '\n';
    }

    QHash<PyObject*, _CppImage::DType> objectCppImageDTypes;
    PyObject* cppImageDTypeObjects[_CppImage::DTYPE_COUNT];
    PyObject** cppImageDTypeObjectsEnd = cppImageDTypeObjects + _CppImage::DTYPE_COUNT;

    QHash<QString, _CppImage::Components> stringCppImageComponents;
    QString cppImageComponentStrings[_CppImage::COMPONENTS_COUNT] = {
        "",
        "G",
        "Ga",
        "rgb",
        "rgba"
    };
    QString* cppImageComponentStringsEnd = cppImageComponentStrings + _CppImage::COMPONENTS_COUNT;

    QString cppImageStatusStrings[_CppImage::STATUS_COUNT] = {
        "Empty",
        "AsyncLoading",
        "AsyncLoadingFailed",
        "Valid"
    };
%End

%PostInitialisationCode
    do_import_array();
    cppImageDTypeObjects[_CppImage::DTypeNull]    = (PyObject*)Py_None;
    Py_INCREF(Py_None);
    cppImageDTypeObjects[_CppImage::DTypeUInt8]   = (PyObject*)PyArray_TypeObjectFromType(NPY_UINT8);
    cppImageDTypeObjects[_CppImage::DTypeUInt12]  = (PyObject*)PyArray_TypeObjectFromType(NPY_UINT16);
    cppImageDTypeObjects[_CppImage::DTypeUInt16]  = (PyObject*)PyArray_TypeObjectFromType(NPY_UINT16);
    cppImageDTypeObjects[_CppImage::DTypeUInt32]  = (PyObject*)PyArray_TypeObjectFromType(NPY_UINT32);
//  cppImageDTypeObjects[_CppImage::DTypeUInt64]  = (PyObject*)PyArray_TypeObjectFromType(NPY_UINT64);
//  cppImageDTypeObjects[_CppImage::DTypeFloat16] = (PyObject*)PyArray_TypeObjectFromType(NPY_FLOAT16);
    cppImageDTypeObjects[_CppImage::DTypeFloat32] = (PyObject*)PyArray_TypeObjectFromType(NPY_FLOAT32);
//  cppImageDTypeObjects[_CppImage::DTypeFloat64] = (PyObject*)PyArray_TypeObjectFromType(NPY_FLOAT64);
    int e = 0;
    for(PyObject** cppImageDTypeObject{cppImageDTypeObjects};
        cppImageDTypeObject != cppImageDTypeObjectsEnd;
        ++cppImageDTypeObject, ++e)
    {
        objectCppImageDTypes[*cppImageDTypeObject] = (_CppImage::DType)e;
    }
    e = 0;
    for(QString* cppImageComponentString{cppImageComponentStrings};
        cppImageComponentString != cppImageComponentStringsEnd;
        ++cppImageComponentString, ++e)
    {
        stringCppImageComponents[*cppImageComponentString] = (_CppImage::Components)e;
    }
    FreeImage_SetOutputMessage(freeimage_error_callback);
%End

%Exception std::exception(SIP_Exception) /PyName=StdException/
{
    %TypeHeaderCode
    #include <exception>
    %End
    %RaiseCode
        const char *detail = sipExceptionRef.what();

        SIP_BLOCK_THREADS
        PyErr_SetString(sipException_std_exception, detail);
        SIP_UNBLOCK_THREADS
    %End
};

%DefaultDocstringFormat "deindented"

class _CppImage
  : public QObject
{
%TypeHeaderCode
    #include "_CppImage.h"
%End
public:
    enum SetFlag
    {
        SetData = 1,
        SetMask = 2,
        SetIsTwelveBit = 4,
        SetImposedFloatRange = 8,
        SetTitle = 16
    };
    explicit _CppImage(const QString& title=QString(), QObject* parent=nullptr);
    virtual ~_CppImage();

    virtual void read(const QString& fpath, bool async=true);

    QString get_title() const;
    void set_title(const QString& title);
    const Status& get_status() const;
    bool get_is_valid() const;
    const quint64& get_data_serial() const;
    const quint64& get_mask_serial() const;
    PyObject* get_data();
    PyObject* get_mask();
    const DType& get_dtype() const;
    const Components& get_components() const;
    const QSize& get_size() const;
    const QList<int> get_strides() const;

%Property(name=title, get=get_title, set=set_title)
%Property(name=status, get=get_status)
%Property(name=is_valid, get=get_is_valid)
%Property(name=data_serial, get=get_data_serial)
%Property(name=mask_serial, get=get_mask_serial)
%Property(name=dtype, get=get_dtype)
%Property(name=components, get=get_components)
%Property(name=size, get=get_size)
%Property(name=strides, get=get_strides)

signals:
    void title_changed(_CppImage*);
    void status_changed(_CppImage*);
    void is_valid_changed(_CppImage*);
    void data_serial_changed(_CppImage*);
    void mask_serial_changed(_CppImage*);
};

%MappedType _CppImage::DType
{
    %TypeHeaderCode
        #include "_CppImage.h"
    %End

    %ConvertToTypeCode
        if(!sipIsErr)
        {
            if(objectCppImageDTypes.contains(sipPy)) return 1;
            return 0;
        }
        *sipCppPtr = new _CppImage::DType(objectCppImageDTypes[sipPy]);
        return sipGetState(sipTransferObj);
    %End

    %ConvertFromTypeCode
        PyObject* r{cppImageDTypeObjects[*sipCpp]};
        Py_INCREF(r);
        return r;
    %End
};

%MappedType _CppImage::Components
{
    %TypeHeaderCode
        #include "_CppImage.h"
    %End

    %ConvertToTypeCode
        if(!sipIsErr)
        {
            if(sipPy == Py_None) return 1;
            if(PyUnicode_Check(sipPy))
            {
                Py_UCS4* str{PyUnicode_AsUCS4Copy(sipPy)};
                if(!str) return 0; // MemoryError
                bool r{stringCppImageComponents.contains(QString::fromUcs4(str))};
                PyMem_Free(str);
                if(r) return 1;
            }
            return 0;
        }
        if(sipPy == Py_None)
        {
            *sipCppPtr = new _CppImage::Components(_CppImage::ComponentsNull);
        }
        else
        {
            Py_UCS4* str{PyUnicode_AsUCS4Copy(sipPy)};
            if(!str) return 0; // MemoryError
            *sipCppPtr = new _CppImage::Components(stringCppImageComponents[QString::fromUcs4(str)]);
            PyMem_Free(str);
        }
        return sipGetState(sipTransferObj);
    %End

    %ConvertFromTypeCode
        if(*sipCpp == _CppImage::ComponentsNull)
        {
            Py_RETURN_NONE;
        }
        QByteArray str = cppImageComponentStrings[*sipCpp].toUtf8();
        return PyUnicode_FromString(str.data());
    %End
};

%MappedType _CppImage::Status
{
    %TypeHeaderCode
        #include "_CppImage.h"
    %End

    %ConvertToTypeCode
        if(!sipIsErr)
        {
            // There is never a need to go from, for example, "Empty" to _CppImage::Empty.  So, we return 0 
            // in all cases when asked if conversion to _CppImage::Status is possible.
            return 0;
        }
        return sipGetState(sipTransferObj);
    %End

    %ConvertFromTypeCode
        QByteArray str = cppImageStatusStrings[*sipCpp].toUtf8();
        return PyUnicode_FromString(str.data());
    %End
};
