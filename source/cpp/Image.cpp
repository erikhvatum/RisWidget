// The MIT License (MIT)
// 
// Copyright (c) 2014 WUSTL ZPLAB
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
// Authors: Erik Hvatum

#include "Common.h"
#include "GilStateScopeOperators.h"
#include "Image.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL RisWidget_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

PyObject* g_numpyLoadFunction;

extern "C" void freeImageErrorCallback(FREE_IMAGE_FORMAT fif, const char* msg)
{
    throw std::string(msg);
}

void initFreeImageErrorHandler()
{
    FreeImage_SetOutputMessage(freeImageErrorCallback);
}

void loadImage(const QImage& qimage, ImageData& imageData, QSize& imageSize)
{
    if(qimage.isNull())
    {
        imageData.clear();
        imageSize = QSize();
    }
    else
    {
        QImage rgbImage(qimage.convertToFormat(QImage::Format_RGB888));
        imageSize = rgbImage.size();
        imageData.resize(imageSize.width() * imageSize.height());
        const GLubyte* rgbIt{rgbImage.bits()};
        const GLubyte* rgbItE{rgbIt + imageData.size() * 3};
        for(GLushort* imageIt{imageData.data()}; rgbIt != rgbItE; ++imageIt, rgbIt += 3)
        {
            *imageIt = GLushort(256) * static_cast<GLushort>(0.2126f * rgbIt[0] + 0.7152f * rgbIt[1] + 0.0722f * rgbIt[2]);
        }
    }
}

void loadImage(PyObject* image, ImageData& imageData, QSize& imageSize)
{
    GilLocker gilLock;
    if(image == Py_None)
    {
        imageData.clear();
        imageSize = QSize();
    }
    else
    {
        PyArrayObject* imageao = reinterpret_cast<PyArrayObject*>(PyArray_FromAny(image, PyArray_DescrFromType(NPY_USHORT),
                                                                                  2, 2, NPY_ARRAY_CARRAY_RO, nullptr));
        if(imageao == nullptr)
        {
            throw RisWidgetException("RisWidget::showImage(PyObject* image): image argument must be an "
                                     "array-like object convertable to a 2d uint16 numpy array.");
        }
        npy_intp* shape = PyArray_DIMS(imageao);
        imageSize.setWidth(shape[1]);
        imageSize.setHeight(shape[0]);
        imageData.resize(imageSize.width() * imageSize.height());
        memcpy(imageData.data(), PyArray_DATA(imageao), imageData.size() * 2);
        Py_DECREF(imageao);
    }
}

void loadImage(const std::string& fileName, ImageData& imageData, QSize& imageSize)
{
    if(caseInsensitiveEndsWith(fileName, std::string(".npy")))
    {
        // File appears to be a numpy array
        std::string err;

        GilLocker gilLock;
        PyObject* fnpystr = PyUnicode_FromString(fileName.c_str());
        PyObject* image = PyObject_CallFunctionObjArgs(g_numpyLoadFunction, fnpystr, nullptr);
        if(image == nullptr)
        {
            PyObject *ptype(nullptr), *pvalue(nullptr), *ptraceback(nullptr);
            PyErr_Fetch(&ptype, &pvalue, &ptraceback);
            if(ptype != nullptr && pvalue != nullptr)
            {
                PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
                PyObject* epystr{PyObject_Str(pvalue)};
                err = PyUnicode_AsUTF8(epystr);
                Py_DECREF(epystr);
            }
            else
            {
                err = "(Failed to retrieve error information from Python.)";
            }
            Py_XDECREF(ptype);
            Py_XDECREF(pvalue);
            Py_XDECREF(ptraceback);
        }
        else
        {
            try
            {
                loadImage(image, imageData, imageSize);
            }
            catch(const RisWidgetException& e)
            {
                err = "Failed to convert data in \"" + fileName + "\" to a 2d uint16 numpy array.";
            }
        }
        Py_XDECREF(fnpystr);
        Py_XDECREF(image);

        if(!err.empty())
        {
            throw err;
        }
    }
    else
    {
        // File does not appear to be a numpy array; attempt to open it with FreeImage.  FreeImage does not call its
        // error notification callback or throw an exception if fipImage::load(..) fails for a mundane reason (file
        // doesn't exist, file permission error, file format not understood), so we have to do some extra checking upon
        // failure in order to provide error info other than "unknown freeimage oops happened, bursting into flames now"
        // in mundane failure modes
        fipImage image;
        if(image.load(fileName.c_str()))
        {
            if(!image.convertToUINT16())
            {
                throw std::string("Failed to convert \"") + fileName + "\" to grayscale uint16.";
            }
            if(!image.flipVertical())
            {
                throw std::string("Failed to flip \"") + fileName + "\".";
            }
            imageSize.setWidth(image.getWidth());
            imageSize.setHeight(image.getHeight());
            imageData.resize(imageSize.width() * imageSize.height());
            memcpy(imageData.data(), image.accessPixels(), imageSize.width() * imageSize.height() * 2);
        }
        else
        {
            QFile f(fileName.c_str());
            if(!f.exists())
            {
                throw std::string("\"") + fileName + "\" does not exist or is in an inaccessible directory.";
            }
            if(!f.open(QIODevice::ReadOnly))
            {
                throw std::string("\"") + fileName + "\" exists but can not be opened.";
            }
            f.close();
            if(fipImage::identifyFIF(fileName.c_str()) == FIF_UNKNOWN)
            {
                throw std::string("FreeImage failed to recognize the format of \"") + fileName + "\".";
            }
            throw std::string("Failed to open\"") + fileName + "\" for unknown reasons.";
        }
    }
}
