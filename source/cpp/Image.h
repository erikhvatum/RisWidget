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

#pragma once

#include "Common.h"

extern PyObject* g_numpyLoadFunction;

void initFreeImageErrorHandler();

// Note that QVector<> does implicit sharing with reference counting and copy-on-write:
// http://qt-project.org/doc/qt-5/qvector.html#QVector-4
typedef QVector<GLushort> ImageData;

// Convert qimage to uint16 grayscale, storing image size in imageSize and image data in imageData.  If qimage is empty
// (0 width/height), imageData is cleared and imageSize is set to invalid (imageSize.isNull() will return true).
void loadImage(const QImage& qimage, ImageData& imageData, QSize& imageSize);

// Attempt to create a uint16 numpy view of image and copy the resulting data into imageData.  If image is None,
// imageData is cleared and imageSize is set to invalid (imageSize.isNull() will return true).
void loadImage(PyObject* image, ImageData& imageData, QSize& imageSize);

// Attempt to load image data from file specified by fileName.  fileName may refer to a numpy data file (.npy) or any
// image format supported by freeimage.  Throws std::string describing error if file could not be opened or understood.
void loadImage(const std::string& fileName, ImageData& imageData, QSize& imageSize);
