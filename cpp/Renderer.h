// The MIT License (MIT)
//
// Copyright (c) 2014 Erik Hvatum
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

#pragma once

#include "Common.h"
#include "GlProgram.h"

class ImageView;
class HistogramView;

typedef std::vector<uint16_t> ImageData;
typedef std::shared_ptr<ImageData> ImageDataPtr;

class Renderer
  : public QObject
{
public:
    Renderer();
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator = (const Renderer&) = delete;

    void attachViews(ImageView* imageView, HistogramView* histogramView);
    void detachViews();

    // Supply nullptr for imageDataPtr to display nothing, removing current image if there is one
    void showImage(const ImageDataPtr& imageDataPtr, const QSize& imageSize, const bool& filter);

private:
    // The thread on which all OpenGL calls are made
    QPointer<QThread> m_thread;
    QMutex* m_lock;

    ImageView* m_imageView;
    HistogramView* m_histogramView;

    HistoCalcProg m_histoCalcProg;
    HistoConsolidateProg m_histoConsolidateProg;
    ImageDrawProg m_imageDrawProg;
    HistoDrawProg m_histoDrawProg;

    // Raw image data to be loaded into m_image upon next rendering
    ImageDataPtr m_imageDataPtr;
    // Dimensions of image represented by m_imageDataPtr
    QSize m_imageDataPtrDims;

    // Image texture
    GLuint m_image;
    // Dimensions of image texture
    QSize m_imageDims;
    // Aspect ratio of image texture
    float m_imageAspectRatio;

    bool m_imageIsStale;
    bool m_histogramIsStale;
    bool m_histogramDataStructuresAreStale;
    const std::uint32_t m_histogramBinCount;
    GLuint m_histogramBlocks;
    GLuint m_histogram;
    std::vector<std::uint32_t> m_histogramData;

signals:
    void _newImage(ImageDataPtr imageDataPtr, QSize imageSize, bool filter);
    void _threadStarted();

private slots:
    void newImageSlot(ImageDataPtr imageDataPtr, QSize imageSize, bool filter);
    void threadStartedSlot();

private:
    QMetaObject::Connection m_newImageConnection;
    QMetaObject::Connection m_threadStartedConnection;
};
