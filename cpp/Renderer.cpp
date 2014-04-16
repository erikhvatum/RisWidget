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

#include "Common.h"
#include "HistogramView.h"
#include "ImageView.h"
#include "Renderer.h"

Renderer::Renderer(ImageView* imageView, HistogramView* histogramView)
  : m_thread(nullptr),
    m_lock(new QMutex(QMutex::Recursive)),
    m_imageView(nullptr),
    m_histogramView(nullptr),
    m_histoCalcProg("histoCalcProg"),
    m_histoConsolidateProg("histoConsolidateProg"),
    m_imageDrawProg("imageDrawProg"),
    m_histoDrawProg("histoDrawProg"),
    m_image(std::numeric_limits<GLuint>::max()),
    m_imageDims(std::numeric_limits<int>::min(), std::numeric_limits<int>::min()),
    m_imageIsStale(true),
    m_histogramIsStale(true),
    m_histogramDataStructuresAreStale(true),
    m_histogramBinCount(2048),
    m_histogramBlocks(std::numeric_limits<GLuint>::max()),
    m_histogram(std::numeric_limits<GLuint>::max()),
    m_histogramData(histogramBinCount, 0),
    m_histogramPmv(1.0f),
{
}

Renderer::~Renderer()
{
    delete m_lock;
}

void Renderer::attachViews(ImageView* imageView, HistogramView* histogramView)
{
    QMutexLocker locker(m_lock);
    if(m_thread)
    {
        throw RisWidgetException("void Renderer::attachViews(..): Called while views are already attached..");
    }

    m_imageView = imageView;
    m_histogramView = histogramView;

    m_thread = new QThread();

    moveToThread(m_thread);
    m_imageView->context()->moveToThread(m_thread);
    m_histogramView->context()->moveToThread(m_thread);

    m_threadStartedConnection = connect(this, &Renderer::_threadStarted, this, &Renderer::threadStartedSlot, Qt::QueuedConnection);
    m_newImageConnection = connect(this, &Renderer::_newImage, this, &Renderer::newImageSlot, Qt::QueuedConnection);
}

void Renderer::detachViews()
{
    QMutexLocker locker(m_lock);
    if(!m_thread)
    {
        throw RisWidgetException("void Renderer::detachViews(..): Called without views attached.");
    }
    disconnect(m_threadStartedConnection);
    disconnect(m_newImageConnection);
    QThread* gt = QApplication::instance()->thread();
    moveToThread(gt);
    m_imageView->context()->moveToThread(gt);
    m_imageView = nullptr;
    m_histogramView->context()->moveToThread(gt);
    m_histogramView = nullptr;
    m_thread->quit();
    m_thread->wait();
}

void Renderer::threadStartedSlot()
{
    QMutexLocker locker(m_lock);

    m_histoCalcProg.setView(histogramView);
    m_histoCalcProg.build();

    m_histoConsolidateProg.setView(histogramView);
    m_histoConsolidateProg.build();

    m_imageDrawProg.setView(imageView);
    m_imageDrawProg.build();

    m_histoDrawProg.setView(histogramView);
    m_histoDrawProg.build();
}

void Renderer::newImageSlot(ImageDataPtr imageDataPtr, QSize imageSize, bool filter)
{
}
