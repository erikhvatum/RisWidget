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
#include "HistogramWidget.h"
#include "ImageView.h"

HistogramWidget::HistogramWidget(QWidget* parent)
  : QWidget(parent),
    m_histogramViewHolder(nullptr),
    m_histogramView(nullptr)
{
    setupUi(this);
    if(layout() == nullptr)
    {
        QHBoxLayout* layout_(new QHBoxLayout);
        setLayout(layout_);
    }
}

HistogramWidget::~HistogramWidget()
{
}

HistogramView* HistogramWidget::histogramView()
{
    return m_histogramView;
}


void HistogramWidget::makeHistogramView(const QSurfaceFormat& format, const SharedGlObjectsPtr& sharedGlObjects, ImageView* imageView)
{
    if(m_histogramViewHolder != nullptr || m_histogramView != nullptr)
    {
        throw RisWidgetException("HistogramWidget::makeHistogramView(..): m_histogramViewHolder != nullptr || m_histogramView != nullptr.  "
                                 "HistogramWidget::makeHistogramView(..) must not be called more than once per "
                                 "HistogramWidget instance.");
    }
    m_histogramView = new HistogramView(format, sharedGlObjects, imageView);
    m_histogramViewHolder = QWidget::createWindowContainer(m_histogramView, this);
    layout()->addWidget(m_histogramViewHolder);
    m_histogramView->show();
}

View* HistogramWidget::instantiateView(const QSurfaceFormat& format)
{
    return new HistogramView(format);
}
