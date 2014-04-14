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
#include "ImageWidget.h"

ImageWidget::ImageWidget(QWidget* parent)
  : QWidget(parent),
    m_imageViewHolder(nullptr),
    m_imageView(nullptr)
{
    setupUi(this);
    if(layout() == nullptr)
    {
        QHBoxLayout* layout_(new QHBoxLayout);
        setLayout(layout_);
    }
}

ImageWidget::~ImageWidget()
{
}

ImageView* ImageWidget::imageView()
{
    return m_imageView;
}

void ImageWidget::makeImageView(const QSurfaceFormat& format, const SharedGlObjectsPtr& sharedGlObjects)
{
    if(m_imageViewHolder != nullptr || m_imageView != nullptr)
    {
        throw RisWidgetException("ImageWidget::makeImageView(..): m_imageViewHolder != nullptr || m_imageView != nullptr.  "
                                 "ImageWidget::makeImageView(..) must not be called more than once per "
                                 "ImageWidget instance.");
    }
    m_imageView = new ImageView(format, sharedGlObjects);
    m_imageViewHolder = QWidget::createWindowContainer(m_imageView, this);
    layout()->addWidget(m_imageViewHolder);
    m_imageViewHolder->show();
    m_imageView->show();
}
