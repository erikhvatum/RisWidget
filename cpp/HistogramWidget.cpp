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
  : ViewWidget(parent),
    m_gtpEnabled(true),
    m_gtpMin(0),
    m_gtpMax(65535),
    m_gtpGamma(1.0f)
{
    setupUi(this);
}

HistogramWidget::~HistogramWidget()
{
}

HistogramView* HistogramWidget::histogramView()
{
    return dynamic_cast<HistogramView*>(m_view.data());
}

void HistogramWidget::makeView(bool /*doAddWidget*/)
{
    QMutexLocker locker(m_lock);
    ViewWidget::makeView(false);
    m_viewContainerWidget->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
    m_midHorizontalLayout->addWidget(m_viewContainerWidget);
    m_viewContainerWidget->show();
    m_view->show();
}

View* HistogramWidget::instantiateView()
{
    return new HistogramView(windowHandle());
}

void HistogramWidget::gtpGammaSliderValueChanged(int value)
{
}

void HistogramWidget::gtpMaxSliderValueChanged(int value)
{
    value = 65535 - value;
    {
        QMutexLocker locker(m_lock);
        m_gtpMax = static_cast<GLushort>(value);
    }
    m_gtpMaxEdit->setText(QString::number(value));
    if(value < m_gtpMinSlider->value())
    {
        m_gtpMinSlider->setValue(value);
    }
    m_view->update();
    gtpChanged();
}

void HistogramWidget::gtpMinSliderValueChanged(int value)
{
    {
        QMutexLocker locker(m_lock);
        m_gtpMin = static_cast<GLushort>(value);
    }
    m_gtpMinEdit->setText(QString::number(value));
    if(value > 65535 - m_gtpMaxSlider->value())
    {
        m_gtpMaxSlider->setValue(65535 - value);
    }
    m_view->update();
    gtpChanged();
}

void HistogramWidget::gtpGammaEditChanged()
{
}

void HistogramWidget::gtpMaxEditChanged()
{
    bool ok{false};
    int value = 65535 - m_gtpMaxEdit->text().toInt(&ok);
    if(ok)
    {
        m_gtpMaxSlider->setValue(value);
    }
}

void HistogramWidget::gtpMinEditChanged()
{
    bool ok{false};
    int value = m_gtpMinEdit->text().toInt(&ok);
    if(ok)
    {
        m_gtpMinSlider->setValue(value);
    }
}
