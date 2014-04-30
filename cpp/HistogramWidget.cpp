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
    m_imageLoaded(false),
    m_gtpEnabled(true),
    m_gtpAutoMinMaxEnabled(false),
    m_gtpMin(0),
    m_gtpMax(65535),
    m_gtpGamma(1.0f),
    m_imageExtremaValid(false)
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

void HistogramWidget::setGtpEnabled(bool gtpEnabled)
{
    // A little indirect, but using the gui infrastructure as our gtp parameter modification clearing house is
    // convenient with the caveat that gtp parameter calls must originate from the thread owning this HistogramWidget
    // instance.  Renderer of course runs on a different thread and needs these values, but it only reads them
    m_gtpEnableCheckBox->setChecked(gtpEnabled);
}

void HistogramWidget::setGtpAutoMinMax(bool gtpAutoMinMax)
{
    m_gtpAutoMinMaxCheckBox->setChecked(gtpAutoMinMax);
}

void HistogramWidget::setGtpMin(GLushort gtpMin)
{
    m_gtpMinSlider->setValue(65535 - gtpMin);
}

void HistogramWidget::setGtpMax(GLushort gtpMax)
{
    m_gtpMaxSlider->setValue(gtpMax);
}

void HistogramWidget::setGtpGamma(GLfloat gtpGamma)
{
    m_gtpGammaSlider->setValue(gtpGamma);
}

bool HistogramWidget::getGtpEnabled() const
{
    QMutexLocker locker(m_lock);
    return m_gtpEnabled;
}

bool HistogramWidget::getGtpAutoMinMax() const
{
    QMutexLocker locker(m_lock);
    return m_gtpAutoMinMaxEnabled;
}

GLushort HistogramWidget::getGtpMin() const
{
    QMutexLocker locker(m_lock);
    return m_gtpMin;
}

GLushort HistogramWidget::getGtpMax() const
{
    QMutexLocker locker(m_lock);
    return m_gtpMax;
}

GLfloat HistogramWidget::getGtpGamma() const
{
    QMutexLocker locker(m_lock);
    return m_gtpGamma;
}

void HistogramWidget::makeView(bool /*doAddWidget*/)
{
    QMutexLocker locker(m_lock);
    ViewWidget::makeView(false);
    m_viewContainerWidget->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
    m_bottomLeftVerticalLayout->insertWidget(1, m_viewContainerWidget);
    m_viewContainerWidget->show();
    m_view->show();
}

View* HistogramWidget::instantiateView()
{
    return new HistogramView(windowHandle());
}

void HistogramWidget::updateImageLoaded(const bool& imageLoaded)
{
    m_imageLoaded = imageLoaded;
    m_imageExtremaValid = false;
    updateEnablement();
}

void HistogramWidget::updateEnablement()
{
    if(m_imageLoaded && m_gtpEnabled)
    {
        m_gtpEnableCheckBox->setEnabled(true);
        m_gtpAutoMinMaxCheckBox->setEnabled(true);
        m_gtpGammaSlider->setEnabled(true);
        m_gtpGammaEditLabel->setEnabled(true);
        m_gtpGammaEdit->setEnabled(true);
        m_gtpMaxSlider->setEnabled(!m_gtpAutoMinMaxEnabled);
        m_gtpMaxEditLabel->setEnabled(true);
        m_gtpMaxEdit->setEnabled(true);
        m_gtpMaxEdit->setReadOnly(m_gtpAutoMinMaxEnabled);
        m_gtpMinSlider->setEnabled(!m_gtpAutoMinMaxEnabled);
        m_gtpMinEditLabel->setEnabled(true);
        m_gtpMinEdit->setEnabled(true);
        m_gtpMinEdit->setReadOnly(m_gtpAutoMinMaxEnabled);
    }
    else
    {
        if(!m_imageLoaded)
        {
            m_gtpEnableCheckBox->setEnabled(false);
        }
        m_gtpAutoMinMaxCheckBox->setEnabled(false);
        m_gtpGammaSlider->setEnabled(false);
        m_gtpGammaEditLabel->setEnabled(false);
        m_gtpGammaEdit->setEnabled(false);
        m_gtpMaxSlider->setEnabled(false);
        m_gtpMaxEditLabel->setEnabled(false);
        m_gtpMaxEdit->setEnabled(false);
        m_gtpMinSlider->setEnabled(false);
        m_gtpMinEditLabel->setEnabled(false);
        m_gtpMinEdit->setEnabled(false);
    }
}

void HistogramWidget::gtpEnabledToggled(bool enabled)
{
    {
        QMutexLocker locker(m_lock);
        m_gtpEnabled = enabled;
    }
    gtpChanged();
    updateEnablement();
}

void HistogramWidget::gtpAutoMinMaxToggled(bool enabled)
{
    {
        QMutexLocker locker(m_lock);
        m_gtpAutoMinMaxEnabled = enabled;
    }
    if(m_gtpAutoMinMaxEnabled)
    {
        m_gtpMinSlider->setValue(65535 - m_imageExtrema.first);
        m_gtpMaxSlider->setValue(m_imageExtrema.second);
    }
    gtpChanged();
    updateEnablement();
}

void HistogramWidget::gtpGammaSliderValueChanged(int value)
{
}

void HistogramWidget::gtpMaxSliderValueChanged(int value)
{
    {
        QMutexLocker locker(m_lock);
        m_gtpMax = static_cast<GLushort>(value);
    }
    m_gtpMaxEdit->setText(QString::number(value));
    // If auto min/max is enabled, then the slider can only have moved because the renderer informed us of new extrema.
    // As such, the renderer does not need to be informed that the slider moved in this case.  The same is true in
    // gtpMinSliderValueChanged(..) below.
    if(!m_gtpAutoMinMaxEnabled)
    {
        if(value < 65535 - m_gtpMinSlider->value())
        {
            m_gtpMinSlider->setValue(65535 - value);
        }
        gtpChanged();
    }
}

void HistogramWidget::gtpMinSliderValueChanged(int value)
{
    value = 65535 - value;
    {
        QMutexLocker locker(m_lock);
        m_gtpMin = static_cast<GLushort>(value);
    }
    m_gtpMinEdit->setText(QString::number(value));
    if(!m_gtpAutoMinMaxEnabled)
    {
        if(value > m_gtpMaxSlider->value())
        {
            m_gtpMaxSlider->setValue(value);
        }
        gtpChanged();
    }
}

void HistogramWidget::gtpGammaEditChanged()
{
}

void HistogramWidget::gtpMaxEditChanged()
{
    bool ok{false};
    int value = m_gtpMaxEdit->text().toInt(&ok);
    if(ok)
    {
        m_gtpMaxSlider->setValue(value);
    }
}

void HistogramWidget::gtpMinEditChanged()
{
    bool ok{false};
    int value = 65535 - m_gtpMinEdit->text().toInt(&ok);
    if(ok)
    {
        m_gtpMinSlider->setValue(value);
    }
}

void HistogramWidget::newImageExtremaFoundByRenderer(GLuint minIntensity, GLuint maxIntensity)
{
    m_imageExtrema.first = minIntensity;
    m_imageExtrema.second = maxIntensity;
    m_imageExtremaValid = true;
    if(m_gtpEnabled && m_gtpAutoMinMaxEnabled)
    {
        m_gtpMinSlider->setValue(65535 - m_imageExtrema.first);
        m_gtpMaxSlider->setValue(m_imageExtrema.second);
    }
}

