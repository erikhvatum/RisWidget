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
#include "HistogramView.h"
#include "ui_HistogramWidget.h"
#include "ViewWidget.h"

class Renderer;
class RisWidget;

class HistogramWidget
  : public ViewWidget,
    protected Ui::HistogramWidget
{
    Q_OBJECT;
    friend class Renderer;
    friend class RisWidget;

public:
    explicit HistogramWidget(QWidget* parent = nullptr);
    virtual ~HistogramWidget();

    HistogramView* histogramView();

    void setGtpEnabled(bool gtpEnabled);
    void setGtpAutoMinMax(bool gtpAutoMinMax);
    void setGtpMin(GLushort gtpMin);
    void setGtpMax(GLushort gtpMax);
    void setGtpGamma(GLfloat gtpGamma);

    bool getGtpEnabled() const;
    bool getGtpAutoMinMax() const;
    GLushort getGtpMin() const;
    GLushort getGtpMax() const;
    GLfloat getGtpGamma() const;

signals:
    // Gamma transformation parameters changing invalidates the image view, but the widgets that manipulate these
    // parameters belong to HistogramWidget.  Rather than making HistgoramWidget aware of ImageWidget's view, we instead
    // emit this signal to be received by the main window which will, in turn, cause the image view to be updated.
    void gtpChanged();

protected:
    bool m_imageLoaded;
    bool m_gtpEnabled;
    bool m_gtpAutoMinMaxEnabled;
    GLushort m_gtpMin, m_gtpMax;
    GLfloat m_gtpGamma;
    // Cached data from Renderer
    std::pair<GLushort, GLushort> m_imageExtrema;
    bool m_imageExtremaValid;

    virtual void makeView(bool doAddWidget = true) override;
    virtual View* instantiateView();
    void updateEnablement();
    void updateImageLoaded(const bool& imageLoaded);

protected slots:
    void gtpEnabledToggled(bool enabled);
    void gtpAutoMinMaxToggled(bool enabled);
    void gtpGammaSliderValueChanged(int value);
    void gtpMaxSliderValueChanged(int value);
    void gtpMinSliderValueChanged(int value);
    void gtpGammaEditChanged();
    void gtpMaxEditChanged();
    void gtpMinEditChanged();
    void newImageExtremaFoundByRenderer(GLuint minIntensity, GLuint maxIntensity);
};

