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
#include "HistogramWidget.h"
#include "ImageWidget.h"
#include "Renderer.h"
#include "ui_RisWidget.h"

// Unless otherwise noted, all non-inherited RisWidget public functions are meant to be called strictly from the thread
// with which that RisWidget instance is associated (ie, "the GUI thread", or alternately: the thread to which the
// RisWidget instance was most recently moveToThread(..)ed, or if none, the thread on which it was instantiated).
class RisWidget
  : public QMainWindow,
    protected Ui::RisWidget
{
    Q_OBJECT;

public:
    explicit RisWidget(QString windowTitle_ = "RisWidget",
                       QWidget* parent = nullptr,
                       Qt::WindowFlags flags = 0);
    virtual ~RisWidget();

    ImageWidget* imageWidget();
    HistogramWidget* histogramWidget();

    void showCheckerPattern(int width, bool filterTexture=false);
    void risImageAcquired(PyObject* stream, PyObject* image);
    void showImage(const GLushort* imageDataRaw, const QSize& imageSize, bool filterTexture=true);
    void showImage(PyObject* image, bool filterTexture=true);
    PyObject* getHistogram();

    // setGtp* calls must originate from the thread owning the associated instance
    void setGtpEnabled(bool gtpEnabled);
    void setGtpAutoMinMax(bool gtpAutoMinMax);
    void setGtpMin(GLushort gtpMin);
    void setGtpMax(GLushort gtpMax);
    void setGtpGamma(GLfloat gtpGamma);

    // getGtp* calls are thread safe
    bool getGtpEnabled() const;
    bool getGtpAutoMinMax() const;
    GLushort getGtpMin() const;
    GLushort getGtpMax() const;
    GLfloat getGtpGamma() const;

protected:
    QPointer<QActionGroup> m_imageViewInteractionModeGroup;
    QPointer<QToolBar> m_imageViewToolBar;
    QPointer<QComboBox> m_imageViewZoomCombo;
    QPointer<QDoubleValidator> m_imageViewZoomComboValidator;

    std::shared_ptr<Renderer> m_renderer;
    QPointer<QThread> m_rendererThread;
    boost::python::object m_numpy;
    boost::python::object m_numpyLoad;

    static QString formatZoom(const GLfloat& z);

    void setupActions();
    void makeToolBars();
    void makeViews();
    void makeRenderer();
    void destroyRenderer();

#ifdef STAND_ALONE_EXECUTABLE
    void closeEvent(QCloseEvent* event);
#endif

public slots:
    // Presents Open File dialog.  Supports images as well as numpy data files.
    void loadFile();

protected slots:
    void imageViewZoomComboCustomValueEntered();
    void imageViewZoomComboChanged(int index);
    void imageViewZoomChanged(int zoomIndex, GLfloat customZoom);
    void imageViewZoomToFitToggled(bool zoomToFit);
    void showCheckerPatternSlot();
    void imageViewPointerMovedToDifferentPixel(bool isOnPixel, QPoint pixelCoord, GLushort pixelValue);
};
