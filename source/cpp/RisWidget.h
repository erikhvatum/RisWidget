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
#include "Flipper/Flipper.h"
#include "HistogramWidget.h"
#include "Image.h"
#include "ImageWidget.h"
#include "ui_RisWidget.h"

class Renderer;

// Unless otherwise noted, all non-inherited RisWidget public functions are meant to be called strictly from the thread
// with which that RisWidget instance is associated (ie, "the GUI thread", or alternately: the thread to which the
// RisWidget instance was most recently moveToThread(..)ed, or if none, the thread on which it was instantiated).
class RisWidget
  : public QMainWindow,
    protected Ui::RisWidget
{
    Q_OBJECT;
    Q_PROPERTY(QVector<QString> openClDeviceList
                   READ getOpenClDeviceList
                   NOTIFY openClDeviceListChanged);
    Q_PROPERTY(int currentOpenClDeviceListIndex
                   READ getCurrentOpenClDeviceListIndex
                   WRITE setCurrentOpenClDeviceListIndex
                   NOTIFY currentOpenClDeviceListIndexChanged);
    Q_PROPERTY(int histogramBinCount
                   READ getHistogramBinCount
                   WRITE setHistogramBinCount
                   NOTIFY histogramBinCountChanged);

public:
    explicit RisWidget(QString windowTitle_ = "RisWidget",
                       QWidget* parent = nullptr,
                       Qt::WindowFlags flags = 0);
    virtual ~RisWidget();

    ImageWidget* imageWidget();
    HistogramWidget* histogramWidget();

    bool hasFlipper(const QString& flipperName) const;
    Flipper* getFlipper(const QString& flipperName);
    QVector<QString> getFlipperNames() const;
    Flipper* showImagesInNewFlipper(PyObject* images);

    void showCheckerPattern(int width, bool filterTexture=false);
    void showImage(const GLushort* imageDataRaw, const QSize& imageSize, bool filterTexture=true);
    void showImage(PyObject* image, bool filterTexture=true);
    PyObject* getCurrentImage();
    PyObject* getHistogram();
    GLuint getHistogramBinCount() const;
    void setHistogramBinCount(GLuint histogramBinCount);

    // setGtp* calls must originate from the thread owning the associated instance
    void setGtpEnabled(bool gtpEnabled);
    void setGtpAutoMinMax(bool gtpAutoMinMax);
    void setGtpMin(GLushort gtpMin);
    void setGtpMax(GLushort gtpMax);
    void setGtpGamma(GLfloat gtpGamma);
    void setGtpGammaGamma(GLfloat gtpGammaGamma);

    // getGtp* calls are thread safe
    bool getGtpEnabled() const;
    bool getGtpAutoMinMax() const;
    GLushort getGtpMin() const;
    GLushort getGtpMax() const;
    GLfloat getGtpGamma() const;
    GLfloat getGtpGammaGamma() const;

    void refreshOpenClDeviceList();
    QVector<QString> getOpenClDeviceList() const;
    int getCurrentOpenClDeviceListIndex() const;
    void setCurrentOpenClDeviceListIndex(int newOpenClDeviceListIndex);

protected:
    QPointer<QActionGroup> m_imageViewInteractionModeGroup;
    QPointer<QToolBar> m_imageViewToolBar;
    QPointer<QComboBox> m_imageViewZoomCombo;
    QPointer<QDoubleValidator> m_imageViewZoomComboValidator;
    QPointer<QWidget> m_statusBarPixelInfoWidget;
    QPointer<QLabel> m_statusBarPixelInfoWidget_x;
    QPointer<QLabel> m_statusBarPixelInfoWidget_y;
    QPointer<QLabel> m_statusBarPixelInfoWidget_intensity;
    QPointer<QWidget> m_statusBarFpsWidget;
    QPointer<QLabel> m_statusBarFpsWidget_fps;
    QScopedPointer<QActionGroup> m_openClDevicesGroup;
    QVector<QAction*> m_actionsOpenClDevices;

    uint64_t m_nextFlipperId;
    std::map<QString, Flipper*> m_flippers;

    std::shared_ptr<Renderer> m_renderer;
    QPointer<QThread> m_rendererThread;
    PyObject* m_numpyModule;
//  PyObject* m_numpyLoadFunction;

    bool m_showStatusBarPixelInfo;
    bool m_showStatusBarFps;
    std::chrono::steady_clock::time_point m_previousFrameTimestamp;
    bool m_previousFrameTimestampValid;

    static QString formatZoom(const GLfloat& z);

    void setupActions();
    void makeToolBars();
    void makeViews();
    void makeRenderer();
    void updateStatusBarFpsPresence();
    void updateStatusBarPixelInfoPresence();

    void dragEnterEvent(QDragEnterEvent* event);
    void dragMoveEvent(QDragMoveEvent* event);
    void dragLeaveEvent(QDragLeaveEvent* event);
    void dropEvent(QDropEvent* event);
#ifdef STAND_ALONE_EXECUTABLE
    void closeEvent(QCloseEvent* event);
#endif

signals:
    void openClDeviceListChanged(QVector<QString> openClDeviceList);
    void currentOpenClDeviceListIndexChanged(int currentOpenClDeviceListIndex);
    void histogramBinCountChanged(int currentHistogramBinCount);

public slots:
    // Presents Open File dialog.  Supports images as well as numpy data files.
    void loadFile();
    void clearCanvasSlot();
    Flipper* makeFlipper();

protected slots:
    void flipperNameChanged(Flipper* flipper, QString oldName);
    void flipperClosing(Flipper* flipper);
    void openClDeviceListChangedSlot(QVector<QString> openClDeviceList);
    void currentOpenClDeviceListIndexChangedSlot(int currentOpenClDeviceListIndex);
    void imageViewZoomComboCustomValueEntered();
    void imageViewZoomComboChanged(int index);
    void imageViewZoomChanged(int zoomIndex, GLfloat customZoom);
    void imageViewZoomToFitToggled(bool zoomToFit);
    void highlightImagePixelUnderMouseToggled(bool highlight);
    void showCheckerPatternSlot();
    void imageViewPointerMovedToDifferentPixel(bool isOnPixel, QPoint pixelCoord, GLushort pixelValue);
    void statusBarPixelInfoToggled(bool showStatusBarPixelInfo);
    void statusBarFpsToggled(bool showStatusBarFps);
};
