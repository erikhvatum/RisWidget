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

%Module _ris_widget

%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip
%Import QtOpenGL/QtOpenGLmod.sip
%Import QtWidgets/QtWidgetsmod.sip

%DefaultDocstringFormat "deindented"




class RisWidget
  : public QMainWindow
{
%TypeHeaderCode
#include "RisWidget.h"
%End

public:
    explicit RisWidget(QString windowTitle_ = "RisWidget",
                       QWidget* parent = nullptr,
                       Qt::WindowFlags flags = 0) /HoldGIL/;
%Docstring
    This constructor must retain the GIL during execution as it may call numpy's import_array() C API function, which
    can result in a Python import amongst other things that impact the interpreter.
%End
    virtual ~RisWidget();

    ImageWidget* imageWidget();
    HistogramWidget* histogramWidget();

    void showCheckerPattern(int width) /ReleaseGIL/;
%Docstring
    RisWidget.showCheckerPattern(width): Display a width X width checkerboard pattern in which each checkerboard square
    is a single pixel.  For the special case in which width is:
    0: a one pixel black image is shown.
    1: a one pixel white image is shown.
%End

    void risImageAcquired(PyObject* stream, PyObject* image) /HoldGIL/;

    void showImage(PyObject* image) /HoldGIL/;
%Docstring
    RisWidget.showImage(image): Display the contents of image, which must be None or array-like and convertable to a 2d
    numpy array of type uint16.  If image is already a 2d, contiguous, C-layout, aligned numpy array, a conversion/copy
    step is skipped and image's data is loaded directly into an OpenGL texture.  If image is None, the current image
    display, if any, is cleared.

    Note that this function holds the GIL as it creates an intermediate numpy array, although the intermediate may be a
    zero copy view of the image argument.
%End

    PyObject* getCurrentImage() /HoldGIL/;

    PyObject* getHistogram() /HoldGIL/;
%Docstring
    Returns a copy of the currently displayed image's histogram data as a numpy array.
%End

    GLuint getHistogramBinCount();
    void setHistogramBinCount(GLuint histogramBinCount);

    void setGtpEnabled(bool gtpEnabled);
%Docstring
    Note: call this method only from the thread with which its bound instance is associated.
%End
    void setGtpAutoMinMax(bool gtpAutoMinMax);
%Docstring
    Note: call this method only from the thread with which its bound instance is associated.
%End
    void setGtpMin(GLushort gtpMin);
%Docstring
    Note: call this method only from the thread with which its bound instance is associated.
%End
    void setGtpMax(GLushort gtpMax);
%Docstring
    Note: call this method only from the thread with which its bound instance is associated.
%End
    void setGtpGamma(GLfloat gtpGamma);
%Docstring
    Note: call this method only from the thread with which its bound instance is associated.
%End
    void setGtpGammaGamma(GLfloat gtpGammaGamma);
%Docstring
    Note: call this method only from the thread with which its bound instance is associated.
%End

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
    void dragEnterEvent(QDragEnterEvent* event);
    void dragMoveEvent(QDragMoveEvent* event);
    void dragLeaveEvent(QDragLeaveEvent* event);
    void dropEvent(QDropEvent* event);

signals:
    void openClDeviceListChanged(QVector<QString> openClDeviceList);
    void currentOpenClDeviceListIndexChanged(int currentOpenClDeviceListIndex);
    void histogramBinCountChanged(int currentHistogramBinCount);

public slots:
    void loadFile() /ReleaseGIL/;
%Docstring
    Presents Open File dialog.  Supports images as well as numpy data files.
%End
    void clearCanvasSlot() /ReleaseGIL/;
%Docstring
    If an image is currently displayed, it is cleared and replaced with nothing.
%End
};




class ImageWidget
  : public QWidget
{
%TypeHeaderCode
#include "ImageWidget.h"
%End

public:
    explicit ImageWidget(QWidget* parent = nullptr);
    virtual ~ImageWidget();

    ImageView* imageView();
};




class ImageView
  : public View /NoDefaultCtors/
{
%TypeHeaderCode
#include "ImageView.h"
%End

public:
    virtual ~ImageView();
};




class HistogramWidget
  : public QWidget
{
%TypeHeaderCode
#include "HistogramWidget.h"
%End

public:
    HistogramWidget(QWidget* parent = nullptr);
    virtual ~HistogramWidget();

    HistogramView* histogramView();
};




class HistogramView
  : public View /NoDefaultCtors/
{
%TypeHeaderCode
#include "HistogramView.h"
%End

public:
    virtual ~HistogramView();

};




class View
  : public QGLWidget /NoDefaultCtors/
{
%TypeHeaderCode
#include "View.h"
%End

public:
    /*class SharedGlObjects
    {
    %TypeHeaderCode
    #include "View.h"
    %End
    };
    typedef std::shared_ptr<SharedGlObjects> SharedGlObjectsPtr;*/

    virtual ~View();
};




class Flipper
  : public QWidget
{
%TypeHeaderCode
#include "Flipper/Flipper.h"
%End

public:
    explicit Flipper(QWidget* parent);
    virtual ~Flipper();

    int getFrameIndex() const;

signals:
    void frameIndexChanged(int frameIndex);

public slots:
    void setFrameIndex(int frameIndex);
    void incrementFrameIndex();

protected slots:
    void playbackButtonClicked();
    void frameIndexSliderPressed();
    void frameIndexSliderReleased();
};
