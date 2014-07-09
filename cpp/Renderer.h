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
#include "ImageDrawProg.h"
#include "HistoDrawProg.h"

class View;
class ViewWidget;
class ImageView;
class ImageWidget;
class HistogramView;
class HistogramWidget;

// Note that QVector<> does implicit sharing with reference counting and copy-on-write:
// http://qt-project.org/doc/qt-5/qvector.html#QVector-4
typedef QVector<GLushort> ImageData;
typedef std::vector<GLuint> HistogramData;

// In terms of the first example in the "detailed description" section of "http://qt-project.org/doc/qt-5/qthread.html",
// this class would be termed Worker, and the RisWidget class would be Controller.  All public functions are meant to be
// called from the GUI thread.
class Renderer
  : public QObject
{
    Q_OBJECT;

public:
    static void staticInit();
    static const QSurfaceFormat sm_format;

    Renderer(ImageWidget* imageWidget, HistogramWidget* histogramWidget);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator = (const Renderer&) = delete;

    void refreshOpenClDeviceList();
    QVector<QString> getOpenClDeviceList() const;
    int getCurrentOpenClDeviceListIndex() const;
    void setCurrentOpenClDeviceListIndex(int newOpenClDeviceListIndex);

    // Queue a refresh of view (no-op if a refresh of view is already pending).  Refer to View::update()'s declaration
    // in View.h for more details.
    void updateView(View* view);
    // Supply an empty ImageData QVector for imageData with any imageSize and filter arguments to display nothing,
    // removing current image if there is one
    void showImage(const ImageData& imageData, const QSize& imageSize, const bool& filter);
    void setHistogramBinCount(const GLuint& histogramBinCount);
    void getImageDataAndSize(ImageData& imageData, QSize& imageSize) const;
    std::shared_ptr<LockedRef<const HistogramData>> getHistogram();

private:
    struct OpenClDeviceListEntry
    {
        QString description;
        cl_device_type type;
        cl_platform_id platform;
        cl_device_id device;
        inline bool operator == (const OpenClDeviceListEntry& rhs) const
        {
            return description == rhs.description
                && type == rhs.type
                && platform == rhs.platform
                && device == rhs.device;
        }
    };

    static bool sm_staticInited;
    bool m_threadInited;

    QMutex* m_lock;

    std::vector<OpenClDeviceListEntry> m_openClDeviceList;
    int m_currOpenClDeviceListEntry;
    std::unique_ptr<cl::Device> m_openClDevice;
    std::unique_ptr<cl::Context> m_openClContext;
    std::unique_ptr<cl::CommandQueue> m_openClCq;

    static void openClErrorCallbackWrapper(const char* errorInfo, const void* privateInfo, std::size_t cb, void* userData);

    QPointer<ImageWidget> m_imageWidget;
    QPointer<ImageView> m_imageView;
    std::atomic_bool m_imageViewUpdatePending;
    QPointer<HistogramWidget> m_histogramWidget;
    QPointer<HistogramView> m_histogramView;
    std::atomic_bool m_histogramViewUpdatePending;

    // There is good reason not to provide an accessor for this variable: in order that all OpenGL calls originate from
    // the Renderer, only Renderer and GlProgram methods have cause to use m_glfs.
    QOpenGLFunctions_4_1_Core* m_glfs;
#ifdef ENABLE_GL_DEBUG_LOGGING
    QOpenGLDebugLogger* m_glDebugLogger;
#endif

    QPointer<ImageDrawProg> m_imageDrawProg;
    std::unique_ptr<cl::Program> m_histoCalcProg;
    std::unique_ptr<cl::Kernel> m_histoBlocksKern;
    std::unique_ptr<cl::Event> m_histoBlocksKernComplete;
    std::unique_ptr<cl::Kernel> m_histoReduceKern;
    QPointer<HistoDrawProg> m_histoDrawProg;

    // Raw image data
    ImageData m_imageData;
    // Image texture
    std::unique_ptr<QOpenGLTexture> m_image;
    // OpenCL GL sharing reference to m_image
    std::unique_ptr<cl::Image2DGL> m_imageCl;
    void delImage();
    // Dimensions of image texture
    QSize m_imageSize;
    // Aspect ratio of image texture
    float m_imageAspectRatio;
    // When a new image is received, an m_imageExtremaFuture is kicked off.  Later on, when the image extrema (the min
    // and max pixel intensities) are required, they are retrieved into m_imageExtrema from the future in a call that
    // will block until extrema calculation is complete (returning immediately if it already is).  Because
    // m_imageExtremaFuture takes a copy-on-modify reference to m_imageData's data, it is safe for multiple
    // m_imageExtremaFutures to be evaluating simultaneously, although not particularly useful.  The data blobs each
    // instance uses will only be released when that instance terminates.
    std::future<std::pair<GLushort, GLushort>> m_imageExtremaFuture;
    std::pair<GLushort, GLushort> m_imageExtrema;
    // Used when "highlight image pixel under mouse" is enabled in order to remember the location of the previous
    // highlight so that it may be erased when the highlight moves
    bool m_prevHightlightPointerDrawn;
    QPoint m_prevHightlightPointerCoord;

    GLuint m_histogramBinCount;
    std::unique_ptr<cl::Buffer> m_histogramBlocks;
    void delHistogramBlocks();
    // OpenGL buffer backing the histogram buffer texture
    GLuint m_histogramGlBuffer;
    // OpenGL histogram buffer texture
    GLuint m_histogram;
    // OpenCL reference to histogram buffer
    std::unique_ptr<cl::BufferGL> m_histogramClBuffer;
    void delHistogram();
    // RAM cache of histogram data computed by OpenCL
    HistogramData m_histogramData;

    void makeGlContexts();
    void makeGlfs();
    void buildGlProgs();
    void makeClContext();
    void buildClProgs();

    void execHistoCalc();
    void execHistoConsolidate();
    void execImageDraw();
    void execHistoDraw();

    // Helper for execImageDraw() and execHistoDraw()
    void updateGlViewportSize(ViewWidget* viewWidget);
    // The function executed in yet another thread by m_imageExtremaFuture
    static std::pair<GLushort, GLushort> findImageExtrema(ImageData imageData);

    void openClErrorCallback(const char* errorInfo, const void* privateInfo, std::size_t cb);

signals:
    // Leading _ indicates that a signal is private and is used internally for cross-thread procedure calls
    void _refreshOpenClDeviceList();
    void _setCurrentOpenClDeviceListIndex(int newOpenClDeviceListIndex);
    void _newImage(ImageData imageData, QSize imageSize, bool filter);
    void _updateView(View* view);
    void _setHistogramBinCount(GLuint histogramBinCount);
    void openClDeviceListChanged(QVector<QString> openClDeviceList);
    void currentOpenClDeviceListIndexChanged(int currentOpenClDeviceListIndex);
    // Used to notify HistogramWidget so that it can update its min/max sliders and editbox values when in auto min/max
    // mode
    void newImageExtrema(GLushort minIntensity, GLushort maxIntensity);

public slots:
    void threadInitSlot();
    // After shutting down the thread event loop but before exiting the thread entirely, Qt emits a finished() signal.
    // Reception of this signal indicates that we may safely destroy OpenGL resources currently held by Renderer
    // contexts: rendering is initiated by a signal to the thread, and as the thread's event loop has shut down, no more
    // rendering iterations can possibly occur.  Note that if OpenGL resources are not released here and are wrapped in
    // objects with destructors that attempt to release, errors will occur (in the case of QOpenGLTexture, for example).
    void threadDeInitSlot();

private slots:
    void refreshOpenClDeviceListSlot();
    void setCurrentOpenClDeviceListIndexSlot(int newOpenClDeviceListIndex);
    void newImageSlot(ImageData imageData, QSize imageSize, bool filter);
    void updateViewSlot(View* view);
    void setHistogramBinCountSlot(GLuint histogramBinCount);
#ifdef ENABLE_GL_DEBUG_LOGGING
    void glDebugMessageLogged(const QOpenGLDebugMessage& debugMessage);
#endif
};

