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
#include "GlProgram.h"

class View;
class ImageView;
class HistogramView;

// Note that QVector<> does implicit sharing with reference counting and copy-on-write:
// http://qt-project.org/doc/qt-5/qvector.html#QVector-4
typedef QVector<GLushort> ImageData;
typedef std::vector<GLuint> HistogramData;

// In terms of the first example in the "detailed description" section of "http://qt-project.org/doc/qt-5/qthread.html",
// this class would be termed Worker, and the RisWidget class would be Controller.
class Renderer
  : public QObject
{
    Q_OBJECT;

public:
    static void staticInit();
    static const QSurfaceFormat sm_format;

    Renderer(ImageView* imageView, HistogramView* histogramView);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator = (const Renderer&) = delete;

    // Queue a refresh of view (no-op if a refresh of view is already pending).  Refer to View::update()'s declaration
    // in View.h for more details.
    void updateView(View* view);
    // Supply an empty ImageData QVector for imageData with any imageSize and filter arguments to display nothing,
    // removing current image if there is one
    void showImage(const ImageData& imageData, const QSize& imageSize, const bool& filter);
    void setHistogramBinCount(const GLuint& histogramBinCount);
    std::shared_ptr<LockedRef<const HistogramData>> getHistogram();

private:
    static bool sm_staticInited;
    bool m_threadInited;

    QMutex* m_lock;

    QPointer<ImageView> m_imageView;
    bool m_imageViewUpdatePending;
    QPointer<HistogramView> m_histogramView;
    bool m_histogramViewUpdatePending;

    // There is good reason not to provide an accessor for this variable: in order that all OpenGL calls originate from
    // the Renderer thread without requiring functions outside of the RenderThread class to execute on the Renderer
    // thread, only Renderer methods have cause to use m_glfs.
    QOpenGLFunctions_4_3_Core* m_glfs;
#ifdef ENABLE_GL_DEBUG_LOGGING
    QOpenGLDebugLogger* m_glDebugLogger;
#endif

    HistoCalcProg m_histoCalcProg;
    HistoConsolidateProg m_histoConsolidateProg;
    ImageDrawProg m_imageDrawProg;
    HistoDrawProg m_histoDrawProg;

    // Raw image data
    ImageData m_imageData;
    // Image texture
    GLuint m_image;
    void delImage();
    // Dimensions of image texture
    QSize m_imageSize;
    // Aspect ratio of image texture
    float m_imageAspectRatio;

    GLuint m_histogramBinCount;
    GLuint m_histogramBlocks;
    void delHistogramBlocks();
    GLuint m_histogram;
    void delHistogram();
    HistogramData m_histogramData;

    void makeContexts();
    void makeGlfs();
    void buildGlProgs();

    void execHistoCalc();
    void execHistoConsolidate();
    void execImageDraw();
    void execHistoDraw();

    // Helper for execImageDraw() and execHistoDraw()
    void updateGlViewportSize(View* view, QSize& size);

signals:
    void _newImage(ImageData imageData, QSize imageSize, bool filter);
    void _updateView(View* view);
    void _setHistogramBinCount(GLuint histogramBinCount);

public slots:
    void threadInitSlot();

private slots:
    void newImageSlot(ImageData imageData, QSize imageSize, bool filter);
    void updateViewSlot(View* view);
    void setHistogramBinCountSlot(GLuint histogramBinCount);
#ifdef ENABLE_GL_DEBUG_LOGGING
    void glDebugMessageLogged(const QOpenGLDebugMessage& debugMessage);
#endif
};

