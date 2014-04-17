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

class ImageView;
class HistogramView;

typedef std::vector<uint16_t> ImageData;
typedef std::shared_ptr<ImageData> ImageDataPtr;

// In terms of the first example in the "detailed description" section of "http://qt-project.org/doc/qt-5/qthread.html",
// this class would be termed Worker, and the RisWidget class would be Controller.
class Renderer
  : public QObject
{
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

    // Supply nullptr for imageDataPtr with any imageSize and filter arguments to display nothing, removing current
    // image if there is one
    void showImage(const ImageDataPtr& imageDataPtr, const QSize& imageSize, const bool& filter);

    void setHistogramBinCount(const GLuint& histogramBinCount);

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

    HistoCalcProg m_histoCalcProg;
    HistoConsolidateProg m_histoConsolidateProg;
    ImageDrawProg m_imageDrawProg;
    HistoDrawProg m_histoDrawProg;

    // Raw image data
    ImageDataPtr m_imageDataPtr;
    // Image texture
    GLuint m_image;
    void delImage();
    // Dimensions of image texture
    QSize m_imageSize;
    // Aspect ratio of image texture
    float m_imageAspectRatio;

    bool m_histogramDataStructuresAreStale;
    GLuint m_histogramBinCount;
    GLuint m_histogramBlocks;
    void delHistogramBlocks();
    bool m_histogramDataIsStale;
    GLuint m_histogram;
    void delHistogram();
    std::vector<GLuint> m_histogramData;

    void makeContexts();
    void makeGlfs();
    void buildGlProgs();

    void execHistoCalc();
    void execHistoConsolidate();
    void execImageDraw();
    void execHistoDraw();

signals:
    void _newImage(ImageDataPtr imageDataPtr, QSize imageSize, bool filter);
    void _updateView(View* view);
    void _setHistogramBinCount(GLuint histogramBinCount);

public slots:
    void threadInitSlot();

private slots:
    void newImageSlot(ImageDataPtr imageDataPtr, QSize imageSize, bool filter);
    void updateViewSlot(View* view);
    void setHistogramBinCountSlot(GLuint histogramBinCount);
};

