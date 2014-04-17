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
#include "HistogramView.h"
#include "ImageView.h"
#include "Renderer.h"

bool Renderer::sm_staticInited = false;
QSurfaceFormat Renderer::m_format{/*QSurfaceFormat::DebugContext*/};

void Renderer::staticInit()
{
    if(!sm_staticInited)
    {
        // Our weakest target platform is Macmini6,1, having Intel HD 4000 graphics, supporting up to OpenGL 4.1 on OS X.
        sm_format.setVersion(4, 3);
        sm_format.setProfile(QSurfaceFormat::CoreProfile);
        sm_format.setSwapBehavior(QSurfaceFormat::TripleBuffer);
        sm_format.setRenderableType(QSurfaceFormat::OpenGL);
//      QGLFormat format
//      (
//          // Want hardware rendering (should be enabled by default, but this can't hurt)
//          QGL::DirectRendering |
//          // Likewise, double buffering should be enabled by default
//          QGL::DoubleBuffer |
//          // We avoid relying on depcrecated fixed-function pipeline functionality; any attempt to use legacy OpenGL calls
//          // should fail.
//          QGL::NoDeprecatedFunctions |
//          // Disable unused features
//          QGL::NoDepthBuffer |
//          QGL::NoAccumBuffer |
//          QGL::NoStencilBuffer |
//          QGL::NoStereoBuffers |
//          QGL::NoOverlay |
//          QGL::NoSampleBuffers
//      );
        sm_staticInited = true;
    }
}

Renderer::Renderer(ImageView* imageView, HistogramView* histogramView)
  : m_threadInited(false),
    m_lock(new QMutex(QMutex::Recursive)),
    m_imageView(imageView),
    m_imageViewUpdatePending(false),
    m_histogramView(histogramView),
    m_histogramViewUpdatePending(false),
    m_histoCalcProg("histoCalcProg"),
    m_histoConsolidateProg("histoConsolidateProg"),
    m_imageDrawProg("imageDrawProg"),
    m_histoDrawProg("histoDrawProg"),
    m_image(std::numeric_limits<GLuint>::max()),
    m_imageSize(0, 0),
    m_histogramBinCount(2048),
    m_histogramBlocks(std::numeric_limits<GLuint>::max()),
    m_histogram(std::numeric_limits<GLuint>::max()),
    m_histogramData(histogramBinCount, 0),
    m_histogramPmv(1.0f),
{
    connect(this, &Renderer::_updateView, this, &Renderer::updateViewSlot, Qt::QueuedConnection);
    connect(this, &Renderer::_newImage, this, &Renderer::newImageSlot, Qt::QueuedConnection);
    connect(this, &Renderer::_setHistogramBinCount, this, &Renderer::setHistogramBinCountSlot, Qt::QueuedConnection);
}

Renderer::~Renderer()
{
    delete m_lock;
}

void Renderer::updateView(View* view)
{
    bool* updatePending;
    if(view == m_imageView)
    {
        updatePending = &m_imageViewUpdatePending;
    }
    else if(view == m_histogramView)
    {
        updatePending = &m_histogramViewUpdatePending;
    }
    else
    {
        throw RisWidgetException("Renderer::updateView(View* view): View argument refers to neither image nor histogram view.");
    }

    QMutexLocker locker(m_lock);
    if(!*updatePending && view->m_context)
    {
        emit _updateView(view);
    }
}

void Renderer::showImage(const ImageDataPtr& imageDataPtr, const QSize& imageSize, const bool& filter)
{
    if(imageDataPtr && (imageSize.width() <= 0 || imageSize.height() <= 0))
    {
        throw RisWidgetException("Renderer::showImage(const ImageDataPtr& imageDataPtr, const QSize& imageSize, const bool& filter): "
                                 "imageDataPtr is not null, but at least one dimension of imageSize is less than or equal to zero.");
    }
    emit _newImage(imageDataPtr, imageSize, filter);
}

void Renderer::setHistogramBinCount(const GLuint& histogramBinCount)
{
    emit _setHistogramBinCount(histogramBinCount);
}

void Renderer::delImage()
{
    if(m_image != std::numeric_limits<GLuint>::max())
    {
        m_imageDataPtr.reset();
        m_glfs->glDeleteTextures(1, &m_image);
        m_image = std::numeric_limits<GLuint>::max();
        m_imageSize.setWidth(0);
        m_imageSize.setHeight(0);
    }
}

void Renderer::delHistogramBlocks()
{
    if(m_histogramBlocks != std::numeric_limits<GLuint>::max())
    {
        m_glfs->glDeleteTextures(1, &m_histogramBlocks);
        m_histogramBlocks = std::numeric_limits<GLuint>::max();
    }
}

void Renderer::delHistogram()
{
    if(m_histogram != std::numeric_limits<GLuint>::max())
    {
        m_glfs->glDeleteTextures(1, &m_histogram);
        m_histogram = std::numeric_limits<GLuint>::max();

        m_histogramView->makeCurrent();
        m_glfs->glUseProgram(m_histoDrawProg);
        m_glfs->glDeleteVertexArrays(1, &m_histoDrawProg.pointVao);
        m_histoDrawProg.pointVao = std::numeric_limits<GLuint>::max();
        m_glfs->glDeleteBuffers(1, &m_histoDrawProg.pointVaoBuff);
        m_histoDrawProg.pointVaoBuff = std::numeric_limits<GLuint>::max();
    }
}

void Renderer::makeContexts()
{
    m_imageView->m_renderer = this;
    m_imageView->m_context = new QOpenGLContext(this);
    m_imageView->m_context->setFormat(sm_format);

    m_histogramView->m_renderer = this;
    m_histogramView->m_context = new QOpenGLContext(this);
    m_histogramView->m_context->setFormat(sm_format);

    m_imageView->m_context->setShareContext(m_histogramView->m_context);
    m_histogramView->m_context->setShareContext(m_imageView->m_context);

    if(!m_imageView->m_context->create())
    {
        throw RisWidgetException("Renderer::makeContexts(): Failed to create OpenGL context for imageView.");
    }
    if(!m_histogramView->m_context->create())
    {
        throw RisWidgetException("Renderer::makeContexts(): Failed to create OpenGL context for histogramView.");
    }
}

void Renderer::makeGlfs()
{
    // An QOpenGLFunctions_X function bundle instance is associated with a specific context in two ways:
    // 1) The context is responsible for deleting the function bundle instance
    // 2) The function bundle provides OpenGL functions up to, at most, the OpenGL version of the context.  So, you
    // can't get GL4.3 functions from a GL3.3 context, for example.
    // 
    // Therefore, because the image and histogram necessarily are of the same OpenGL version, and because no functions
    // will be needed from either's function bundle while the other does not exist, we can arbitrarily choose to use
    // either view's function bundle exclusively regardless of which view is being manipulated.  We don't need to call
    // through a view's own function bundle when drawing to it.  (However, the specific view's context _does_ need to be
    // current in order to draw to its frame buffer.)
    m_glfs = m_imageView->m_context->versionFunctions<QOpenGLFunctions_4_3_Core>();
    if(m_glfs == nullptr)
    {
        throw RisWidgetException("Renderer::makeGlfs(): Failed to retrieve OpenGL function bundle.");
    }
    if(!m_glfs->initializeOpenGLFunctions())
    {
        throw RisWidgetException("Renderer::makeGlfs(): Failed to initialize OpenGL function bundle.");
    }
}

void Renderer::buildGlProgs()
{
    m_histogramView->makeCurrent();
    m_histoCalcProg.build(m_glfs);
    m_histoConsolidateProg.build(m_glfs);
    m_histoDrawProg.build(m_glfs);

    m_imageView->makeCurrent();
    m_imageDrawProg.build(m_glfs);
}

void Renderer::execHistoCalc()
{
    m_histogramView->makeCurrent();
    m_glfs->glUseProgram(m_histoCalcProg);

    /* Set up data */

    if(m_histogramBlocks == std::numeric_limits<GLuint>::max())
    {
        m_glfs->glGenTextures(1, &m_histogramBlocks);
        m_glfs->glBindTexture(GL_TEXTURE_2D_ARRAY, m_histogramBlocks);
        m_glfs->glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_R32UI,
                               m_histoCalcProg.wgCountPerAxis,
                               m_histoCalcProg.wgCountPerAxis,
                               m_histogramBinCount);
    }
    else
    {
        m_glfs->glBindTexture(GL_TEXTURE_2D_ARRAY, m_histogramBlocks);
    }

    // Zero-out block histogram data... this is slow and should be improved
    std::vector<GLuint> zerosV(m_histoCalcProg.wgCountPerAxis *
                                   m_histoCalcProg.wgCountPerAxis *
                                   m_histogramBinCount,
                               0);
    m_glfs->glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0,
                            m_histoCalcProg.wgCountPerAxis,
                            m_histoCalcProg.wgCountPerAxis,
                            m_histogramBinCount,
                            GL_RED_INTEGER, GL_UNSIGNED_INT,
                            reinterpret_cast<const GLvoid*>(zerosV.data()));

    GLint axisInvocations = m_histoCalcProg.wgCountPerAxis * m_histoCalcProg.liCountPerAxis;
    m_glfs->glUniform2i(m_histoCalcProg.invocationRegionSizeLoc,
                        static_cast<GLint>( ceil(static_cast<double>(m_imageSize.width())  / axisInvocations) ),
                        static_cast<GLint>( ceil(static_cast<double>(m_imageSize.height()) / axisInvocations) ));
    m_glfs->glUniform1f(m_histoCalcProg.binCountLoc, m_histogramBinCount);

    /* Execute */

    m_glfs->glBindTexture(GL_TEXTURE_2D, 0);
    m_glfs->glBindImageTexture(m_histoCalcProg.imageLoc, m_image, 0, false, 0, GL_READ_ONLY, GL_R16UI);

    m_glfs->glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    m_glfs->glBindImageTexture(m_histoCalcProg.blocksLoc, m_histogramBlocks, 0, true, 0, GL_WRITE_ONLY, GL_R32UI);

    m_glfs->glDispatchCompute(m_histoCalcProg.wgCountPerAxis, m_histoCalcProg.wgCountPerAxis, 1);

    // Wait for compute shader execution to complete
    m_glfs->glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void Renderer::execHistoConsolidate()
{
    m_histogramView->makeCurrent();
    m_glfs->glUseProgram(m_histoConsolidateProg);

    /* Set up data */

    if(m_histogram == std::numeric_limits<GLuint>::max())
    {
        m_glfs->glGenTextures(1, &m_histogram);
        m_glfs->glBindTexture(GL_TEXTURE_1D, m_histogram);
        m_glfs->glTexStorage1D(GL_TEXTURE_1D, 1, GL_R32UI, m_histogramBinCount);
    }
    else
    {
        m_glfs->glBindTexture(GL_TEXTURE_1D, m_histogram);
    }

    if(m_histogramData.size() != m_histogramBinCount)
    {
        m_histogramData.resize(m_histogramBinCount, 0);
    }
    // Zero-out histogram data... this is slow and should be improved
    std::fill(m_histogramData.begin(), m_histogramData.end(), 0);
    m_glfs->glTexSubImage1D(GL_TEXTURE_1D, 0, 0, m_histogramBinCount, GL_RED_INTEGER, GL_UNSIGNED_INT,
                            reinterpret_cast<const GLvoid*>(m_histogramData.data()));

    m_glfs->glUniform1ui(m_histoConsolidateProg.binCountLoc, m_histogramBinCount);
    m_glfs->glUniform1ui(m_histoConsolidateProg.invocationBinCountLoc,
                         static_cast<GLuint>( ceil(static_cast<double>(m_histogramBinCount) / m_histoConsolidateProg.liCount) ));

    m_glfs->glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_histoConsolidateProg.extremaBuff);
    m_histoConsolidateProg.extrema[0] = std::numeric_limits<GLuint>::max();
    m_histoConsolidateProg.extrema[1] = std::numeric_limits<GLuint>::min();
    m_glfs->glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(m_histoConsolidateProg.extrema),
                            reinterpret_cast<const GLvoid*>(m_histoConsolidateProg.extrema));

    /* Execute */

    m_glfs->glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    m_glfs->glBindImageTexture(m_histoConsolidateProg.blocksLoc, m_histogramBlocks, 0, true, 0, GL_READ_ONLY, GL_R32UI);

    m_glfs->glBindTexture(GL_TEXTURE_1D, 0);
    m_glfs->glBindImageTexture(m_histoConsolidateProg.histogramLoc, m_histogram, 0, false, 0, GL_READ_WRITE, GL_R32UI);

    m_glfs->glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    m_glfs->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, m_histoConsolidateProg.extremaLoc, m_histoConsolidateProg.extremaBuff);

    m_glfs->glDispatchCompute(m_histoCalcProg.wgCountPerAxis, m_histoCalcProg.wgCountPerAxis, 1);

    // Wait for shader execution to complete
    m_glfs->glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    /* Retrieve results */

    m_glfs->glBindTexture(GL_TEXTURE_1D, m_histogram);
    m_glfs->glGetTexImage(GL_TEXTURE_1D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT,
                          reinterpret_cast<GLvoid*>(const_cast<GLuint*>(m_histogramData.data())));

    m_glfs->glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_histoConsolidateProg.extremaBuff);
    m_glfs->glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(m_histoConsolidateProg.extrema),
                               reinterpret_cast<GLvoid*>(m_histoConsolidateProg.extrema));
}

void Renderer::execImageDraw()
{
    m_imageView->makeCurrent();

    m_imageView->swapBuffers();
}

void Renderer::execHistoDraw()
{
    m_histogramView->makeCurrent();

    m_histogramView->swapBuffers();
}

void Renderer::threadInitSlot()
{
    QMutexLocker locker(m_lock);

    if(m_threadInited)
    {
        throw RisWidgetException("Renderer::threadInit(): Called multiple times for one Renderer instance.");
    }
    m_threadInited = true;

    makeContexts();
    makeGlfs();
    buildGlProgs();
}

void Renderer::updateViewSlot(View* view)
{
    QMutexLocker locker(m_lock);

    if(view == m_imageView)
    {
        if(m_imageViewUpdatePending)
        {
            m_imageViewUpdatePending = false;
            drawImageView();
        }
    }
    else if(view == m_histogramView)
    {
        if(m_histogramViewUpdatePending)
        {
            m_histogramViewUpdatePending = false;
            drawHistogramView();
        }
    }
}

void Renderer::newImageSlot(ImageDataPtr imageDataPtr, QSize imageSize, bool filter)
{
    QMutexLocker locker(m_lock);
    m_imageView->makeCurrent();

    if(m_imageDataPtr && (!imageDataPtr || m_imageSize != imageSize))
    {
        delImage();
        delHistogramBlocks();
    }

    if(imageDataPtr)
    {
        m_imageDataPtr = imageDataPtr;
        m_imageSize = imageSize;
        m_imageAspectRatio = static_cast<float>(m_imageSize.width()) / m_imageSize.height();

        if(m_image == std::numeric_limits<GLuint>::max())
        {
            m_glfs->glGenTextures(1, &m_image);
            m_glfs->glBindTexture(GL_TEXTURE_2D, m_image);
            m_glfs->glTexStorage2D(GL_TEXTURE_2D, 1, GL_R16UI,
                                   m_imageSize.width(), m_imageSize.height());
            m_glfs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            m_glfs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        else
        {
            m_glfs->glBindTexture(GL_TEXTURE_2D, m_image);
        }

        m_glfs->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                m_imageSize.width(), m_imageSize.height(),
                                GL_RED_INTEGER, GL_UNSIGNED_SHORT,
                                reinterpret_cast<GLvoid*>(m_imageDataPtr->data()));
        GLenum filterType = filter ? GL_LINEAR : GL_NEAREST;
        m_glfs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filterType);
        m_glfs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filterType);

        execHistoCalc();
        execHistoConsolidate();
    }

    execImageDraw();
    execHistoDraw();
}

void Renderer::setHistogramBinCountSlot(GLuint histogramBinCount)
{
    QMutexLocker locker(m_lock);

    if(histogramBinCount != m_histogramBinCount)
    {
        m_histogramView->makeCurrent();
        delHistogramBlocks();
        delHistogram();
        m_histogramBinCount = histogramBinCount;

        if(m_imageDataPtr)
        {
            execHistoCalc();
            execHistoConsolidate();
            execHistoDraw();
        }
    }
}
