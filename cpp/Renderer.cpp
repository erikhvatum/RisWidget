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
const QSurfaceFormat Renderer::sm_format{QSurfaceFormat::DebugContext};

void Renderer::staticInit()
{
    if(!sm_staticInited)
    {
        QSurfaceFormat& format = const_cast<QSurfaceFormat&>(sm_format);
        // Our weakest target platform is Macmini6,1, having Intel HD 4000 graphics, supporting up to OpenGL 4.1 on OS X.
        format.setRenderableType(QSurfaceFormat::OpenGL);
        format.setVersion(4, 3);
        format.setProfile(QSurfaceFormat::CoreProfile);
        format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
        format.setStereo(false);
//      format.setSwapBehavior(QSurfaceFormat::TripleBuffer);
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
    m_histogramData(m_histogramBinCount, 0)
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
        *updatePending = true;
        emit _updateView(view);
    }
}

void Renderer::showImage(const ImageData& imageData, const QSize& imageSize, const bool& filter)
{
    if(!imageData.empty() && (imageSize.width() <= 0 || imageSize.height() <= 0))
    {
        throw RisWidgetException("Renderer::showImage(const ImageData& imageData, const QSize& imageSize, const bool& filter): "
                                 "imageData is not empty, but at least one dimension of imageSize is less than or equal to zero.");
    }
    emit _newImage(imageData, imageSize, filter);
}

void Renderer::setHistogramBinCount(const GLuint& histogramBinCount)
{
    emit _setHistogramBinCount(histogramBinCount);
}

std::shared_ptr<LockedRef<const HistogramData>> Renderer::getHistogram()
{
    return std::shared_ptr<LockedRef<const HistogramData>>{
        new LockedRef<const HistogramData>(m_histogramData, *m_lock)};
}

void Renderer::delImage()
{
    if(m_image != std::numeric_limits<GLuint>::max())
    {
        m_imageData.clear();
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
    m_imageView->makeCurrent();
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

void Renderer::updateGlViewportSize(View* view, QSize& size)
{
    QMutexLocker sizeLocker(view->m_sizeLock);
    if(view->m_size != view->m_glSize)
    {
        m_glfs->glViewport(0, 0, view->m_size.width(), view->m_size.height());
        view->m_glSize = view->m_size;
    }
    size = view->m_size;
}

void Renderer::execImageDraw()
{
    m_imageView->makeCurrent();
    m_glfs->glUseProgram(m_imageDrawProg);

    QSize viewSize;
    updateGlViewportSize(m_imageView, viewSize);

    glm::vec4 clearColor{m_imageView->clearColor()};
    m_glfs->glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
    m_glfs->glClearDepth(1.0f);
    m_glfs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if(!m_imageData.empty())
    {
        // Image aspect ratio is always maintained.  The image is centered along whichever axis does not fit.
        float viewAspectRatio = static_cast<float>(viewSize.width()) / viewSize.height();
        float correctionFactor = m_imageAspectRatio / viewAspectRatio;
        glm::mat4 pmv(1.0f);
        if(correctionFactor <= 1)
        {
            pmv = glm::scale(pmv, glm::vec3(correctionFactor, 1.0f, 1.0f));
        }
        else
        {
            pmv = glm::scale(pmv, glm::vec3(1.0f, 1.0f / correctionFactor, 1.0f));
        }
        m_glfs->glUniformMatrix4fv(m_imageDrawProg.projectionModelViewMatrixLoc,
                                   1, GL_FALSE, glm::value_ptr(pmv));

        m_glfs->glBindVertexArray(m_imageDrawProg.quadVao);
        GLuint sub = m_imageDrawProg.gtpEnabled ? 
            m_imageDrawProg.imagePanelGammaTransformColorerIdx : m_imageDrawProg.imagePanelPassthroughColorerIdx;
        m_glfs->glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &sub);
        m_glfs->glBindTexture(GL_TEXTURE_2D, m_image);

        m_glfs->glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        std::cerr << "m_glfs->glDrawArrays(GL_TRIANGLE_FAN, 0, 4);" << std::endl;
    }

    m_imageView->swapBuffers();
}

void Renderer::execHistoDraw()
{
    m_histogramView->makeCurrent();
    m_glfs->glUseProgram(m_histoDrawProg);

    QSize viewSize;
    updateGlViewportSize(m_histogramView, viewSize);

    glm::vec4 clearColor{m_histogramView->clearColor()};
    m_glfs->glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
    m_glfs->glClearDepth(1.0f);
    m_glfs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if(!m_imageData.empty())
    {
        m_glfs->glUniform1ui(m_histoDrawProg.binCountLoc, m_histogramBinCount);
        m_glfs->glUniform1f(m_histoDrawProg.binScaleLoc, m_histoConsolidateProg.extrema[1]);
        glm::mat4 pmv(1.0f);
        m_glfs->glUniformMatrix4fv(m_histoDrawProg.projectionModelViewMatrixLoc, 1, GL_FALSE, glm::value_ptr(pmv));

        if(m_histoDrawProg.pointVao == std::numeric_limits<GLuint>::max())
        {
            m_glfs->glGenVertexArrays(1, &m_histoDrawProg.pointVao);
            m_glfs->glBindVertexArray(m_histoDrawProg.pointVao);

            m_glfs->glGenBuffers(1, &m_histoDrawProg.pointVaoBuff);
            m_glfs->glBindBuffer(GL_ARRAY_BUFFER, m_histoDrawProg.pointVaoBuff);
            {
                std::vector<float> points;
                points.reserve(m_histogramBinCount);
                std::uint32_t i = 0;
                for(std::vector<float>::iterator point{points.begin()}; point != points.end(); ++point, ++i)
                {
                    *point = i;
                }
                m_glfs->glBufferData(GL_ARRAY_BUFFER, m_histogramBinCount,
                                     reinterpret_cast<const GLvoid*>(points.data()),
                                     GL_STATIC_DRAW);
            }

            m_glfs->glEnableVertexAttribArray(m_histoDrawProg.binIndexLoc);
            m_glfs->glVertexAttribPointer(m_histoDrawProg.binIndexLoc, 1, GL_FLOAT, GL_FALSE, 0, nullptr);
        }
        else
        {
            m_glfs->glBindVertexArray(m_histoDrawProg.pointVao);
        }

        m_glfs->glBindTexture(GL_TEXTURE_1D, 0);
        m_glfs->glBindImageTexture(m_histoDrawProg.histogramLoc, m_histogram, 0, true, 0, GL_READ_ONLY, GL_R32UI);

        m_glfs->glDrawArrays(GL_LINE_STRIP, 0, m_histogramBinCount);
        m_glfs->glPointSize(4);
        m_glfs->glDrawArrays(GL_POINTS, 0, m_histogramBinCount);
    }

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
            execImageDraw();
        }
    }
    else if(view == m_histogramView)
    {
        if(m_histogramViewUpdatePending)
        {
            m_histogramViewUpdatePending = false;
            execHistoDraw();
        }
    }
}

void Renderer::newImageSlot(ImageData imageData, QSize imageSize, bool filter)
{
    QMutexLocker locker(m_lock);
    m_imageView->makeCurrent();

    if(!m_imageData.empty() && (imageData.empty() || m_imageSize != imageSize))
    {
        delImage();
        delHistogramBlocks();
    }

    if(!imageData.empty())
    {
        m_imageData = imageData;
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
                                reinterpret_cast<GLvoid*>(m_imageData.data()));
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

        if(!m_imageData.empty())
        {
            execHistoCalc();
            execHistoConsolidate();
            execHistoDraw();
        }
    }
}
