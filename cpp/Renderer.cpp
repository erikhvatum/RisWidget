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
#include "HistogramWidget.h"
#include "ImageView.h"
#include "ImageWidget.h"
#include "Renderer.h"

bool Renderer::sm_staticInited = false;

#ifdef ENABLE_GL_DEBUG_LOGGING
const QSurfaceFormat Renderer::sm_format{QSurfaceFormat::DebugContext};
#else
const QSurfaceFormat Renderer::sm_format;
#endif

void Renderer::staticInit()
{
    if(!sm_staticInited)
    {
        QSurfaceFormat& format = const_cast<QSurfaceFormat&>(sm_format);
        format.setRenderableType(QSurfaceFormat::OpenGL);
        // OpenGL 3.3 support is pretty much ubiquitous these days
        format.setVersion(3, 3);
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

Renderer::Renderer(ImageWidget* imageWidget, HistogramWidget* histogramWidget)
  : m_threadInited(false),
    m_lock(new QMutex(QMutex::Recursive)),
    m_imageWidget(imageWidget),
    m_imageView(m_imageWidget->imageView()),
    m_imageViewUpdatePending(false),
    m_histogramWidget(histogramWidget),
    m_histogramView(m_histogramWidget->histogramView()),
    m_histogramViewUpdatePending(false),
    m_glfs(nullptr),
#ifdef ENABLE_GL_DEBUG_LOGGING
    m_glDebugLogger(nullptr),
#endif
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
    if(!imageData.empty())
    {
        if(imageSize.width() <= 0 || imageSize.height() <= 0)
        {
            throw RisWidgetException("Renderer::showImage(const ImageData& imageData, const QSize& imageSize, const bool& filter): "
                                     "imageData is not empty, but at least one dimension of imageSize is less than or equal to zero.");
        }
        {
            QMutexLocker lock(m_lock);
            m_imageExtremaFuture = std::async(&Renderer::findImageExtrema, imageData);
        }
    }
    else
    {
        // It is important to cancel any currently processing or outstanding extrema futures when reverting to
        // displaying no image: if not canceled, it would be possible to show an image, revert to no image, then show an
        // image, and have this third action result in a stale future from the first being used.  (Replacing the
        // m_imageExremaFuture instance with a null future instance accomplishes this.)
        QMutexLocker lock(m_lock);
        m_imageExtremaFuture = std::future<std::pair<GLushort, GLushort>>();
    }
    emit _newImage(imageData, imageSize, filter);
}

void Renderer::setHistogramBinCount(const GLuint& histogramBinCount)
{
    emit _setHistogramBinCount(histogramBinCount);
}

void Renderer::getImageDataAndSize(ImageData& imageData, QSize& imageSize) const
{
    QMutexLocker locker(const_cast<QMutex*>(m_lock));
    imageData = m_imageData;
    imageSize = m_imageSize;
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
//  if(m_histogramBlocks != std::numeric_limits<GLuint>::max())
//  {
//      m_glfs->glDeleteTextures(1, &m_histogramBlocks);
//      m_histogramBlocks = std::numeric_limits<GLuint>::max();
//  }
}

void Renderer::delHistogram()
{
//  if(m_histogram != std::numeric_limits<GLuint>::max())
//  {
//      m_glfs->glDeleteTextures(1, &m_histogram);
//      m_histogram = std::numeric_limits<GLuint>::max();
// 
//      m_histogramView->makeCurrent();
//      m_glfs->glUseProgram(m_histoDrawProg);
//      m_glfs->glDeleteVertexArrays(1, &m_histoDrawProg.pointVao);
//      m_histoDrawProg.pointVao = std::numeric_limits<GLuint>::max();
//      m_glfs->glDeleteBuffers(1, &m_histoDrawProg.pointVaoBuff);
//      m_histoDrawProg.pointVaoBuff = std::numeric_limits<GLuint>::max();
//  }
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
#ifdef ENABLE_GL_DEBUG_LOGGING
    m_imageView->makeCurrent();
    m_glDebugLogger = new QOpenGLDebugLogger(this);
    if(!m_glDebugLogger->initialize())
    {
        throw RisWidgetException("Renderer::makeContexts(): Failed to initialize OpenGL logger.");
    }
    connect(m_glDebugLogger, &QOpenGLDebugLogger::messageLogged, this, &Renderer::glDebugMessageLogged);
    m_glDebugLogger->startLogging(QOpenGLDebugLogger::SynchronousLogging);
    m_glDebugLogger->enableMessages();
#endif
}

#ifdef ENABLE_GL_DEBUG_LOGGING
void Renderer::glDebugMessageLogged(const QOpenGLDebugMessage& debugMessage)
{
    std::cerr << "GL: " << debugMessage.message().toStdString() << std::endl;
}
#endif

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
    m_glfs = m_imageView->m_context->versionFunctions<QOpenGLFunctions_3_3_Core>();
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
//  m_histoCalcProg.build(m_glfs);
//  m_histoConsolidateProg.build(m_glfs);
//  m_histoDrawProg.build(m_glfs);

    m_imageView->makeCurrent();
    m_imageDrawProg = new ImageDrawProg(this);
    // Note that a colon prepended to a filename opened by a Qt object refers to a path in the Qt resource bundle built
    // into a program/library's binary
    if(!m_imageDrawProg->link())
    {
        throw RisWidgetException("Renderer::buildGlProgs(): Failed to link image drawing GLSL program.");
    }
    m_imageDrawProg->bind();
    m_imageDrawProg->init(m_glfs);

//  m_imageDrawProg.build(m_glfs);
}

void Renderer::execHistoCalc()
{
}

void Renderer::execHistoConsolidate()
{
}

void Renderer::updateGlViewportSize(ViewWidget* viewWidget)
{
    // In order for 1:1 zoom to map exactly from image pixels to screen pixels, (0, 0) in OpenGL coordinates must fall
    // as close to the middle of a screen pixel as possible.
    if ( viewWidget->m_viewSize != viewWidget->m_viewGlSize
      && viewWidget->m_viewSize.width() > 0
      && viewWidget->m_viewSize.height() > 0 )
    {
        m_glfs->glViewport(0, 0, viewWidget->m_viewSize.width(), viewWidget->m_viewSize.height());
        viewWidget->m_viewGlSize = viewWidget->m_viewSize;
    }
}

std::pair<GLushort, GLushort> Renderer::findImageExtrema(ImageData imageData)
{
    std::pair<GLushort, GLushort> ret{65535, 0};
    if(imageData.size() == 1)
    {
        ret.first = ret.second = imageData[0];
    }
    else
    {
        for(GLushort p : imageData)
        {
            // Only an image with one pixel must ever have the same pixel be the min and max, which is handled in the if
            // clause above.  An image with two pixels of the same value will fall through to the else below and set the
            // max as well.  Therefore, the special case check for image size of one above lets us use the if/else-if
            // optimization here.
            if(p < ret.first)
            {
                ret.first = p;
            }
            else if(p > ret.second)
            {
                ret.second = p;
            }
        }
    }
    return ret;
}

void Renderer::execImageDraw()
{
    m_imageView->makeCurrent();
    m_imageDrawProg->bind();

    QMutexLocker widgetLocker(m_imageWidget->m_lock);
    updateGlViewportSize(m_imageWidget);

    m_glfs->glClearColor(m_imageWidget->m_clearColor.r,
                         m_imageWidget->m_clearColor.g,
                         m_imageWidget->m_clearColor.b,
                         m_imageWidget->m_clearColor.a);
    m_glfs->glClearDepth(1.0f);
    m_glfs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if(!m_imageData.empty())
    {
        glm::mat4 pmv(1.0f);
        bool highlightPointer{m_imageWidget->m_highlightPointer};
        bool pointerIsOnImagePixel{m_imageWidget->m_pointerIsOnImagePixel};
        QPoint pointerImagePixelCoord(m_imageWidget->m_pointerImagePixelCoord);

        if(m_imageWidget->m_zoomToFit)
        {
            // Image aspect ratio is always maintained.  The image is centered along whichever axis does not fit.
            float viewAspectRatio = static_cast<float>(m_imageWidget->m_viewSize.width()) / m_imageWidget->m_viewSize.height();
            widgetLocker.unlock();
            float correctionFactor = m_imageAspectRatio / viewAspectRatio;
            if(correctionFactor <= 1)
            {
                pmv = glm::scale(pmv, glm::vec3(correctionFactor, 1.0f, 1.0f));
            }
            else
            {
                pmv = glm::scale(pmv, glm::vec3(1.0f, 1.0f / correctionFactor, 1.0f));
            }
        }
        else
        {
            // Image aspect ratio is always maintained; the image is centered, panned, and scaled as directed by the
            // user
            GLfloat zoomFactor = m_imageWidget->m_zoomIndex == -1 ? 
                m_imageWidget->m_customZoom : m_imageWidget->sm_zoomPresets[m_imageWidget->m_zoomIndex];
            glm::vec2 viewSize(m_imageWidget->m_viewSize.width(), m_imageWidget->m_viewSize.height());
            glm::vec2 pan(m_imageWidget->m_pan.x(), m_imageWidget->m_pan.y());
            widgetLocker.unlock();

            GLfloat viewAspectRatio = viewSize.x / viewSize.y;
            GLfloat correctionFactor = m_imageAspectRatio / viewAspectRatio;
            GLfloat sizeRatio(m_imageSize.height());
            sizeRatio /= viewSize.y;
            sizeRatio *= zoomFactor;
            // Scale to same aspect ratio
            pmv = glm::scale(pmv, glm::vec3(correctionFactor, 1.0f, 1.0f));
            // Pan.  We've scaled to y along x, so a pan along x in image coordinates relative to y is doubly relative
            // or straight through, depending on your perspective.  Sliders slide in y-up coordinates, whereas graphics
            // stuff addresses pixels y-down: thus the omission of a - before pans.y in the translate call.  If you want
            // pan offset to be in the "natural" direction like the OS-X trackpad default designed to confuse old
            // people, the x and y term signs must be swapped.
            glm::vec2 pans((pan / viewSize) * 2.0f);
            pmv = glm::translate(pmv, glm::vec3(-(pans.x * (1.0f / correctionFactor)), pans.y, 0.0f));
            // Zoom
            pmv = glm::scale(pmv, glm::vec3(sizeRatio, sizeRatio, 1.0f));
        }
        
        m_glfs->glUniformMatrix4fv(m_imageDrawProg->m_pmvLoc, 1, GL_FALSE, glm::value_ptr(pmv));

        m_imageDrawProg->m_quadVao->bind();
        m_glfs->glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }

    m_imageView->swapBuffers();
}

void Renderer::execHistoDraw()
{
    m_histogramView->makeCurrent();

    QMutexLocker widgetLocker(m_histogramWidget->m_lock);
    updateGlViewportSize(m_histogramWidget);

    m_glfs->glClearColor(m_histogramWidget->m_clearColor.r,
                         m_histogramWidget->m_clearColor.g,
                         m_histogramWidget->m_clearColor.b,
                         m_histogramWidget->m_clearColor.a);
    m_glfs->glClearDepth(1.0f);
    m_glfs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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
// 
//      if(m_image == std::numeric_limits<GLuint>::max())
//      {
//          m_glfs->glGenTextures(1, &m_image);
//          m_glfs->glBindTexture(GL_TEXTURE_2D, m_image);
//          m_glfs->glTexStorage2D(GL_TEXTURE_2D, 1, GL_R16UI,
//                                 m_imageSize.width(), m_imageSize.height());
//          m_glfs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//          m_glfs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//      }
//      else
//      {
//          m_glfs->glBindTexture(GL_TEXTURE_2D, m_image);
//      }
// 
//      m_glfs->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
//                              m_imageSize.width(), m_imageSize.height(),
//                              GL_RED_INTEGER, GL_UNSIGNED_SHORT,
//                              reinterpret_cast<GLvoid*>(m_imageData.data()));
//      GLenum filterType = filter ? GL_LINEAR : GL_NEAREST;
//      m_glfs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filterType);
//      m_glfs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filterType);

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
