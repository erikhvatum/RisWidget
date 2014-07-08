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
        // OpenGL 4.1 introduces many features including GL_ARB_debug_output and the GLProgramUniform* functions that
        // are painful to be without.
        format.setVersion(4, 1);
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

void Renderer::openClErrorCallbackWrapper(const char* errorInfo, const void* privateInfo, std::size_t cb, void* userData)
{
    reinterpret_cast<Renderer*>(userData)->openClErrorCallback(errorInfo, privateInfo, cb);
}

Renderer::Renderer(ImageWidget* imageWidget, HistogramWidget* histogramWidget)
  : m_threadInited(false),
    m_lock(new QMutex(QMutex::Recursive)),
    m_currOpenClDeviceListEntry(std::numeric_limits<int>::min()),
    m_imageWidget(imageWidget),
    m_imageView(m_imageWidget->imageView()),
    m_histogramWidget(histogramWidget),
    m_histogramView(m_histogramWidget->histogramView()),
    m_glfs(nullptr),
#ifdef ENABLE_GL_DEBUG_LOGGING
    m_glDebugLogger(nullptr),
#endif
    m_imageSize(0, 0),
    m_prevHightlightPointerDrawn(false),
    m_histogramBinCount(2048),
    m_histogramBlocks(std::numeric_limits<GLuint>::max()),
    m_histogram(std::numeric_limits<GLuint>::max()),
    m_histogramData(m_histogramBinCount, 0)
{
    m_imageViewUpdatePending.store(false);
    m_histogramViewUpdatePending.store(false);
    connect(this, &Renderer::_refreshOpenClDeviceList, this, &Renderer::refreshOpenClDeviceListSlot, Qt::QueuedConnection);
    connect(this, &Renderer::_setCurrentOpenClDeviceListIndex, this, &Renderer::setCurrentOpenClDeviceListIndexSlot, Qt::QueuedConnection);
    connect(this, &Renderer::_updateView, this, &Renderer::updateViewSlot, Qt::QueuedConnection);
    connect(this, &Renderer::_newImage, this, &Renderer::newImageSlot, Qt::QueuedConnection);
    connect(this, &Renderer::_setHistogramBinCount, this, &Renderer::setHistogramBinCountSlot, Qt::QueuedConnection);
}

Renderer::~Renderer()
{
    delete m_lock;
}

void Renderer::refreshOpenClDeviceList()
{
    emit _refreshOpenClDeviceList();
}

QVector<QString> Renderer::getOpenClDeviceList() const
{
    QVector<QString> ret;
    QMutexLocker lock(m_lock);
    ret.reserve(m_openClDeviceList.size());
    for(const OpenClDeviceListEntry& entry : m_openClDeviceList)
    {
        ret.append(entry.description);
    }
    return ret;
}

int Renderer::getCurrentOpenClDeviceListIndex() const
{
    QMutexLocker lock(m_lock);
    return m_currOpenClDeviceListEntry;
}

void Renderer::setCurrentOpenClDeviceListIndex(int newOpenClDeviceListIndex)
{
    emit _setCurrentOpenClDeviceListIndex(newOpenClDeviceListIndex);
}

void Renderer::refreshOpenClDeviceListSlot()
{
    try
    {
        QMutexLocker lock(m_lock);
        std::vector<OpenClDeviceListEntry> openClDeviceList;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if(platforms.empty())
        {
            throw RisWidgetException("Renderer::makeClContext(): No OpenCL platform available.");
        }
        for(cl::Platform& platform : platforms)
        {
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            for(cl::Device& device : devices)
            {
                QString typeName;
                cl_device_type type{device.getInfo<CL_DEVICE_TYPE>()}; 
                switch(type)
                {
                case CL_DEVICE_TYPE_CPU:
                    typeName = "CPU";
                    break;
                case CL_DEVICE_TYPE_GPU:
                    typeName = "GPU";
                    break;
                case CL_DEVICE_TYPE_ACCELERATOR:
                    typeName = "Special Purpose Accelerator";
                    break;
                default:
                    typeName = "[unknown]";
                    break;
                }
                std::string deviceName(device.getInfo<CL_DEVICE_NAME>());
                if(deviceName.empty()) deviceName = "[unnamed]";
                std::string supportedOpenClVersion(device.getInfo<CL_DEVICE_VERSION>());
                if(supportedOpenClVersion.empty()) supportedOpenClVersion = "[unknown]";
                QString description(QString("%1 (%2) (%3)").arg(deviceName.c_str(), typeName, supportedOpenClVersion.c_str()));
                openClDeviceList.push_back({description, type, platform(), device()});
            }
        }
        if(openClDeviceList != m_openClDeviceList)
        {
            m_openClDeviceList.swap(openClDeviceList);
            QVector<QString> signalOpenClDeviceList;
            signalOpenClDeviceList.reserve(m_openClDeviceList.size());
            for(const OpenClDeviceListEntry& entry : m_openClDeviceList)
            {
                signalOpenClDeviceList.append(entry.description);
            }
            emit openClDeviceListChanged(signalOpenClDeviceList);
        }
    }
    catch(cl::Error e)
    {
        std::ostringstream o;
        o << "Renderer::refreshOpenClDeviceListSlot(): Failed enumerate OpenCL devices and platforms: " << e.what() << " ";
        o << '(' << e.err() << ").";
        throw RisWidgetException(o.str());
    }
}

void Renderer::setCurrentOpenClDeviceListIndexSlot(int newOpenClDeviceListIndex)
{
    QMutexLocker lock(m_lock);
    if(newOpenClDeviceListIndex != m_currOpenClDeviceListEntry)
    {
        if(newOpenClDeviceListIndex < 0 || static_cast<std::size_t>(newOpenClDeviceListIndex) >= m_openClDeviceList.size())
        {
            std::ostringstream o;
            o << "Renderer::setCurrentOpenClDeviceListIndexSlot(int newOpenClDeviceListIndex): newOpenClDeviceListIndex ";
            o << "must be in the range [0, " << m_openClDeviceList.size() << ").  Note that the right limit of this open ";
            o << "interval is simply the number of logical OpenCL devices made available by the host.";
            throw RisWidgetException(o.str());
        }
        cl::Device device(m_openClDeviceList[newOpenClDeviceListIndex].device);
        cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)m_openClDeviceList[newOpenClDeviceListIndex].platform, 0};
        m_openClContext.reset(new cl::Context(device, properties, &Renderer::openClErrorCallbackWrapper, reinterpret_cast<void*>(this)));
        m_currOpenClDeviceListEntry = newOpenClDeviceListIndex;
        emit currentOpenClDeviceListIndexChanged(m_currOpenClDeviceListEntry);
    }
}

void Renderer::updateView(View* view)
{
    std::atomic_bool* updatePending;
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

    bool updateWasAlreadyPending{updatePending->exchange(true)};
    if(!updateWasAlreadyPending && view->m_context)
    {
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
    if(m_image && m_image->isCreated())
    {
        m_imageCl.reset();
        m_imageData.clear();
        m_image.reset();
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

void Renderer::makeGlContexts()
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
    m_glfs = m_imageView->m_context->versionFunctions<QOpenGLFunctions_4_1_Core>();
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

void Renderer::makeClContext()
{
    // Due to #define __CL_ENABLE_EXCEPTIONS before #include "cl.hpp", the OpenCL API, when accessed through the C++
    // interface provided by cl.hpp, will exception upon error.
    try
    {
        refreshOpenClDeviceListSlot();
        int index = -1, i = 0;
        // A GPU is preferred
        for(OpenClDeviceListEntry& entry : m_openClDeviceList)
        {
            if((entry.type & CL_DEVICE_TYPE_GPU) != 0)
            {
                index = i;
                break;
            }
            ++i;
        }
        // An accelerator (such as a Xeon Phi) is our second choice
        if(index == -1)
        {
            for(OpenClDeviceListEntry& entry : m_openClDeviceList)
            {
                if((entry.type & CL_DEVICE_TYPE_ACCELERATOR) != 0)
                {
                    index = i;
                    break;
                }
                ++i;
            }
            // Running on anything that's not the CPU is our third choice
            if(index == -1)
            {
                for(OpenClDeviceListEntry& entry : m_openClDeviceList)
                {
                    if((entry.type & CL_DEVICE_TYPE_CPU) == 0)
                    {
                        index = i;
                        break;
                    }
                    ++i;
                }
                // Running on the CPU is our final fallback
                if(index == -1)
                {
                    for(OpenClDeviceListEntry& entry : m_openClDeviceList)
                    {
                        if((entry.type & CL_DEVICE_TYPE_CPU) != 0)
                        {
                            index = i;
                            break;
                        }
                        ++i;
                    }
                    if(index == -1)
                    {
                        throw RisWidgetException("No OpenCL device available.");
                    }
                }
            }
        }
        cl::Device device(m_openClDeviceList[index].device);
        m_imageView->makeCurrent();
        cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                                 (cl_context_properties)m_openClDeviceList[index].platform,
#if defined(__APPLE__) || defined(__MACOSX)
                                              // OS X
                                              CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
                                                 (cl_context_properties)CGLGetShareGroup(CGLGetCurrentContext()),
#elif defined(_WIN32)
                                              // Windows
                                              CL_GL_CONTEXT_KHR,
                                                 (cl_context_properties)wglGetCurrentContext(),
                                              CL_WGL_HDC_KHR,
                                                 (cl_context_properties)wglGetCurrentDC()
#else
                                              // Linux (and anything else supporting GLX and all required features)
                                              CL_GL_CONTEXT_KHR,
                                                 (cl_context_properties)glXGetCurrentContext(),
                                              CL_GLX_DISPLAY_KHR,
                                                 (cl_context_properties)glXGetCurrentDisplay(),
#endif
                                              0};
        m_openClContext.reset(new cl::Context(device, properties, &Renderer::openClErrorCallbackWrapper, reinterpret_cast<void*>(this)));
        m_openClCq.reset(new cl::CommandQueue(*m_openClContext, device));
        m_currOpenClDeviceListEntry = index;
        emit currentOpenClDeviceListIndexChanged(m_currOpenClDeviceListEntry);
    }
    catch(cl::Error e)
    {
        std::ostringstream o;
        o << "Renderer::makeClContext(): Failed to create OpenCL context: " << e.what() << " ";
        o << '(' << e.err() << ").";
        throw RisWidgetException(o.str());
    }
    catch(RisWidgetException e)
    {
        throw RisWidgetException(std::string("Renderer::makeClContext(): Failed to create OpenCL context:\n\t") + 
                                 e.description());
    }
}

void Renderer::buildClProgs()
{
    std::vector<cl::Device> devices;
    devices.push_back(cl::Device(m_openClDeviceList[m_currOpenClDeviceListEntry].device));
    auto buildProg = [&](QString sfn, const char* kernFuncName, std::unique_ptr<cl::Program>& prog, std::unique_ptr<cl::Kernel>& kern)
    {
        QFile sf(sfn);
        if(!sf.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            throw RisWidgetException(std::string("Renderer::buildClProgs(): Failed to open OpenCL source file \"") +
                                     sfn.toStdString() + "\".");
        }
        QByteArray s{sf.readAll()};
        if(s.isEmpty())
        {
            throw RisWidgetException(std::string("Renderer::buildClProgs(): Failed to read any data from OpenCL source ")
                                     + std::string("file \"") + sfn.toStdString() + "\".  Is it a zero byte file?  If so, "
                                     "it probably shouldn't be.");
        }
        cl::Program::Sources source(1, std::make_pair(s.data(), s.size()));
        prog.reset(new cl::Program(*m_openClContext, source));
        try
        {
            prog->build(devices);
        }
        catch(cl::Error e)
        {
            if(e.err() == CL_BUILD_PROGRAM_FAILURE)
            {
                throw RisWidgetException(std::string("Failed to build OpenCL source file \"") + sfn.toStdString() +
                                         std::string("\": ") + prog->getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]));
            }
        }
        kern.reset(new cl::Kernel(*prog, kernFuncName));
    };
    
    buildProg(":/gpu/histogramCalc.cl", "histogramCalc", m_histoCalcProg, m_histoCalcKern);
    buildProg(":/gpu/histogramConsolidate.cl", "histogramConsolidate", m_histoConsolidateProg, m_histoConsolidateKern);
}

void Renderer::execHistoCalc()
{
    std::vector<float> in{00.0f, 01.0f, 02.0f, 03.0f,
                          04.0f, 05.0f, 06.0f, 07.0f,
                          08.0f, 09.0f, 10.0f, 11.0f,
                          12.0f, 13.0f, 14.0f, 15.0f};
    std::vector<float> out(16, std::numeric_limits<float>::lowest());
    cl::Buffer inb(*m_openClContext, CL_MEM_READ_ONLY, in.size() * sizeof(float));
    cl::Buffer outb(*m_openClContext, CL_MEM_READ_ONLY, in.size() * sizeof(float));
    m_openClCq->enqueueWriteBuffer(inb, CL_TRUE, 0, in.size() * sizeof(float), in.data());
    m_histoCalcKern->setArg(0, inb);
    m_histoCalcKern->setArg(1, outb);
    m_openClCq->enqueueNDRangeKernel(*m_histoCalcKern, cl::NullRange, cl::NDRange(in.size()), cl::NDRange(1));
    m_openClCq->enqueueReadBuffer(outb, CL_TRUE, 0, in.size() * sizeof(float), const_cast<float*>(out.data()));
    for(float v : out)
    {
        std::cout << v << ' ';
    }
    std::cout << std::endl;
}

void Renderer::execHistoConsolidate()
{
}

void Renderer::updateGlViewportSize(ViewWidget* viewWidget)
{
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

void Renderer::openClErrorCallback(const char* errorInfo, const void*, std::size_t)
{
    std::cerr << "OpenCL error: " << errorInfo << std::endl;
}

void Renderer::execImageDraw()
{
    m_imageView->makeCurrent();

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
        m_imageDrawProg->bind();
        glm::dmat4 pmv(1.0);
        glm::dmat3 fragToTex;
        double zoomFactor;
        glm::dvec2 viewSize(m_imageWidget->m_viewSize.width(), m_imageWidget->m_viewSize.height());
//      bool highlightPointer{m_imageWidget->m_highlightPointer};
//      bool pointerIsOnImagePixel{m_imageWidget->m_pointerIsOnImagePixel};
//      QPoint pointerImagePixelCoord(m_imageWidget->m_pointerImagePixelCoord);

        if(m_imageWidget->m_zoomToFit)
        {
            // Image aspect ratio is always maintained.  The image is centered along whichever axis does not fit.
            widgetLocker.unlock();
            double viewAspectRatio = viewSize.x / viewSize.y;
            double correctionFactor = static_cast<double>(m_imageAspectRatio) / viewAspectRatio;
            if(correctionFactor <= 1)
            {
                pmv = glm::scale(pmv, glm::dvec3(correctionFactor, 1.0, 1.0));
                zoomFactor = viewSize.y / m_imageSize.height();
                // Note that glm wants matrixes in column-major order, so glm matrix element access and constructors are
                // transposed as compared to regular C style 2D arrays
                fragToTex = glm::dmat3(1, 0, 0,
                                       0, 1, 0,
                                       -(viewSize.x - zoomFactor * m_imageSize.width()) / 2, 0, 1);
            }
            else
            {
                pmv = glm::scale(pmv, glm::dvec3(1.0, 1.0 / correctionFactor, 1.0));
                zoomFactor = viewSize.x / m_imageSize.width();
                fragToTex = glm::dmat3(1, 0, 0,
                                       0, 1, 0,
                                       0, -(viewSize.y - zoomFactor * m_imageSize.height()) / 2, 1);
            }
            fragToTex = glm::dmat3(1, 0, 0,
                                   0, 1, 0,
                                   0, 0, zoomFactor) * fragToTex;
        }
        else
        {
            /* Compute vertex transformation matrix */

            // Image aspect ratio is always maintained; the image is centered, panned, and scaled as directed by the
            // user
            zoomFactor = (m_imageWidget->m_zoomIndex == -1) ? m_imageWidget->m_customZoom :
                                                              m_imageWidget->sm_zoomPresets[m_imageWidget->m_zoomIndex];
            glm::dvec2 pan(m_imageWidget->m_pan.x(), m_imageWidget->m_pan.y());
            widgetLocker.unlock();

            double viewAspectRatio = viewSize.x / viewSize.y;
            double correctionFactor = m_imageAspectRatio / viewAspectRatio;
            double sizeRatio(m_imageSize.height());
            sizeRatio /= viewSize.y;
            sizeRatio *= zoomFactor;
            // Scale to same aspect ratio
            pmv = glm::scale(pmv, glm::dvec3(correctionFactor, 1.0, 1.0));
            // Pan.  We've scaled to y along x, so a pan along x in image coordinates relative to y is doubly relative
            // or straight through, depending on your perspective.  Sliders slide in y-up coordinates, whereas graphics
            // stuff addresses pixels y-down: thus the omission of a - before pans.y in the translate call.  If you want
            // pan offset to be in the "natural" direction like the OS-X trackpad default designed to confuse old
            // people, the x and y term signs must be swapped.
            glm::dvec2 pans((pan / viewSize) * 2.0);
            pmv = glm::translate(pmv, glm::dvec3(-(pans.x * (1.0 / correctionFactor)), pans.y, 0.0));
            // Zoom
            pmv = glm::scale(pmv, glm::dvec3(sizeRatio, sizeRatio, 1.0));

            /* Compute gl_FragCoord to texture transformation matrix */

            fragToTex = glm::dmat3(1.0);
            glm::dvec2 imageSize(m_imageSize.width(), m_imageSize.height());
            if(zoomFactor == 1)
            {
                // Facilitate correct one to one drawing by aligning screen and texture coordinates in 100% zoom mode.
                // Not being able to correctly represent a one-to-one image would be disreputable.
                fragToTex[2][0] = std::floor((imageSize.x > viewSize.x) ?
                                             -(viewSize.x - imageSize.x) / 2 + pan.x : -(viewSize.x - imageSize.x) / 2);
                fragToTex[2][1] = std::floor((imageSize.y > viewSize.y) ?
                                             -(viewSize.y - imageSize.y) / 2 - pan.y : -(viewSize.y - imageSize.y) / 2);
            }
            else if(zoomFactor < 1)
            {
                // This case primarily serves to make high frequency, zoomed out image artifacts stay put rather than
                // crawl about when the window is resized
                imageSize *= zoomFactor;
                fragToTex[2][0] = floor((imageSize.x > viewSize.x) ?
                                        -(viewSize.x - imageSize.x) / 2 + pan.x : -(viewSize.x - imageSize.x) / 2);
                fragToTex[2][1] = floor((imageSize.y > viewSize.y) ?
                                        -(viewSize.y - imageSize.y) / 2 - pan.y : -(viewSize.y - imageSize.y) / 2);
                fragToTex = glm::dmat3(1, 0, 0,
                                       0, 1, 0,
                                       0, 0, zoomFactor) * fragToTex;
            }
            else
            {
                // Zoomed in, texture coordinates are unavoidably fractional.  Doing a floor here would cause the image
                // to scroll a pixel at a time even when zoomed in very far.
                imageSize *= zoomFactor;
                fragToTex[2][0] = (imageSize.x > viewSize.x) ?
                    -(viewSize.x - imageSize.x) / 2 + pan.x : -(viewSize.x - imageSize.x) / 2;
                fragToTex[2][1] = (imageSize.y > viewSize.y) ?
                    -(viewSize.y - imageSize.y) / 2 - pan.y : -(viewSize.y - imageSize.y) / 2;
                fragToTex = glm::dmat3(1, 0, 0,
                                       0, 1, 0,
                                       0, 0, zoomFactor) * fragToTex;
            }
        }

        fragToTex = glm::dmat3(1.0 / m_imageSize.width(), 0, 0,
                               0, 1.0 / m_imageSize.height(), 0,
                               0, 0, 1) * fragToTex;

        glm::mat4 pmvf(pmv);
        m_glfs->glUniformMatrix4fv(m_imageDrawProg->m_pmvLoc, 1, GL_FALSE, glm::value_ptr(pmvf));
        glm::mat3 fragToTexf(fragToTex);
        m_glfs->glUniformMatrix3fv(m_imageDrawProg->m_fragToTexLoc, 1, GL_FALSE, glm::value_ptr(fragToTexf));

        m_imageDrawProg->m_quadVao->bind();
        m_image->bind();
        m_glfs->glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        m_image->release();
        m_imageDrawProg->m_quadVao->release();
        m_imageDrawProg->release();
    }
    else
    {
        widgetLocker.unlock();
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

    makeGlContexts();
    makeGlfs();
    buildGlProgs();
    makeClContext();
    buildClProgs();
}

void Renderer::threadDeInitSlot()
{
    if(!m_imageView.isNull() && m_imageView->context() != nullptr)
    {
        m_imageView->makeCurrent();
        if(m_image)
        {
            m_image.reset();
        }
#ifdef ENABLE_GL_DEBUG_LOGGING
        if(m_glDebugLogger != nullptr)
        {
            delete m_glDebugLogger;
            m_glDebugLogger = nullptr;
        }
#endif
    }
    m_histoCalcKern.reset();
    m_histoCalcProg.reset();
    m_histoConsolidateKern.reset();
    m_histoConsolidateProg.reset();
    m_openClCq.reset();
    m_openClContext.reset();
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

        if(!m_image || !m_image->isCreated())
        {
            m_image.reset(new QOpenGLTexture(QOpenGLTexture::Target2D));
            m_image->setFormat(QOpenGLTexture::R32F);
            m_image->setWrapMode(QOpenGLTexture::ClampToEdge);
            m_image->setAutoMipMapGenerationEnabled(true);
            m_image->setSize(imageSize.width(), imageSize.height(), 1);
            m_image->setMipLevels(4);
            m_image->allocateStorage();
        }

        QOpenGLTexture::Filter filterval{filter ? QOpenGLTexture::LinearMipMapLinear : QOpenGLTexture::Nearest};
        m_image->setMinMagFilters(filterval, QOpenGLTexture::Nearest);
        m_image->bind();
        m_glfs->glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        m_image->setData(QOpenGLTexture::Red, QOpenGLTexture::UInt16, reinterpret_cast<GLvoid*>(m_imageData.data()));
        m_image->release();

        m_imageCl.reset(new cl::Image2DGL(*m_openClContext, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, m_image->textureId()));

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
