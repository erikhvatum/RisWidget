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
#include "GlxFunctions.h"
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
    m_pointerIsInImageView(false),
    m_imageViewPointerGlCoord(-1, -1),
    m_imageViewPointerIsOnPixel(false),
    m_imageViewPointerPixelCoord(-1, -1),
    m_histogramBinCount(512),
    m_histogramGlBuffer(std::numeric_limits<GLuint>::max()),
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
    connect(m_imageWidget.data(), &ImageWidget::viewPointerMoved, this, &Renderer::imageViewPointerMovedSlot, Qt::QueuedConnection);
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
        currentOpenClDeviceListIndexChanged(m_currOpenClDeviceListEntry);
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

GLuint Renderer::getHistogramBinCount() const
{
    QMutexLocker locker(const_cast<QMutex*>(m_lock));
    return m_histogramBinCount;
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
    m_histogramBlocks.reset();
    m_histogramZeroBlock.reset();
    m_histoXxKernArgs.reset();
}

void Renderer::delHistogram()
{
    if(m_histogram != std::numeric_limits<GLuint>::max())
    {
        m_histogramClBuffer.reset();
        m_glfs->glDeleteTextures(1, &m_histogram);
        m_histogram = std::numeric_limits<GLuint>::max();
        m_glfs->glDeleteBuffers(1, &m_histogramGlBuffer);
        m_histogramGlBuffer = std::numeric_limits<GLuint>::max();
    }
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
    if(!m_imageView->m_context->hasExtension(QByteArrayLiteral("GL_KHR_debug")))
    {
        std::cerr << "GL_KHR_debug OpenGL extension is required for OpenGL debug logging.  Logging disabled.\n";
    }
    else
    {
        m_glDebugLogger = new QOpenGLDebugLogger(this);
        if(!m_glDebugLogger->initialize())
        {
            throw RisWidgetException("Renderer::makeContexts(): Failed to initialize OpenGL logger.");
        }
        connect(m_glDebugLogger, &QOpenGLDebugLogger::messageLogged, this, &Renderer::glDebugMessageLogged);
        m_glDebugLogger->startLogging(QOpenGLDebugLogger::SynchronousLogging);
        m_glDebugLogger->enableMessages();
    }
#endif
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
        m_openClDevice.reset(new cl::Device(m_openClDeviceList[index].device));
        m_imageView->makeCurrent();
        // Not-these-other-things leaves only GLX as a supported possibility.  It would be good to investigate using EGL
        // for this purpose, if available, before finally attempting GLX.  In the case where EGL is present, if at run
        // time EGL decides to be difficult (perhaps because it happens to be misconfigured and only be aware of OpenGL
        // ES profiles), GLX should be attempted. Does Wayland use EGL?
        std::vector<cl_context_properties> properties
        {
            (cl_context_properties)CL_CONTEXT_PLATFORM,
            (cl_context_properties)m_openClDeviceList[index].platform
        };
#if defined(__APPLE__) || defined(__MACOSX)
            // OS X
        properties.push_back((cl_context_properties)CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE);
        properties.push_back((cl_context_properties)CGLGetShareGroup(CGLGetCurrentContext()));
#elif defined(_WIN32)
        // Windows
        properties.push_back((cl_context_properties)CL_GL_CONTEXT_KHR);
        properties.push_back((cl_context_properties)wglGetCurrentContext());
        properties.push_back((cl_context_properties)CL_WGL_HDC_KHR);
        properties.push_back((cl_context_properties)wglGetCurrentDC());
#else
        // Linux (and anything else supporting GLX and all required features).  The GLX headers #define a bunch of
        // things that really shouldn't be #defined ever, by anyone, and that happen to break Qt.  So, we have to farm
        // out the vector insert operations that require function calls and types from the GLX headers to a plain C
        // compilation unit which can safely do the GLX #includes, being a C file that therefore has a _separate_
        // precompiled header blob with everything that could possibly conflict with GLX headers safely fenced off
        // behind #ifdef __cplusplus.
        properties.push_back((cl_context_properties)0); // dummy replaced by implantClContextGlxSharingProperties(..) call below
        properties.push_back((cl_context_properties)0); // dummy replaced by implantClContextGlxSharingProperties(..) call below
        properties.push_back((cl_context_properties)0); // dummy replaced by implantClContextGlxSharingProperties(..) call below
        properties.push_back((cl_context_properties)0); // dummy replaced by implantClContextGlxSharingProperties(..) call below
        implantClContextGlxSharingProperties(properties.data(), static_cast<unsigned int>(properties.size()));
#endif
        // Terminator
        properties.push_back((cl_context_properties)0);
        m_openClContext.reset(new cl::Context(*m_openClDevice,
                                              properties.data(),
                                              &Renderer::openClErrorCallbackWrapper,
                                              reinterpret_cast<void*>(this)));
        cl_command_queue_properties commandQueueProps{0};
#ifdef ENABLE_CL_PROFILING
        if(m_openClDevice->getInfo<CL_DEVICE_QUEUE_PROPERTIES>() & CL_QUEUE_PROFILING_ENABLE)
        {
            commandQueueProps |= CL_QUEUE_PROFILING_ENABLE;
        }
        else
        {
            throw RisWidgetException("Renderer::makeClContext(): OpenCL profiling was requested but is not supported by the device.");
        }
#endif
        if(m_openClDevice->getInfo<CL_DEVICE_QUEUE_PROPERTIES>() & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
        {
#ifdef ENABLE_CL_PROFILING
            std::cerr << "NOTE: OpenCL command queue out of order execution is SUPPORTED by the OpenCL device, but is not enabled "
                         "(OpenCL profiling is enabled, which can be problematic in combination with out of order execution).\n";
#else
            std::cerr << "NOTE: OpenCL command queue out of order execution is SUPPORTED by the OpenCL device and is ENABLED.\n";
            commandQueueProps = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
#endif
        }
        else
        {
            std::cerr << "NOTE: OpenCL command queue out of order execution not supported by the OpenCL device and is not enabled.\n";
        }
        m_openClCq.reset(new cl::CommandQueue(*m_openClContext, *m_openClDevice, commandQueueProps));
        m_currOpenClDeviceListEntry = index;
        currentOpenClDeviceListIndexChanged(m_currOpenClDeviceListEntry);
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
    m_histoDrawProg = new HistoDrawProg(this);
    if(!m_histoDrawProg->link())
    {
        throw RisWidgetException("Renderer::buildGlProgs(): Failed to link histogram drawing GLSL program.");
    }
    m_histoDrawProg->bind();
    m_histoDrawProg->init(m_glfs);

    m_imageView->makeCurrent();
    m_imageDrawProg = new ImageDrawProg(this);
    if(!m_imageDrawProg->link())
    {
        throw RisWidgetException("Renderer::buildGlProgs(): Failed to link image drawing GLSL program.");
    }
    m_imageDrawProg->bind();
    m_imageDrawProg->init(m_glfs);
}

void Renderer::buildClProgs()
{
    auto buildProg = [&](QString sfn,
                         std::unique_ptr<cl::Program>& prog,
                         std::vector<std::pair<std::string, std::unique_ptr<cl::Kernel>*>>&& kps)
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
            prog->build({*m_openClDevice});
        }
        catch(cl::Error e)
        {
            if(e.err() == CL_BUILD_PROGRAM_FAILURE)
            {
                throw RisWidgetException(std::string("Failed to build OpenCL source file \"") + sfn.toStdString() +
                                         std::string("\": ") + prog->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*m_openClDevice));
            }
        }
        for(std::pair<std::string, std::unique_ptr<cl::Kernel>*>& kp : kps)
        {
            kp.second->reset(new cl::Kernel(*prog, kp.first.c_str()));
        }
    };

    buildProg(":/gpu/histogram.cl", m_histoCalcProg, {std::make_pair(std::string("computeBlocks"), &m_histoBlocksKern),
                                                      std::make_pair(std::string("reduceBlocks"), &m_histoReduceKern)});
}

#ifdef ENABLE_CL_PROFILING
static cl_ulong getClEventDelta(cl::Event& e)
{
    return e.getProfilingInfo<CL_PROFILING_COMMAND_END>() - e.getProfilingInfo<CL_PROFILING_COMMAND_START>();
}

static void printClEventDelta(cl::Event& e, const char* name)
{
    std::cout << name << ": ";
    double d = getClEventDelta(e);
    if(d >= 1.0e9)
    {
        std::cout << (d/1.0e9) << " s";
    }
    else if(d >= 1.0e6)
    {
        std::cout << (d/1.0e6) << " ms";
    }
    else if(d >= 1.0e3)
    {
        std::cout << (d/1.0e3) << " µs";
    }
    else
    {
        std::cout << d << " ns";
    }
    std::cout << std::endl;
}
#endif

void Renderer::execHistoCalc()
{
    static const std::size_t threadsPerWorkgroupAxis{8};
    static const std::size_t threadsPerAxis{128};
    static_assert(threadsPerAxis % threadsPerWorkgroupAxis == 0,
                  "ThreadsPerAxis must be divisible by threadsPerWorkgroupAxis.");
    static const std::size_t workgroupsPerAxis{threadsPerAxis / threadsPerWorkgroupAxis};
    static const std::size_t workgroups{workgroupsPerAxis * workgroupsPerAxis};
    const std::size_t histoByteCount{sizeof(cl_uint) * m_histogramBinCount};
    // Block consolidation is vectorized into blocks of 16 uint32s, so blocks composing the m_histogramBlocks array are
    // padded to 128 byte increments
    const std::size_t histoPaddedBlockByteCount{(histoByteCount % sizeof(cl_uint16) ?
                                                 histoByteCount / sizeof(cl_uint16) + 1 :
                                                 histoByteCount / sizeof(cl_uint16)) * sizeof(cl_uint16)};
    const std::size_t histoBlocksByteCount{histoPaddedBlockByteCount * workgroups};

    cl::Event e0, e1, e2, e3, e4;
    void *b0, *b1, *b2;
/*
    Note the same wait vector contents may not be reused.  Each time a cl::Event object is supplied as the
    output/completion parameter of an OpenCL host function, _a_new_event_is_generated_, and the cl::Event instance no
    longer refers to the same event.  So, this will not work (when attempted, it caused a memory error in the userland
    portion of the OS X Intel driver and a hard lock on a Windows7 NVidia GTX Titan system):

    cl::Event e;
    // e is uninitialized and can not be waited upon
    std::vector<cl::Event> w{e};
    m_openClCq->enqueueOpA(..., nullptr, &e);
    // e now refers to an event, but the shallow copy in w remains uninitialized
    m_openClCq->enqueueOpB_DependingOnA(..., &w);  // SEGFAULT OR HARD LOCK

    Likewise:

    cl::Event e;
    m_openClCq->enqueueOpA(..., nullptr, &e);
    std::vector<cl::Event> w{e};
    m_openClCq->enqueueOpB_DependingOnA(..., &w, &e);
    // e has been waited upon by enqueueOpB_DependingOnA and deleted upon wait completion. Subsequently, a new event
    // was generated and e modified to refer to it.  Upon completion of enqueueOpB_DependingOnA, this event will be
    // triggered.  However, the shallow copy in w still refers to the old event, which was deleted.
    m_openClCq->enqueueOpC_DependingOnB(..., &w); // SEFAULT OR HARD LOCK

    To avoid this, the e value in w must be refreshed before m_openClCq->enqueueOpC_DependingOnB(..., &w).  EG,
    before m_openClCq->enqueueOpC_DependingOnB(..., &w), there should be w[0] = e.
*/

    std::unique_ptr<std::vector<cl::Event>> waits(new std::vector<cl::Event>);
    waits->reserve(4);
    if(m_histogramGlBuffer == std::numeric_limits<GLuint>::max())
    {
        struct XxBlocksConstArgs
        {
            cl_uint2 imageSize;
            cl_uint2 invocationRegionSize;
            cl_uint binCount;
            cl_uint paddedBlockSize;
            cl_uint workgroupsPerAxis;
        };
        m_glfs->glGenBuffers(1, &m_histogramGlBuffer);
        m_glfs->glBindBuffer(GL_TEXTURE_BUFFER, m_histogramGlBuffer);
        m_glfs->glBufferData(GL_TEXTURE_BUFFER, histoByteCount, nullptr, GL_STATIC_COPY);
        m_glfs->glGenTextures(1, &m_histogram);
        m_glfs->glBindTexture(GL_TEXTURE_BUFFER, m_histogram);
        m_glfs->glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, m_histogramGlBuffer);
        m_histogramClBuffer.reset(new cl::BufferGL(*m_openClContext, CL_MEM_WRITE_ONLY, m_histogramGlBuffer));
        m_histogramBlocks.reset(new cl::Buffer(*m_openClContext, CL_MEM_READ_WRITE, histoBlocksByteCount));
        m_histogramZeroBlock.reset(new cl::Buffer(*m_openClContext, CL_MEM_READ_ONLY, histoByteCount));
        m_histoXxKernArgs.reset(new cl::Buffer(*m_openClContext, CL_MEM_READ_ONLY, sizeof(XxBlocksConstArgs)));
        b0 = m_openClCq->enqueueMapBuffer(*m_histoXxKernArgs, CL_FALSE, CL_MAP_WRITE, 0, sizeof(XxBlocksConstArgs), nullptr, &e0);
        b1 = m_openClCq->enqueueMapBuffer(*m_histogramZeroBlock, CL_FALSE, CL_MAP_WRITE, 0, histoByteCount, nullptr, &e1);
        auto roundUp = [&](cl_uint w){
            cl_uint r = w / threadsPerAxis;
            if(w % threadsPerAxis) ++r;
            return r;
        };
        e0.wait();
        // HistoBlocksKernArgs change only when image size and/or histogram bin count change
        *reinterpret_cast<XxBlocksConstArgs*>(b0) =
        {
            {{static_cast<cl_uint>(m_imageSize.width()), static_cast<cl_uint>(m_imageSize.height())}},
            {{roundUp(m_imageSize.width()), roundUp(m_imageSize.height())}},
            m_histogramBinCount,
            static_cast<cl_uint>(histoPaddedBlockByteCount / sizeof(cl_uint)),
            static_cast<cl_uint>(workgroupsPerAxis)
        };
        m_openClCq->enqueueUnmapMemObject(*m_histoXxKernArgs, b0, nullptr, &e0);
        e1.wait();
        memset(b1, 0, histoByteCount);
        m_openClCq->enqueueUnmapMemObject(*m_histogramZeroBlock, b1, nullptr, &e1);
        waits->push_back(e0);
        waits->push_back(e1);
        m_histoBlocksKern->setArg(0, *m_histoXxKernArgs);
        m_histoBlocksKern->setArg(3, histoByteCount, nullptr);
        m_histoBlocksKern->setArg(4, *m_histogramZeroBlock);

        m_histoReduceKern->setArg(0, *m_histoXxKernArgs);
//        cl_uint cibc{m_histogramBinCount / 16};
//        if(m_histogramBinCount % 16) ++cibc;
//        m_histoReduceKern->setArg(1, cibc);
    }
    b2 = m_openClCq->enqueueMapBuffer(*m_histogramBlocks, CL_FALSE, CL_MAP_WRITE, 0, histoBlocksByteCount, nullptr, &e2);

    // All shared GL contexts that in turn share with the CL context must be idle while CL has GL objects acquired
    m_imageView->makeCurrent();
    m_glfs->glFinish();
    m_histogramView->makeCurrent();
    m_glfs->glFinish();

    std::vector<cl::Memory> memObjs{*m_imageCl, *m_histogramClBuffer};
    m_openClCq->enqueueAcquireGLObjects(&memObjs, nullptr, &e3);
    waits->push_back(e3);

    // Zero out histogram blocks buffer
    e2.wait();
    memset(b2, 0, histoBlocksByteCount);
    m_openClCq->enqueueUnmapMemObject(*m_histogramBlocks, b2, nullptr, &e2);
    waits->push_back(e2);

    m_histoBlocksKern->setArg(1, *m_imageCl);
    m_histoBlocksKern->setArg(2, *m_histogramBlocks);

    // Compute histograms for image blocks
    m_openClCq->enqueueNDRangeKernel(*m_histoBlocksKern,
                                     cl::NullRange,
                                     cl::NDRange(threadsPerAxis, threadsPerAxis),
                                     cl::NDRange(threadsPerWorkgroupAxis, threadsPerWorkgroupAxis),
                                     waits.get(),
                                     &e4);

    m_histoReduceKern->setArg(1, *m_histogramBlocks);
    waits->push_back(e4);

    // Sum all block histograms into a histogram representing the entire image
    m_openClCq->enqueueNDRangeKernel(*m_histoReduceKern,
                                     cl::NullRange,
                                     cl::NDRange(workgroupsPerAxis, workgroupsPerAxis),
                                     cl::NDRange(1, 1),
                                     waits.get(),
                                     &e0);

    waits.reset(new std::vector<cl::Event>{e0});
    // Copy first block histogram to GL buffer
    m_openClCq->enqueueCopyBuffer(*m_histogramBlocks, *m_histogramClBuffer, 0, 0, histoByteCount, waits.get(), &e1);
#ifdef ENABLE_CL_PROFILING
    m_openClCq->flush();
    m_openClCq->finish();
    printClEventDelta(e4, "histoBlocksKern");
    printClEventDelta(e0, "histoReduceKern");
#endif
    // Cache histogram data in system RAM and find min & max histogram bin values
    b0 = m_openClCq->enqueueMapBuffer(*m_histogramBlocks, CL_TRUE, CL_MAP_READ, 0, histoByteCount, waits.get());
    m_histogramData.resize(m_histogramBinCount);
    {
        // This will segfault if m_histogramBinCount == 0.  However, if m_histogramBinCount == 0, an explosion would
        // have occurred long before this code is reached.
        GLuint *s{reinterpret_cast<GLuint*>(b0)};
        GLuint *se{reinterpret_cast<GLuint*>(b0) + m_histogramBinCount};
        GLuint *d{m_histogramData.data()};
        m_histogramExtrema.first = *s;
        m_histogramExtrema.second = *s;
        *d = *s;
        ++s;
        if(s != se)
        {
            ++d;
            for(;;)
            {
                if(*s < m_histogramExtrema.first)
                {
                    m_histogramExtrema.first = *s;
                }
                else if(*s > m_histogramExtrema.second)
                {
                    m_histogramExtrema.second = *s;
                }
                *d = *s;
                ++s;
                if(s == se)
                {
                    break;
                }
                ++d;
            }
        }
    }
    m_openClCq->enqueueUnmapMemObject(*m_histogramBlocks, b0, nullptr, &e2);

    waits.reset(new std::vector<cl::Event>{e1, e2});
    m_openClCq->enqueueReleaseGLObjects(&memObjs, waits.get(), &e0);
    e0.wait();

//  uint32_t sum{0};
//  for(auto i(m_histogramData.begin()); i != m_histogramData.end(); ++i)
//  {
//      std::cout << *i << ' ';
//      sum += *i;
//  }
//  std::cout << std::endl << sum << std::endl << std::endl;

//  std::ofstream o("/home/ehvatum/debug.txt", std::ios_base::out | std::ios_base::trunc);
//  b0 = m_openClCq->enqueueMapBuffer(*m_histogramBlocks, CL_TRUE, CL_MAP_READ, 0, histoBlocksByteCount);
//  uint row{0};
//  for(const cl_uint *v{reinterpret_cast<cl_uint*>(b0)}, *re; row < workgroups; ++row)
//  {
//      bool first{true};
//      for(re = v + ((histoByteCount % sizeof(cl_uint16) ?
//                    histoByteCount / sizeof(cl_uint16) + 1 :
//                    histoByteCount / sizeof(cl_uint16))) * sizeof(cl_uint16) / sizeof(cl_uint);
//          v != re;
//          ++v)
//      {
//          if(first) first = false;
//          else o << ' ';
//          o << *v;
//      }
//      o << std::endl;
//  }
//  m_openClCq->enqueueUnmapMemObject(*m_histogramBlocks, b0, nullptr, &e0);
//  e0.wait();
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
    const GLushort *i{imageData.constData()};
    const GLushort *ie{i + imageData.size()};
    std::pair<GLushort, GLushort> ret{*i, *i};
    ++i;
    if(i != ie)
    {
        for(;;)
        {
            if(*i < ret.first)
            {
                ret.first = *i;
            }
            else if(*i > ret.second)
            {
                ret.second = *i;
            }
            ++i;
            if(i == ie)
            {
                break;
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

    std::unique_ptr<QMutexLocker> widgetLocker(new QMutexLocker(m_histogramWidget->m_lock));
    bool gtpEnabled{m_histogramWidget->m_gtpEnabled};
    bool gtpAutoMinMaxEnabled{m_histogramWidget->m_gtpAutoMinMaxEnabled};
    GLfloat gtpMin{static_cast<GLfloat>(m_histogramWidget->m_gtpMin)};
    GLfloat gtpMax{static_cast<GLfloat>(m_histogramWidget->m_gtpMax)};
    GLfloat gtpGamma{m_histogramWidget->m_gtpGamma};

    widgetLocker.reset(new QMutexLocker(m_imageWidget->m_lock));
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
            widgetLocker->unlock();
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
                                       0, (viewSize.y - zoomFactor * m_imageSize.height()) / 2, 1);
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
            widgetLocker->unlock();

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
                                             (viewSize.y - imageSize.y) / 2 + pan.y : (viewSize.y - imageSize.y) / 2);
            }
            else if(zoomFactor < 1)
            {
                // This case primarily serves to make high frequency, zoomed out image artifacts stay put rather than
                // crawl about when the window is resized
                imageSize *= zoomFactor;
                fragToTex[2][0] = floor((imageSize.x > viewSize.x) ?
                                        -(viewSize.x - imageSize.x) / 2 + pan.x : -(viewSize.x - imageSize.x) / 2);
                fragToTex[2][1] = floor((imageSize.y > viewSize.y) ?
                                        (viewSize.y - imageSize.y) / 2 + pan.y : (viewSize.y - imageSize.y) / 2);
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
                    (viewSize.y - imageSize.y) / 2 + pan.y : (viewSize.y - imageSize.y) / 2;
                fragToTex = glm::dmat3(1, 0, 0,
                                       0, 1, 0,
                                       0, 0, zoomFactor) * fragToTex;
            }
        }

        fragToTex = glm::dmat3(1.0 / m_imageSize.width(), 0, 0,
                               0, 1.0 / m_imageSize.height(), 0,
                               0, 0, 1) * fragToTex *
                    glm::dmat3(1, 0, 0,
                               0, -1, 0,
                               0, m_imageSize.height() * zoomFactor, 1);

        m_imageDrawProg->setPmv(glm::mat4(pmv));
        m_imageDrawProg->setFragToTex(glm::mat3(fragToTex));

        if(gtpAutoMinMaxEnabled)
        {
            if(m_imageExtremaFuture.valid())
            {
                // This is the first time we've needed image extrema data since the program was started or a new image
                // was loaded, so we have to retrieve the results from our async worker future object.
                m_imageExtrema = m_imageExtremaFuture.get();
                emit newImageExtrema(m_imageExtrema.first, m_imageExtrema.second);
            }
            gtpMin = m_imageExtrema.first;
            gtpMax = m_imageExtrema.second;
        }

        if(gtpEnabled)
        {
            m_imageDrawProg->setGtpRange(gtpMin / 65535.0f, gtpMax / 65535.0f);
            m_imageDrawProg->setGtpGamma(gtpGamma);
            m_imageDrawProg->setDrawImageSubroutineIdx(m_imageDrawProg->m_drawImageGammaIdx);
        }
        else
        {
            m_imageDrawProg->setDrawImageSubroutineIdx(m_imageDrawProg->m_drawImagePassthroughIdx);
        }

        // Update pointer position highlight in image texture
//      {
//          bool erasePrev;
//          bool drawCur;
// 
//          if(m_prevHightlightPointerDrawn)
//          {
//              if(highlightPointer)
//              {
//                  if(pointerIsOnImagePixel)
//                  {
//                      erasePrev = drawCur = (pointerImagePixelCoord != m_prevHightlightPointerDrawn);
//                  }
//                  else
//                  {
//                      erasePrev = true;
//                      drawCur = false;
//                  }
//              }
//              else
//              {
//                  erasePrev = true;
//                  drawCur = false;
//              }
//          }
//          else
//          {
//              erasePrev = false;
//              drawCur = (highlightPointer && pointerIsOnImagePixel);
//          }
// 
//          if(erasePrev || drawCur)
//          {
//              m_image->bind();
//              const QRect imrect(QPoint(0, 0), m_imageSize);
//              if(erasePrev)
//              {
// 
//              }
//              m_image->generateMipMaps();
//          }
//      }

        m_imageDrawProg->m_quadVao->bind();
        m_image->bind();
        m_glfs->glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        m_image->release();
        m_imageDrawProg->m_quadVao->release();
        m_imageDrawProg->release();
    }
    else
    {
        widgetLocker->unlock();
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

    if(!m_imageData.empty())
    {
        GLfloat gtpGammaGamma{m_histogramWidget->m_gtpGammaGamma};
        widgetLocker.unlock();
        m_histoDrawProg->bind();
        m_histoDrawProg->setGtpGammaGamma(gtpGammaGamma);
        widgetLocker.unlock();
        m_histoDrawProg->setBinCount(m_histogramBinCount);
        m_histoDrawProg->setBinScale(powf(m_histogramExtrema.second, gtpGammaGamma));
        auto binVaoBinder(m_histoDrawProg->getBinVao());

        m_glfs->glBindTexture(GL_TEXTURE_BUFFER, m_histogram);

        m_glfs->glDrawArrays(GL_LINE_STRIP, 0, m_histogramBinCount);
        m_glfs->glPointSize(4);
        m_glfs->glDrawArrays(GL_POINTS, 0, m_histogramBinCount);

        m_glfs->glBindTexture(GL_TEXTURE_BUFFER, 0);
        m_histoDrawProg->release();
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
        m_image.reset();
        if(m_histogram != std::numeric_limits<GLuint>::max())
        {
            m_glfs->glDeleteTextures(1, &m_histogram);
            m_histogram = std::numeric_limits<GLuint>::max();
        }
        m_histogramClBuffer.reset();
        if(m_histogramGlBuffer != std::numeric_limits<GLuint>::max())
        {
            m_glfs->glDeleteBuffers(1, &m_histogramGlBuffer);
            m_histogramGlBuffer = std::numeric_limits<GLuint>::max();
        }
#ifdef ENABLE_GL_DEBUG_LOGGING
        if(m_glDebugLogger != nullptr)
        {
            delete m_glDebugLogger;
            m_glDebugLogger = nullptr;
        }
#endif
    }
    m_histoBlocksKern.reset();
    m_histoReduceKern.reset();
    m_histoCalcProg.reset();
    m_imageCl.reset();
    m_histogramBlocks.reset();
    m_histogramZeroBlock.reset();
    m_histoXxKernArgs.reset();
    m_openClCq.reset();
    m_openClContext.reset();
    m_openClDevice.reset();
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

void Renderer::imageViewPointerMovedSlot(bool isInView, QPoint glViewCoord)
{
    m_pointerIsInImageView = isInView;
    m_imageViewPointerGlCoord = glViewCoord;

    if(!m_imageData.empty() && isInView)
    {
        const glm::mat3 fragToTex(m_imageDrawProg->getFragToTex());
        glm::vec3 pointerPixelCoordh(fragToTex * glm::vec3(glViewCoord.x(), glViewCoord.y(), 1));
        QPoint pointerPixelCoord(static_cast<int>( (pointerPixelCoordh.x / pointerPixelCoordh.z) * m_imageSize.width() ),
                                 static_cast<int>( (pointerPixelCoordh.y / pointerPixelCoordh.z) * m_imageSize.height() ));
        if ( pointerPixelCoord.x() < 0 || pointerPixelCoord.x() >= m_imageSize.width()
          || pointerPixelCoord.y() < 0 || pointerPixelCoord.y() >= m_imageSize.height() )
        {
            if(m_imageViewPointerIsOnPixel)
            {
                m_imageViewPointerIsOnPixel = false;
                imageViewPointerMovedToDifferentPixel(m_imageViewPointerIsOnPixel, QPoint(), 0);
            }
        }
        else
        {
            std::ptrdiff_t i{static_cast<std::ptrdiff_t>(pointerPixelCoord.y())};
            i *= m_imageSize.width();
            i += pointerPixelCoord.x();
            GLushort pointerPixelIntensity{m_imageData[i]};
            if ( m_imageViewPointerIsOnPixel == false
              || pointerPixelCoord != m_imageViewPointerPixelCoord
              || pointerPixelIntensity != m_imageViewPointerPixelIntensity )
            {
                m_imageViewPointerIsOnPixel = true;
                m_imageViewPointerPixelCoord = pointerPixelCoord;
                m_imageViewPointerPixelIntensity = pointerPixelIntensity;
                imageViewPointerMovedToDifferentPixel(m_imageViewPointerIsOnPixel,
                                                      m_imageViewPointerPixelCoord,
                                                      m_imageViewPointerPixelIntensity);
            }
        }
    }
    else
    {
        if(m_imageViewPointerIsOnPixel)
        {
            m_imageViewPointerIsOnPixel = false;
            imageViewPointerMovedToDifferentPixel(m_imageViewPointerIsOnPixel, QPoint(), 0);
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
        delHistogram();
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
    }

    execImageDraw();
    execHistoDraw();
    imageViewPointerMovedSlot(m_pointerIsInImageView, m_imageViewPointerGlCoord);
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
            execHistoDraw();
        }

        histogramBinCountChanged(m_histogramBinCount);
    }
}
