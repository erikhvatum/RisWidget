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
#include "RisWidget.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL RisWidget_ARRAY_API
#include <numpy/arrayobject.h>

static void* do_import_array()
{
    // import_array() is actually a macro that returns NULL if it fails, so it has to be wrapped in order to be called
    // from a constructor which necessarily does not return anything
    import_array();
    return reinterpret_cast<void*>(1);
}

RisWidget::RisWidget(QString windowTitle_,
                     QWidget* parent,
                     Qt::WindowFlags flags)
  : QMainWindow(parent, flags),
    m_sharedGlObjects(new SharedGlObjects)
{
    static bool oneTimeInitDone{false};
    if(!oneTimeInitDone)
    {
        Q_INIT_RESOURCE(RisWidget);
        do_import_array();
        oneTimeInitDone = true;
    }

    setWindowTitle(windowTitle_);
    setupUi(this);
    Renderer::staticInit();
    makeViews();
    makeRenderer();
}

RisWidget::~RisWidget()
{
}

void RisWidget::makeViews()
{
    m_imageWidget->makeView();
    m_histogramWidget->makeView();

    m_imageWidget->imageView()->setClearColor(glm::vec4(1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 0.0f));
    m_histogramWidget->histogramView()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));
}

void RisWidget::makeRenderer()
{
    m_rendererThread = new QThread(this);
    m_renderer.reset( new Renderer(m_imageWidget->imageView(),
                                   m_histogramWidget->histogramView()) );
    m_renderer->moveToThread(m_rendererThread);
    connect(m_rendererThread.data(), &QThread::started(), m_renderer.get(), &Renderer::threadInitSlot, Qt::QueuedConnection);
    m_rendererThread->start();
}

ImageWidget* RisWidget::imageWidget()
{
    return m_imageWidget;
}

HistogramWidget* RisWidget::histogramWidget()
{
    return m_histogramWidget;
}

void RisWidget::showImage(const std::uint16_t* imageData, const QSize& imageSize, bool filterTexture)
{
    m_imageWidget->imageView()->makeCurrent();
    loadImageData(imageData, imageSize, filterTexture);
    update();
}

void RisWidget::showImage(PyObject* image, bool filterTexture)
{
    PyObject* imageao = PyArray_FromAny(image, PyArray_DescrFromType(NPY_USHORT), 2, 2, NPY_ARRAY_CARRAY_RO, nullptr);
    if(imageao == nullptr)
    {
        throw RisWidgetException("RisWidget::showImage(PyObject* image): image argument must be an "
                                 "array-like object convertable to a 2d uint16 numpy array.");
    }
    npy_intp* shape = PyArray_DIMS(imageao);
    showImage(reinterpret_cast<const std::uint16_t*>(PyArray_DATA(imageao)), QSize(shape[1], shape[0]));
    Py_DECREF(imageao);
}

void RisWidget::loadImageData(const std::uint16_t* imageData, const QSize& imageSize, const bool& filterTexture)
{
    if(imageSize.width() <= 0 || imageSize.height() <= 0)
    {
        throw RisWidgetException("RisWidget::showImage(const std::uint16_t* imageData, const QSize& imageSize): "
                                 "At least one dimension of imageSize is less than or equal to zero.");
    }
    m_sharedGlObjects->imageAspectRatio = static_cast<float>(imageSize.width()) / imageSize.height();
    bool reallocate = m_sharedGlObjects->imageSize != imageSize;
    m_sharedGlObjects->imageSize = imageSize;

    m_sharedGlObjects->histogramIsStale = true;
    m_sharedGlObjects->histogramDataStructuresAreStale = reallocate;

    QOpenGLFunctions_4_3_Core* glfs = m_imageWidget->imageView()->glfs();

    if(reallocate && m_sharedGlObjects->image != std::numeric_limits<GLuint>::max())
    {
        glfs->glDeleteTextures(1, m_sharedGlObjects->image);
        m_sharedGlObjects->image = std::numeric_limits<GLuint>::max();
    }

    if(m_sharedGlObjects->image == std::numeric_limits<GLuint>::max())
    {
        glfs->glGenTextures(1, &m_sharedGlObjects->image);
        glfs->glBindTexture(GL_TEXTURE_2D, m_sharedGlObjects->image)
        glfs->glTexStorage2D(GL_TEXTURE_2D, 1, GL_R16UI, imageSize.width(), imageSize.height());
    }
    else
    {
        glfs->glBindTexture(GL_TEXTURE_2D, m_sharedGlObjects->image)
    }

    glfs->glTexImage2D(GL_TEXTURE_2D,
                       0,
                       GL_R16UI,
                       imageSize.width(), imageSize.height(),
                       0,
                       GL_RED_INTEGER, GL_UNSIGNED_SHORT,
                       reinterpret_cast<GLvoid*>(imageData));
    GLenum filterType = filterTexture ? GL_LINEAR : GL_NEAREST;
    glfs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filterType);
    glfs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filterType);
    glfs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glfs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void RisWidget::makeRenderer()
{
}

void RisWidget::destroyRenderer()
{
}

#ifdef STAND_ALONE_EXECUTABLE
#include <QApplication>

int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    RisWidget risWidget;
    risWidget.show();
    return app.exec();
}

#endif

