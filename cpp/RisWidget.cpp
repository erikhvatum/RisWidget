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
    setupImageAndHistogramWidgets();
}

RisWidget::~RisWidget()
{
}

void RisWidget::setupImageAndHistogramWidgets()
{
    QSurfaceFormat format{/*QSurfaceFormat::DebugContext*/};
    // Our weakest target platform is Macmini6,1, having Intel HD 4000 graphics, supporting up to OpenGL 4.1 on OS X.
    format.setVersion(4, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setSwapBehavior(QSurfaceFormat::TripleBuffer);
    format.setRenderableType(QSurfaceFormat::OpenGL);
//  QGLFormat format
//  (
//      // Want hardware rendering (should be enabled by default, but this can't hurt)
//      QGL::DirectRendering |
//      // Likewise, double buffering should be enabled by default
//      QGL::DoubleBuffer |
//      // We avoid relying on depcrecated fixed-function pipeline functionality; any attempt to use legacy OpenGL calls
//      // should fail.
//      QGL::NoDeprecatedFunctions |
//      // Disable unused features
//      QGL::NoDepthBuffer |
//      QGL::NoAccumBuffer |
//      QGL::NoStencilBuffer |
//      QGL::NoStereoBuffers |
//      QGL::NoOverlay |
//      QGL::NoSampleBuffers
//  );

    m_imageWidget->makeImageView(format, m_sharedGlObjects);
    m_histogramWidget->makeHistogramView(format, m_sharedGlObjects, m_imageWidget->imageView());

    m_sharedGlObjects->init(m_imageWidget->imageView(), m_histogramWidget->histogramView());

    m_imageWidget->imageView()->setClearColor(glm::vec4(1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 0.0f));
    m_histogramWidget->histogramView()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));
}

ImageWidget* RisWidget::imageWidget()
{
    return m_imageWidget;
}

HistogramWidget* RisWidget::histogramWidget()
{
    return m_histogramWidget;
}

void RisWidget::showImage(PyObject* image)
{
    if(!PyArray_Check(image))
    {
        throw RisWidgetException("RisWidget::showImage(PyObject* image): image argument must be a numpy array.");
    }
    PyArrayObject* imageao = reinterpret_cast<PyArrayObject*>(image);
    if(PyArray_NDIM(imageao) != 2)
    {
        throw RisWidgetException("RisWidget::showImage(PyObject* image): image argument must be a 2d numpy array.");
    }
    if(PyArray_TYPE(imageao) != NPY_USHORT)
    {
        throw RisWidgetException("RisWidget::showImage(PyObject* image): image argument must be a numpy array of type uint16.");
    }
    npy_intp* dimensions = PyArray_DIMS(imageao);
    std::cerr << dimensions[0] << ", " << dimensions[1] << std::endl;
    void* d = PyArray_DATA(imageao);
    npy_intp* strides = PyArray_STRIDES(imageao);
    for(int y=0, x; y < dimensions[0]; ++y)
    {
        for(x=0; x < dimensions[1]; ++x)
        {
            if(x != 0) std::cerr << ", ";
            std::cerr << *reinterpret_cast<std::uint16_t*>(reinterpret_cast<std::uint8_t*>(d) + y*strides[0] + x*strides[1]);
        }
        std::cerr << std::endl;
    }
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

