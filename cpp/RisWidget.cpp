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

RisWidget::RisWidget(bool enableSwapInterval1,
                     QString windowTitle_,
                     QWidget* parent,
                     Qt::WindowFlags flags)
  : QMainWindow(parent, flags),
    m_sharedGlObjects(new View::SharedGlObjects)
{
    setWindowTitle(windowTitle_);
    setupUi(this);
    setupImageAndHistogramWidgets(enableSwapInterval1);
}

RisWidget::~RisWidget()
{
}

void RisWidget::setupImageAndHistogramWidgets(const bool& enableSwapInterval1)
{
    QGLFormat format
    (
        // Want hardware rendering (should be enabled by default, but this can't hurt)
        QGL::DirectRendering |
        // Likewise, double buffering should be enabled by default
        QGL::DoubleBuffer |
        // We avoid relying on depcrecated fixed-function pipeline functionality; any attempt to use legacy OpenGL calls
        // should fail.
        QGL::NoDeprecatedFunctions |
        // Disable unused features
        QGL::NoDepthBuffer |
        QGL::NoAccumBuffer |
        QGL::NoStencilBuffer |
        QGL::NoStereoBuffers |
        QGL::NoOverlay |
        QGL::NoSampleBuffers
    );
    // Our weakest target platform is Macmini6,1, having Intel HD 4000 graphics, supporting up to OpenGL 4.1 on OS X.
    format.setVersion(4, 3);
    // It's highly likely that enabling swap interval 1 will not ameliorate tearing: any display supporting GL 4.1
    // supports double buffering, and tearing should not be visible with double buffering.  Therefore, the tearing is
    // caused by vsync being off or some fundamental brain damage in your out-of-date X11 display server; further
    // breaking things with swap interval won't help.  But, perhaps you can manage to convince yourself that it's
    // tearing less, and by the simple expedient of displaying less, it will be.
    format.setSwapInterval(enableSwapInterval1 ? 1 : 0);

    m_imageWidget->makeImageView(format, m_sharedGlObjects);
    m_histogramWidget->makeHistogramView(format, m_sharedGlObjects, m_imageWidget->imageView());
}

ImageWidget* RisWidget::imageWidget()
{
    return m_imageWidget;
}

HistogramWidget* RisWidget::histogramWidget()
{
    return m_histogramWidget;
}
