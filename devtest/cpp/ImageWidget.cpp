#include "ImageWidget.h"
#include <iostream>

ImageWidget::ImageWidget(QWidget* parent)
  : QOpenGLWidget(parent)
{
    QSurfaceFormat qsurface_format;
    qsurface_format.setRenderableType(QSurfaceFormat::OpenGL);
    qsurface_format.setVersion(2, 1);
    qsurface_format.setProfile(QSurfaceFormat::CompatibilityProfile);
    qsurface_format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
    qsurface_format.setStereo(false);
    qsurface_format.setSwapInterval(1);
    setFormat(qsurface_format);
}

void ImageWidget::initializeGL()
{
    std::cout << "ImageWidget::initializeGL()\n";
    if(!initializeOpenGLFunctions())
    {
        std::cerr << "initializeOpenGLFunctions failed.\n";
    }
    glClearColor(0,0,0,1);
    glClearDepth(1);
}

void ImageWidget::paint(QPainter* painter, const QRectF& rect)
{
    std::cout << "ImageWidget::paint(QPainter* painter, const QRectF& rect)\n";
    painter->beginNativePainting();
    glClearColor(0,0,0,1);
    glClearDepth(1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    painter->endNativePainting();
}

void ImageWidget::resizeEvent(QResizeEvent* event)
{
    std::cout << "ImageWidget::resizeEvent(..) w: " << event->size().width() << " h: " << event->size().height() << std::endl;
}

void ImageWidget::resizeGL(int w, int h)
{
    std::cout << "ImageWidget::resizeGL(" << w << ", " << h << ")\n";
}
