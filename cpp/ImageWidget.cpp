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
#include "ImageWidget.h"

const std::vector<GLfloat> ImageWidget::sm_zoomPresets{10.0f, 5.0f, 2.0f, 1.5f, 1.0f, 0.75f, 0.5f, 0.25f, 0.10f};
const std::uint8_t ImageWidget::sm_defaultZoomPreset{4};
const std::pair<GLfloat, GLfloat> ImageWidget::sm_zoomMinMax{0.01f, 10000.0f};
const GLfloat ImageWidget::sm_zoomClickScaleFactor = 0.25f;

ImageWidget::ImageWidget(QWidget* parent)
  : ViewWidget(parent),
    m_interactionMode(InteractionMode::Pointer),
    m_zoomIndex(sm_defaultZoomPreset),
    m_customZoom(0.0f),
    m_zoomToFit(false),
    m_highlightPointer(false),
    m_pointerIsOnImagePixel(false)
{
    setupUi(this);
}

ImageWidget::~ImageWidget()
{
}

ImageView* ImageWidget::imageView()
{
    return dynamic_cast<ImageView*>(m_view.data());
}

void ImageWidget::makeView(bool /*doAddWidget*/)
{
    QMutexLocker locker(m_lock);
    ViewWidget::makeView(false);
    m_scroller->setViewport(m_viewContainerWidget);
    m_viewContainerWidget->show();
    m_view->show();

    connect(m_scroller, &ImageWidgetViewScroller::scrollContentsBySignal, this, &ImageWidget::scrollViewContentsBy);
    connect(m_view.data(), &View::wheelEventSignal, this, &ImageWidget::wheelEventInView);
//  connect(m_view, SIGNAL(mousePressEventSignal(QMouseEvent*)), this, SLOT(mousePressEventInView(QMouseEvent*)));
    connect(m_view.data(), &View::mouseMoveEventSignal, this, &ImageWidget::mouseMoveEventInView);
    connect(m_view.data(), &View::mouseEnterExitSignal, this, &ImageWidget::mouseEnterExitView);
}

View* ImageWidget::instantiateView()
{
    return new ImageView(windowHandle());
}

ImageWidget::InteractionMode ImageWidget::interactionMode() const
{
    return m_interactionMode;
}

void ImageWidget::setInteractionMode(InteractionMode interactionMode)
{
    if(m_interactionMode != interactionMode)
    {
        InteractionMode oldMode{m_interactionMode};
        m_interactionMode = interactionMode;
        switch(m_interactionMode)
        {
        default:
            m_view->unsetCursor();
            break;
        case InteractionMode::Pan:
            m_view->setCursor(Qt::OpenHandCursor);
            break;
        case InteractionMode::Zoom:
            m_view->setCursor(Qt::CrossCursor);
            break;
        }
        interactionModeChanged(m_interactionMode, oldMode);
    }
}

GLfloat ImageWidget::customZoom() const
{
    QMutexLocker locker(const_cast<QMutex*>(m_lock));
    return m_customZoom;
}

int ImageWidget::zoomIndex() const
{
    QMutexLocker locker(const_cast<QMutex*>(m_lock));
    return m_zoomIndex;
}

void ImageWidget::setCustomZoom(GLfloat customZoom)
{
    {
        QMutexLocker locker(m_lock);
        m_customZoom = customZoom;
        m_zoomIndex = -1;
    }
    updateScrollerRanges();
    zoomChanged(m_zoomIndex, m_customZoom);
	m_view->update();
}

void ImageWidget::setZoomIndex(int zoomIndex)
{
    {
        QMutexLocker locker(m_lock);
        m_zoomIndex = zoomIndex;
        m_customZoom = 0.0f;
    }
    updateScrollerRanges();
    zoomChanged(m_zoomIndex, m_customZoom);
	m_view->update();
}

bool ImageWidget::zoomToFit() const
{
    QMutexLocker locker(const_cast<QMutex*>(m_lock));
    return m_zoomToFit;
}

void ImageWidget::setZoomToFit(bool zoomToFit)
{
    {
        QMutexLocker locker(m_lock);
        m_zoomToFit = zoomToFit;
    }
    updateScrollerRanges();
    m_view->update();
}

bool ImageWidget::highlightPointer() const
{
    QMutexLocker locker(const_cast<QMutex*>(m_lock));
    return m_highlightPointer;
}

void ImageWidget::setHighlightPointer(bool highlightPointer)
{
    {
        QMutexLocker locker(m_lock);
        m_highlightPointer = highlightPointer;
    }
    m_view->update();
}

void ImageWidget::updateImageSizeAndData(const QSize& imageSize, const ImageData& imageData)
{
    {
        QMutexLocker locker(m_lock);
        m_imageSize = imageSize;
        m_imageData = imageData;
    }
    updateScrollerRanges();
}

void ImageWidget::resizeEventInView(QResizeEvent* ev)
{
    QMutexLocker locker(m_lock);
    ViewWidget::resizeEventInView(ev);
    updateScrollerRanges();
}

void ImageWidget::updateScrollerRanges()
{
    if(m_zoomToFit)
    {
        m_scroller->horizontalScrollBar()->setRange(0, 0);
        m_scroller->verticalScrollBar()->setRange(0, 0);
    }
    else
    {
        GLfloat z = m_zoomIndex == -1 ? m_customZoom : sm_zoomPresets[m_zoomIndex];

        auto doAxis = [&](GLfloat i, GLfloat w, QScrollBar& s)
        {
            i *= z;
            GLfloat r = std::ceil(i - w);
            if(r <= 0.0f)
            {
                r = 0.0f;
            }
            else
            {
                r /= 2.0f;
            }
            s.setRange(-r, r);
            s.setPageStep(w);
        };

        doAxis(m_imageSize.width(), m_viewSize.width(), *m_scroller->horizontalScrollBar());
        doAxis(m_imageSize.height(), m_viewSize.height(), *m_scroller->verticalScrollBar());
    }
}

void ImageWidget::emitPointerMovedToDifferentPixel(const bool& isOnPixel, const QPoint& pixelCoord, const GLushort& pixelValue)
{
    {
        QMutexLocker locker(m_lock);
        m_pointerIsOnImagePixel = isOnPixel;
        m_pointerImagePixelCoord = pixelCoord;
    }
    if(m_highlightPointer)
    {
        m_view->update();
    }
    pointerMovedToDifferentPixel(isOnPixel, pixelCoord, pixelValue);
}

void ImageWidget::scrollViewContentsBy(int /*dx*/, int /*dy*/)
{
    {
        QMutexLocker locker(m_lock);
        m_pan.setX(m_scroller->horizontalScrollBar()->value());
        m_pan.setY(m_scroller->verticalScrollBar()->value());
    }
    m_view->update();
}

void ImageWidget::mousePressEventInView(QMouseEvent* ev)
{
    ev->accept();
}

void ImageWidget::mouseMoveEventInView(QMouseEvent* ev)
{
    ev->accept();
    Renderer* renderer = m_view->renderer();

    if(renderer == nullptr || m_imageData.isEmpty())
    {
        emitPointerMovedToDifferentPixel(false, QPoint(), 0);
    }
    else
    {
        // Note: double precision is important here; it is trivial to demonstrate that with high zoom and a large image,
        // panning offsets will fall under 32-bit floating point epsilon, causing the calculated image position to be
        // off by > 0.01% near the middle of the image (where coordinate values are large enough for the small pan
        // values to slip under epsilon).  Double precision buys plenty of room, so that by the time error is apparent,
        // the user has bigger problems, such as an image file the size of.... an incomparably massive thing that
        // presents problems for this comparative analogy and anything else it encounters.
        glm::dvec2 ipc(ev->x(), ev->y());
        glm::dvec2 viewSize(m_viewSize.width(), m_viewSize.height());
        glm::dvec2 imageSize(m_imageSize.width(), m_imageSize.height());
        double zoom;
        if(m_zoomToFit)
        {
            double viewAspectRatio = viewSize.x / viewSize.y;
            double imageAspectRatio = imageSize.x / imageSize.y;
            if(imageAspectRatio >= viewAspectRatio)
            {
                // Image is constrained horizontally and centered vertically
                zoom = viewSize.x / imageSize.x;
                ipc.y += (imageSize.y * zoom - viewSize.y) / 2.0;
                ipc /= zoom;
            }
            else
            {
                // Image is constrained vertically and centered horizontally
                zoom = viewSize.y / imageSize.y;
                ipc.x += (imageSize.x * zoom - viewSize.x) / 2.0;
                ipc /= zoom;
            }
        }
        else
        {
            double viewAspectRatio = viewSize.x / viewSize.y;
            double imageAspectRatio = imageSize.x / imageSize.y;
            zoom = m_zoomIndex == -1 ? m_customZoom : sm_zoomPresets[m_zoomIndex];
            // Image is centered vertically and horizontally...
            ipc += (imageSize * zoom - viewSize) / 2.0;
            // ...and is offset by panning
            ipc += glm::dvec2(m_pan.x()*(imageAspectRatio/viewAspectRatio), m_pan.y());
            ipc /= zoom;
        }
        ipc = glm::floor(ipc);
        bool isOnPixel = 
            ipc.x >= 0 && ipc.x < m_imageSize.width() &&
            ipc.y >= 0 && ipc.y < m_imageSize.height();
        GLushort pixelValue = 0;
        if(isOnPixel)
        {
            // Our texture coordinates are set up so that we can feed OpenGL a C-order texture without it being
            // transposed.  This makes indexing back into the texture's data from Qt GUI coordinates a little strange.
            pixelValue = m_imageData[static_cast<std::ptrdiff_t>(imageSize.y - ipc.y - 1.0) * m_imageSize.width() +
                                     static_cast<std::ptrdiff_t>(imageSize.x - ipc.x - 1.0)];
        }
        emitPointerMovedToDifferentPixel(isOnPixel, QPoint(static_cast<int>(ipc.x), static_cast<int>(ipc.y)), pixelValue);
    }
}

void ImageWidget::mouseEnterExitView(bool entered)
{
    if(!entered)
    {
        emitPointerMovedToDifferentPixel(false, QPoint(), 0);
    }
}

void ImageWidget::wheelEventInView(QWheelEvent* ev)
{
    // The number 32 seems good here.  At least, I like it.  Har har.  Ok, 32 is the magic number that makes mouse wheel
    // movement in the view scroll by the same amount as mouse wheel movement on a scrollbar.
    QPoint scrollBy(ev->pixelDelta().isNull() ? ev->angleDelta() / 32 : ev->pixelDelta());
    m_scroller->horizontalScrollBar()->setValue(m_scroller->horizontalScrollBar()->value() - scrollBy.x());
    m_scroller->verticalScrollBar()->setValue(m_scroller->verticalScrollBar()->value() - scrollBy.y());
    ev->accept();
}
