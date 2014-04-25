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

const std::vector<GLfloat> ImageWidget::sm_zoomPresets{2.0f, 1.0f, 0.75f, 0.25f, 0.10f};
const std::pair<GLfloat, GLfloat> ImageWidget::sm_zoomMinMax{0.01f, 1000.0f};
const GLfloat ImageWidget::sm_zoomClickScaleFactor = 0.25f;

ImageWidget::ImageWidget(QWidget* parent)
  : ViewWidget(parent),
    m_interactionMode(InteractionMode::Pointer),
    m_zoomIndex(1),
    m_customZoom(0.0f)
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
    zoomChanged(m_zoomIndex, m_customZoom);
	m_view->update();
}

void ImageWidget::updateImageSize(const QSize& imageSize)
{
    QMutexLocker locker(m_lock);
    m_imageSize = imageSize;
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

void ImageWidget::scrollViewContentsBy(int /*dx*/, int /*dy*/)
{
    {
        QMutexLocker locker(m_lock);
        m_pan.setX(m_scroller->horizontalScrollBar()->value());
        m_pan.setY(m_scroller->verticalScrollBar()->value());
    }
    m_view->update();
}

void ImageWidget::mousePressEventInView(QMouseEvent* event)
{
    event->accept();
}

void ImageWidget::mouseMoveEventInView(QMouseEvent* event)
{
    event->accept();
    pointerMovedToDifferentPixel(true, QPoint(event->x(), event->y()), 65535);
}

void ImageWidget::mouseEnterExitView(bool entered)
{
    if(!entered)
    {
        pointerMovedToDifferentPixel(false, QPoint(), 0);
    }
}

void ImageWidget::wheelEventInView(QWheelEvent* ev)
{
    QPoint scrollBy(ev->pixelDelta().isNull() ? ev->angleDelta() / 32 : ev->pixelDelta());
    m_scroller->horizontalScrollBar()->setValue(m_scroller->horizontalScrollBar()->value() - scrollBy.x());
    m_scroller->verticalScrollBar()->setValue(m_scroller->verticalScrollBar()->value() - scrollBy.y());
    ev->accept();
}
