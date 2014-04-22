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
    m_interactionMode(InteractionMode::Invalid)
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

void ImageWidget::makeView()
{
    ViewWidget::makeView();
    connect(m_view, &View::mousePressEventSignal, this, &ImageWidget::mousePressEventInView);
}

View* ImageWidget::instantiateView()
{
    return new ImageView(windowHandle());
}

InterationMode ImageWidget::interactionMode() const
{
    return m_interactionMode;
}

void ImageWidget::setInteractionMode(InteractionMode interactionMode)
{
    if(interactionMode == InteractionMode::Invalid)
    {
        throw RisWidgetException("ImageWidget::setInteractionMode(InteractionMode interactionMode): "
                                 "Called with InteractionMode::Invalid for interactionMode argument.");
    }
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
    return m_customZoom;
}

int ImageWidget::zoomIndex() const
{
    return m_zoomIndex;
}

void ImageWidget::setCustomZoom(GLfloat customZoom)
{
	m_customZoom = customZoom;
	m_zoomIndex = -1;
	m_view->update();
}

void ImageWidget::setZoomIndex(int zoomIndex)
{
	m_zoomIndex = zoomIndex;
	m_customZoom = 0.0f;
	m_view->update();
}

void ImageWidget::mousePressEventInView(QMouseEvent* event)
{
    event->accept();
}
