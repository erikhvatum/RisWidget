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
#include "Renderer.h"
#include "View.h"

View::View(QWindow* parent)
  : QWindow(parent),
    m_clearColorLock(new QMutex)
{
    setSurfaceType(QWindow::OpenGLSurface);
    setFormat(Renderer::sm_format);
    create();

    connect(this, SIGNAL(deferredUpdate()), this, SLOT(onDeferredUpdate()), Qt::QueuedConnection);
}

View::~View()
{
    delete m_clearColorLock;
}

QOpenGLContext* View::context()
{
    return m_context;
}

void View::makeCurrent()
{
    if(!m_context->makeCurrent(this))
    {
        throw RisWidgetException("View::makeCurrent(): Failed to set current OpenGL context.");
    }
}

void View::swapBuffers()
{
    m_context->swapBuffers(this);
}

const SharedGlObjectsPtr& View::sharedGlObjects()
{
    return m_sharedGlObjects;
}

QOpenGLFunctions_4_3_Core* View::glfs()
{
    return m_glfs;
}

void View::setClearColor(const glm::vec4& color)
{
    QMutexLocker clearColorLocker(m_clearColorLock);
    m_clearColor = color;
    update();
}

glm::vec4 View::clearColor() const
{
    QMutexLocker clearColorLocker(const_cast<QMutex*>(m_clearColorLock));
    glm::vec4 ret{m_clearColor};
    return ret;
}

void View::exposeEvent(QExposeEvent* event)
{
    QWindow::exposeEvent(event);
    if(isExposed() && isVisible())
    {
        update();
    }
}

void View::resizeEvent(QResizeEvent* event)
{
    if(isVisible() && isExposed())
    {
        update();
    }
}

void View::update()
{
    if(m_renderer)
    {
        m_renderer->updateView(this);
    }
}

