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
#include "View.h"

View::View(const QSurfaceFormat& format,
           const SharedGlObjectsPtr& sharedGlObjects_,
           View* sharedContextView)
  : m_context(new QOpenGLContext(this)),
    m_sharedGlObjects(sharedGlObjects_)
{
    setSurfaceType(QWindow::OpenGLSurface);
    setFormat(format);
    create();

    m_context->setFormat(format);

    if(sharedContextView != nullptr)
    {
        m_context->setShareContext(sharedContextView->m_context);
    }
    if(!m_context->create())
    {
        throw RisWidgetException("View::View(..): Failed to create OpenGL context.");
    }

    makeCurrent();
    m_glfs = m_context->versionFunctions<QOpenGLFunctions_4_3_Core>();
    if(m_glfs == nullptr)
    {
        throw RisWidgetException("View::View(..): Failed to retrieve OpenGL functions.");
    }
    if(!m_glfs->initializeOpenGLFunctions())
    {
        throw RisWidgetException("View::View(..): Failed to initialize OpenGL functions.");
    }

    connect(this, SIGNAL(deferredUpdate()), this, SLOT(onDeferredUpdate()), Qt::QueuedConnection);
}

View::~View()
{
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
    m_clearColor = color;
    update();
}

void View::exposeEvent(QExposeEvent* event)
{
    QWindow::exposeEvent(event);
    if(isExposed())
    {
        update();
    }
}

void View::resizeEvent(QResizeEvent* event)
{
    QWindow::resizeEvent(event);
    makeCurrent();
    m_glfs->glViewport(0, 0, event->size().width(), event->size().height());
    update();
}

void View::update()
{
    if(!m_deferredUpdatePending.load())
    {
        emit deferredUpdate();
    }
}

void View::onDeferredUpdate()
{
    if(m_deferredUpdatePending.exchange(false))
    {
        makeCurrent();
        m_glfs->glClearColor(m_clearColor.r, m_clearColor.g, m_clearColor.b, m_clearColor.a);
        m_glfs->glClearDepth(1.0);
        m_glfs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        render();
        m_context->swapBuffers(this);
    }
}
