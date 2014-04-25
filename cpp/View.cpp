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
    m_visibleAndExposed(false)
{
    setSurfaceType(QWindow::OpenGLSurface);
    setFormat(Renderer::sm_format);
    create();
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

void View::swapBuffers()
{
    if(m_visibleAndExposed.load())
    {
        m_context->swapBuffers(this);
    }
    else
    {
//      std::cerr << "View::swapBuffers(): swapBuffers(..) call skipped." << std::endl;
    }
}

void View::update()
{
    if(m_renderer && m_visibleAndExposed.load())
    {
        // Note the m_visibleAndExposed.load() condition.  It never makes sense to queue a render if the user can not
        // see the render result.  A render result can only be needed because of a change to the widget state that
        // entails an event, and when that event is processed, widget visibility is checked and an update is issued at
        // that time if the widget is visible.  So, there is not even a reason to put render requests in a queue for
        // lazy execution: when a render is needed, a request is submitted automatically.  What use would another
        // request be? To render the same thing twice at the same instant?
        // 
        // Additionally, swapBuffers(..), called by Renderer upon completion of drawing to display the new rendering,
        // checks m_visibleAndExposed before actually doing a render.  Thus, if a View becomes hidden before Renderer
        // gets around to rendering it, the update is skipped.  However, the next time the View becomes visible, a new
        // update is queued, so there is no opportunity for a View to be both visible and without being refreshed or at
        // least having a refresh pending.
        m_renderer->updateView(this);
    }
}

bool View::visibleAndExposed() const
{
    return m_visibleAndExposed.load();
}

bool View::event(QEvent* ev)
{
    if(ev->type() == QEvent::Enter)
    {
        mouseEnterExitSignal(true);
    }
    else if(ev->type() == QEvent::Leave)
    {
        mouseEnterExitSignal(false);
    }
    return QWindow::event(ev);
}

void View::exposeEvent(QExposeEvent* ev)
{
    ev->accept();
    if(isExposed() && isVisible())
    {
        m_visibleAndExposed.store(true);
        update();
    }
    else
    {
        m_visibleAndExposed.store(false);
    }
}

void View::resizeEvent(QResizeEvent* ev)
{
    ev->accept();
    resizeEventSignal(ev);
    if(isVisible() && isExposed())
    {
        m_visibleAndExposed.store(true);
        update();
    }
    else
    {
        m_visibleAndExposed.store(false);
    }
}

void View::mouseMoveEvent(QMouseEvent* ev)
{
    mouseMoveEventSignal(ev);
}

void View::mousePressEvent(QMouseEvent* ev)
{
    mousePressEventSignal(ev);
}

void View::wheelEvent(QWheelEvent* ev)
{
    wheelEventSignal(ev);
}
