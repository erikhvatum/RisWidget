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

#pragma once

#include "Common.h"

class Renderer;

class View
  : public QWindow
{
    Q_OBJECT;
    friend class Renderer;

public:
    explicit View(QWindow* parent);
    virtual ~View();

    QOpenGLContext* context();
    void makeCurrent();
    void swapBuffers();
    void setClearColor(const glm::vec4& color);
    glm::vec4 clearColor() const;

    // Call this thread-safe function to refresh view contents.  The refresh is queued and happens when the Renderer
    // thread gets around to it.  Multiple calls to update made for a single view while the Renderer is busy coalesce
    // into a single refresh.  If this View is not attached to a Renderer, update() is a no-op.
    void update();

protected:
    QPointer<QOpenGLContext> m_context;
    QPointer<Renderer> m_renderer;

    QMutex* m_clearColorLock;
    glm::vec4 m_clearColor{0.0f, 0.0f, 0.0f, 0.0f};

    QMutex* m_sizeLock;
    QSize m_size, m_glSize;

    virtual void resizeEvent(QResizeEvent* event);
    virtual void exposeEvent(QExposeEvent* event);
};
