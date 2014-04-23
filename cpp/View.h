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

    // Call this thread-safe function to refresh view contents.  The refresh is queued and happens when the Renderer
    // thread gets around to it.  Multiple calls to update made for a single view while the Renderer is busy coalesce
    // into a single refresh.  If this View is not attached to a Renderer, update() is a no-op.
    void update();

signals:
    void resizeEventSignal(QResizeEvent* ev);
    void mouseMoveEventSignal(QMouseEvent* ev);
    void mousePressEventSignal(QMouseEvent* ev);
    void mouseEnterExitSignal(bool entered);

protected:
    QPointer<QOpenGLContext> m_context;
    QPointer<Renderer> m_renderer;

    virtual bool event(QEvent* ev) override;
    virtual void resizeEvent(QResizeEvent* ev) override;
    virtual void exposeEvent(QExposeEvent* ev) override;
    virtual void mouseMoveEvent(QMouseEvent* ev) override;
    virtual void mousePressEvent(QMouseEvent* ev) override;
};
