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
#include "RisWidget.h"
#include "ViewWidget.h"

ViewWidget::ViewWidget(QWidget* parent)
  : QWidget(parent),
    m_lock(new QMutex(QMutex::Recursive)),
    m_clearColor(0.0f, 0.0f, 0.0f, 0.0f),
    m_viewSize(-1, -1),
    m_viewGlSize(-2, -2)
{
}

ViewWidget::~ViewWidget()
{
    delete m_lock;
}

View* ViewWidget::view()
{
    return m_view;
}

QWidget* ViewWidget::viewContainerWidget()
{
    return m_viewContainerWidget;
}

void ViewWidget::setClearColor(glm::vec4&& color)
{
    QMutexLocker locker(m_lock);
    m_clearColor = std::move(color);
    update();
}

void ViewWidget::setClearColor(const glm::vec4& color)
{
    QMutexLocker locker(m_lock);
    m_clearColor = color;
    update();
}

glm::vec4 ViewWidget::clearColor() const
{
    QMutexLocker locker(m_lock);
    glm::vec4 ret(m_clearColor);
    return std::move(ret);
}

void ViewWidget::makeView(bool doAddWidget, QWidget* parent)
{
    if(parent == nullptr)
    {
        parent = this;
    }
    QMutexLocker locker(m_lock);
    if(m_view || m_viewContainerWidget)
    {
        throw RisWidgetException("ViewWidget::makeView(): View already created.  makeView() must not be "
                                 "called more than once per ViewWidget instance.");
    }
    if(doAddWidget && layout() == nullptr)
    {
        QHBoxLayout* layout_(new QHBoxLayout);
        setLayout(layout_);
    }
    // Note: calling winId() causes an underlying QWindow to be created immediately rather than lazily, and we need the
    // QWindow in order that the GL surface may be properly parented to us
    parent->winId();
#ifdef _DEBUG
    if(parent->windowHandle() == nullptr)
    {
        throw RisWidgetException("ViewWidget::makeView(..): winId() failed to provoke instantiation of underlying QWindow.");
    }
#endif
    m_view = instantiateView(parent);
    connect(m_view.data(), &View::resizeEventSignal, this, &ViewWidget::resizeEventInView);
    m_viewContainerWidget = QWidget::createWindowContainer(m_view, parent, Qt::Widget);
    if(doAddWidget)
    {
        layout()->addWidget(m_viewContainerWidget);
        m_viewContainerWidget->show();
    }
}

void ViewWidget::resizeEventInView(QResizeEvent* ev)
{
    QMutexLocker locker(m_lock);
    m_viewSize = ev->size();
}

// void ViewWidget::showViewWhenTheTimeIsRight()
// {
//     RisWidget* rw{nullptr};
//     QWidget* p{this};
//     // Ascend widget heirarchy until we find our parent RisWidget
//     while((p = dynamic_cast<QWidget*>(p->parent())) != nullptr && (rw = dynamic_cast<RisWidget*>(p)) == nullptr);
//     if(rw == nullptr)
//     {
//         // We never found it, and we ran out of progenitors to interrogate.  Oh well, just show the GL surfaces now.
//         // This may result in the GL surfaces floating around detached until the owning RisWidget instance (that we for
//         // some reason couldn't find) has show() called or an equivalent action occurs.  We tried our best, but there's
//         // no avoiding the possibility of transient floaty GL surfaces if we're in this if statement block.
//         m_view->show();
//     }
//     else
//     {
//         // Listen for polish request received signal.  When it is emitted, show the GL surface and then, because the
//         // floatly problem only occurs before the first becoming-visible, stop listening for subsequent polish requests
//         QMetaObject::Connection listenConnection;
//         listenConnection = connect(rw, &RisWidget::polishRequestReceived, [&](){m_view->show(); rw->disconnect(listenConnection);});
//     }
// }
