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
#include "ViewWidget.h"

ViewWidget::ViewWidget(QWidget* parent)
  : QWidget(parent),
    m_lock(new QMutex(QMutex::Recursive)),
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

void View::setClearColor(const glm::vec4& color)
{
    QMutexLocker locker(m_lock);
    m_clearColor = color;
    update();
}

glm::vec4 View::clearColor() const
{
    QMutexLocker locker(m_lock);
    glm::vec4 ret{m_clearColor};
    return ret;
}

void ViewWidget::makeView(bool doAddWidget)
{
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
    m_view = instantiateView();
    connect(m_view.data(), &View::resizeEventSignal, this, &ViewWidget::resizeEventInView);
    m_viewContainerWidget = QWidget::createWindowContainer(m_view, this, Qt::Widget);
    if(doAddWidget)
    {
        layout()->addWidget(m_viewContainerWidget);
        m_viewContainerWidget->show();
        m_view->show();
    }
}

void ViewWidget::resizeEventInView(QResizeEvent* ev)
{
    QMutexLocker locker(m_lock);
    m_viewSize = ev.size();
}
