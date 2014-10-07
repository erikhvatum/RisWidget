// The MIT License (MIT)
// 
// Copyright (c) 2014 WUSTL ZPLAB
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
// 
// Authors: Erik Hvatum

#include "Common.h"
#include "Flipper.h"
#include "RenameFlipper.h"
#include "RisWidget.h"

Flipper::Flipper(QDockWidget* parent, RisWidget* rw, const QString& flipperName)
  : QWidget(parent),
    m_dw(parent),
    m_rw(rw),
    m_flipperName(flipperName),
    m_alwaysStoreImagesInRam(true),
    m_frameIndex(0)
{
    setupUi(this);
}

Flipper::~Flipper()
{
    closing(this);
}

QString Flipper::getFlipperName() const
{
    return m_flipperName;
}

void Flipper::setFlipperName(const QString& flipperName)
{
    if(flipperName != m_flipperName)
    {
        if(m_rw->hasFlipper(flipperName))
        {
            std::ostringstream o;
            o << "Flipper::setFlipperName(const QString& flipperName): ";
            o << "A flipper with the name \"" << flipperName.toStdString() << "\" already exists.";
            throw RisWidgetException(o.str());
        }
        QString oldName = m_flipperName;
        m_flipperName = flipperName;
        // Do not change Flipper's parent's window title if Flipper has been reparented
        if(m_dw != nullptr && parent() != nullptr && m_dw == qobject_cast<QDockWidget*>(parent()))
        {
            m_dw->setWindowTitle(QString("Flipbook (%1)").arg(m_flipperName));
        }
        flipperNameChanged(this, oldName);
    }
}

int Flipper::getFrameIndex() const
{
    return m_frameIndex;
}

int Flipper::getFrameCount() const
{
    return static_cast<int>(m_frames.size());
}

void Flipper::setFrameIndex(int frameIndex)
{
}

void Flipper::incrementFrameIndex()
{
}

void Flipper::renameButtonClicked()
{
    RenameFlipper renameFlipper(this, this, m_rw);
    renameFlipper.exec();
}

void Flipper::alwaysStoreImagesInRamToggled(bool alwaysStoreImagesInRam)
{
    if(alwaysStoreImagesInRam != m_alwaysStoreImagesInRam)
    {
        m_alwaysStoreImagesInRam = alwaysStoreImagesInRam;
    }
}

void Flipper::playbackButtonClicked()
{
}

void Flipper::frameIndexSliderPressed()
{
}

void Flipper::frameIndexSliderReleased()
{
}

void Flipper::dragEnterEvent(QDragEnterEvent* event)
{
    event->acceptProposedAction();
}

void Flipper::dragMoveEvent(QDragMoveEvent* event)
{
    event->acceptProposedAction();
}

void Flipper::dragLeaveEvent(QDragLeaveEvent* event)
{
    event->accept();
}

void Flipper::dropEvent(QDropEvent* event)
{
    const QMimeData* md{event->mimeData()};
    bool accept{false};

    if(md->hasImage())
    {
        // Raw image data is preferred in the case where both image data and source URL are present.  This is the case,
        // for example, on OS X when an image is dragged from Firefox.
        accept = true;
        QImage rgbImage(md->imageData().value<QImage>().convertToFormat(QImage::Format_RGB888));
        FramePtr frame(new Frame);
        frame->name = md->hasUrls() ? md->urls()[0].toString() : QString::number(m_frames.size());
        frame->isFile = false;
        frame->size = rgbImage.size();
        frame->data.reset(new GLushort[rgbImage.width() * rgbImage.height()]);
        const GLubyte* rgbIt{rgbImage.bits()};
        const GLubyte* rgbItE{rgbIt + rgbImage.width() * rgbImage.height() * 3};
        for(GLushort* gsIt{frame->data.get()}; rgbIt != rgbItE; ++gsIt, rgbIt += 3)
        {
            *gsIt = GLushort(256) * static_cast<GLushort>(0.2126f * rgbIt[0] + 0.7152f * rgbIt[1] + 0.0722f * rgbIt[2]);
        }
        m_frames.push_back(frame);
        m_frameListbox->addItem(frame->name);
        propagateFrameCountChange();
    }
    else if(md->hasUrls())
    {
        for(QUrl& url : md->urls())
        {
            if(url.isLocalFile())
            {
                QString fn(url.toLocalFile());
                fipImage image;
                std::string fnstdstr(fn.toStdString());
                if(image.load(fnstdstr.c_str()) && image.convertToUINT16())
                {
                    accept = true;
                    FramePtr frame(new Frame);
                    frame->name = fn;
                    frame->isFile = true;
                    frame->data.reset(new GLushort[image.getWidth() * image.getHeight()]);
                    frame->size.setWidth(image.getWidth());
                    frame->size.setHeight(image.getHeight());
                    memcpy(reinterpret_cast<void*>(frame->data.get()),
                           reinterpret_cast<const void*>(image.accessPixels()),
                           image.getWidth() * image.getHeight() * 2);
                    m_frames.push_back(frame);
                    m_frameListbox->addItem(frame->name);
                    propagateFrameCountChange();
                }
            }
        }
    }

    if(accept)
    {
        event->acceptProposedAction();
    }
}

void Flipper::propagateFrameCountChange()
{
    bool enable{m_frames.size() > 0};
    if(enable != m_frameIndexSlider->isEnabled())
    {
        m_frameIndexSlider->setEnabled(enable);
        m_playbackButton->setEnabled(enable);
        m_frameIndexLabel->setEnabled(enable);
        m_frameIndexSpinner->setEnabled(enable);
    }
    int m{static_cast<int>(m_frames.size())};
    m_frameIndexSpinner->setMaximum(m);
    m_frameIndexSlider->setMaximum(m);
    frameCountChanged(this, m);
}
