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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL RisWidget_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

Flipper::Frame::Frame(const Type& type_)
  : type(type_),
    py_data(nullptr)
{
}

Flipper::Frame::~Frame()
{
    if(py_data != nullptr)
    {
        GilLocker gilLock;
    }
}

Flipper::Flipper(QDockWidget* parent, RisWidget* rw, const QString& flipperName)
  : QWidget(parent),
    m_dw(parent),
    m_rw(rw),
    m_flipperName(flipperName),
    m_alwaysStoreImagesInRam(true),
    m_frameIndex(0),
    m_nextFrameTimer(new QTimer(this)),
    m_isPlaying(false),
    m_fpsLimit(std::numeric_limits<float>::max()),
    m_spfLimit(0),
    m_prevFrameShowDelta(0)
{
    setupUi(this);
    m_loadingGroupbox->hide();
    // TODO: implement loading of images as needed when possible and remove the following hide() call
    m_keepInRamCheckbox->hide();
    // Prevent playback button from changing size and causing relayout when button text is changed.  The button text is
    // initially (scrubbing), which is the longest string it needs to accomodate.
    int w{m_playbackButton->width()};
    if(w > 0)
    {
        m_playbackButton->setMinimumWidth(w);
        m_playbackButton->setMaximumWidth(w);
    }
    m_playbackButton->setText("Play");
    m_fpsLimitValidator = new QDoubleValidator(m_fpsLimitEdit);
    m_fpsLimitValidator->setRange(0.01, std::numeric_limits<float>::max());
    m_fpsLimitEdit->setValidator(m_fpsLimitValidator);
    fpsLimitEdited(m_fpsLimitEdit->text());
    m_nextFrameTimer->setSingleShot(true);
    connect(m_nextFrameTimer, &QTimer::timeout, this, &Flipper::incrementFrameIndex);
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
            throw RisWidgetException(QString("Flipper::setFlipperName(const QString& flipperName): A flipper with the "
                                             "name \"%1\" already exists.").arg(flipperName));
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
    if(frameIndex != m_frameIndex)
    {
        if(frameIndex < 0 || frameIndex >= getFrameCount())
        {
            QString e("Flipper::setFrameIndex(int frameIndex): The value supplied for frameIndex, "
                      "%1, is not in the valid range [0, %2].");
            throw RisWidgetException(e.arg(frameIndex).arg(std::max(getFrameCount() - 1, 0)));
        }
        m_frameIndex = frameIndex;
        propagateFrameIndexChange();
    }
}

void Flipper::incrementFrameIndex()
{
    if(m_frames.size() > 1)
    {
        ++m_frameIndex;
        if(m_frameIndex >= m_frames.size())
        {
            m_frameIndex = 0;
        }
        propagateFrameIndexChange();
    }
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

void Flipper::playbackButtonClicked(bool checked)
{
    if(checked)
    {
        m_isPlaying = true;
        m_playbackButton->setText("Stop");
        updateNextFrameTimer();
    }
    else
    {
        m_nextFrameTimer->stop();
        m_isPlaying = false;
        m_playbackButton->setText("Play");
    }
}

void Flipper::frameIndexSliderPressed()
{
    if(m_isPlaying)
    {
        m_nextFrameTimer->stop();
        m_playbackButton->setText("(scrubbing)");
        m_wasPlayingBeforeSliderDrag = true;
        m_isPlaying = false;
    }
    else
    {
        m_wasPlayingBeforeSliderDrag = false;
    }
}

void Flipper::frameIndexSliderReleased()
{
    if(m_wasPlayingBeforeSliderDrag)
    {
        m_playbackButton->setText("Stop");
        m_isPlaying = true;
        updateNextFrameTimer();
    }
    else
    {
        m_playbackButton->setText("Play");
    }
}

void Flipper::fpsLimitEdited(QString fpsLimitQStr)
{
    bool converted{false};
    float fpsLimit{fpsLimitQStr.toFloat(&converted)};
    if(converted && m_fpsLimit != fpsLimit)
    {
        m_fpsLimit = fpsLimit;
        // QTimer accepts interval in milliseconds as int, meaning that it can not accept an interval greater than the
        // largest int value milliseconds, or largest int value / 1000 seconds
        m_spfLimit = (m_fpsLimit == 0) ? std::numeric_limits<int>::max() / 1000.0f
                                       : 1.0f / m_fpsLimit;
        updateNextFrameTimer();
    }
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
    bool hadNoFrames{m_frames.empty()};

    if(md->hasImage())
    {
        // Raw image data is preferred in the case where both image data and source URL are present.  This is the case,
        // for example, on OS X when an image is dragged from Firefox.
        accept = true;
        QImage rgbImage(md->imageData().value<QImage>().convertToFormat(QImage::Format_RGB888));
        FramePtr frame(new Frame(Frame::Type::Data));
        frame->name = md->hasUrls() ? md->urls()[0].toString() : QString::number(m_frames.size());
        frame->type = Frame::Type::File;
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
                    FramePtr frame(new Frame(Frame::Type::File));
                    frame->name = fn;
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
        if(hadNoFrames)
        {
            // Show the first frame if we had no frames before the drag and drop operation
            propagateFrameIndexChange();
        }
    }
}

void Flipper::updateNextFrameTimer()
{
    if(m_isPlaying)
    {
        float elapsed{m_nextFrameTimer->isActive() ?
                      std::max(1000.0f * (m_nextFrameTimer->interval() - m_nextFrameTimer->remainingTime()), 0.0f) :
                      0.0f};
        float nextFrameWait{std::max(m_spfLimit - elapsed - m_prevFrameShowDelta, 0.0f)};
        m_nextFrameTimer->start(static_cast<int>(nextFrameWait * 1000));
    }
}

void Flipper::propagateFrameIndexChange()
{
    Frame& frame(*m_frames[m_frameIndex]);
    m_frameListbox->setCurrentRow(m_frameIndex);
    m_frameIndexSlider->setValue(m_frameIndex);
    m_frameIndexSpinner->setValue(m_frameIndex);
    std::chrono::steady_clock::time_point preShowTs(std::chrono::steady_clock::now());
    m_rw->showImage(frame.data.get(), frame.size);
    std::chrono::steady_clock::time_point postShowTs(std::chrono::steady_clock::now());
    std::chrono::duration<float> showDelta(std::chrono::duration_cast<std::chrono::duration<float>>(postShowTs - preShowTs));
    m_prevFrameShowDelta = showDelta.count();
    updateNextFrameTimer();
    frameIndexChanged(this, m_frameIndex);
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
    int m{std::max(getFrameCount() - 1, 0)};
    m_frameIndexSpinner->setMaximum(m);
    m_frameIndexSlider->setMaximum(m);
    m_frameIndexSlider->setTickInterval(1);
    frameCountChanged(this, static_cast<int>(m_frames.size()));
}
