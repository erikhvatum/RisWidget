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

#pragma once

#include "Common.h"
#include "Image.h"
#include "ui_Flipper.h"

class RisWidget;

class Flipper
  : public QWidget,
    protected Ui::Flipper
{
    Q_OBJECT;
    Q_PROPERTY(QString flipperName
                   READ getFlipperName
                   WRITE setFlipperName
                   NOTIFY flipperNameChanged);
    Q_PROPERTY(int frameIndex
                   READ getFrameIndex
                   WRITE setFrameIndex
                   NOTIFY frameIndexChanged);
    Q_PROPERTY(int frameCount
                   READ getFrameCount
                   NOTIFY frameCountChanged);

public:
    Flipper(QDockWidget* parent, RisWidget* rw, const QString& flipperName);
    virtual ~Flipper();

    QString getFlipperName() const;
    int getFrameIndex() const;
    int getFrameCount() const;

    void append(PyObject* images);

protected:
    struct Frame
    {
        enum class Type
        {
            File,
            Data,
            PyData
        };
        explicit Frame(const Type& type_);
        ~Frame();
        Type type;
        QString name;
        QSize size;
        ImageData data;
        PyObject* py_data;
    };
    typedef std::shared_ptr<Frame> FramePtr;
    typedef std::vector<FramePtr> Frames;

    QDockWidget* m_dw;
    RisWidget* m_rw;

    QString m_flipperName;
    bool m_alwaysStoreImagesInRam;
    int m_frameIndex;
    Frames m_frames;
    QTimer* m_nextFrameTimer;
    QDoubleValidator* m_fpsLimitValidator;
    bool m_isPlaying;
    bool m_wasPlayingBeforeSliderDrag;
    // Max frames per second
    float m_fpsLimit;
    // Min seconds per frame
    float m_spfLimit;
    float m_prevFrameShowDelta;

    void dragEnterEvent(QDragEnterEvent* event);
    void dragMoveEvent(QDragMoveEvent* event);
    void dragLeaveEvent(QDragLeaveEvent* event);
    void dropEvent(QDropEvent* event);

    void updateNextFrameTimer();
    void propagateFrameIndexChange();
    void propagateFrameCountChange();

signals:
    // Emitted during Flipper destruction.  QObject offers a similar signal, "destroyed", but this signal is emitted
    // from QObject's destructor.  By the time QObject's destructor is called, our destructor has executed and our
    // member variables, such as m_flipperName, are gone.  However, we need to access this member variable when
    // responding to Flipper close events, and so it is necessary to emit from our own destructor as its first order of
    // business.
    void closing(Flipper* flipper);
    void flipperNameChanged(Flipper* flipper, QString oldName);
    void frameIndexChanged(Flipper* flipper, int frameIndex);
    void frameCountChanged(Flipper* flipper, int frameCount);

public slots:
    void setFlipperName(const QString& flipperName);
    void setFrameIndex(int frameIndex);
    void incrementFrameIndex();

protected slots:
    void renameButtonClicked();
    void alwaysStoreImagesInRamToggled(bool alwaysStoreImagesInRam);
    void playbackButtonClicked(bool checked);
    void frameIndexSliderPressed();
    void frameIndexSliderReleased();
    void fpsLimitEdited(QString fpsLimitQStr);
};

