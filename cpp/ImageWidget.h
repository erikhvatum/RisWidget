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
#include "ImageView.h"
#include "Renderer.h"
#include "ui_ImageWidget.h"
#include "ViewWidget.h"

class RisWidget;

// Click-zooming jumps between preset values until the preset value range is exceeded; thereafter, each click scales
// by sm_zoomClickScaleFactor.
class ImageWidget
  : public ViewWidget,
    protected Ui::ImageWidget
{
    Q_OBJECT;
    friend class Renderer;
    friend class RisWidget;

public:
    enum class InteractionMode
    {
        Pointer,
        Pan,
        Zoom
    };
    static const std::vector<GLfloat> sm_zoomPresets;
    static const std::uint8_t sm_defaultZoomPreset;
    static const std::pair<GLfloat, GLfloat> sm_zoomMinMax;
	static const GLfloat sm_zoomClickScaleFactor;

    explicit ImageWidget(QWidget* parent = nullptr);
    virtual ~ImageWidget();

    ImageView* imageView();

    InteractionMode interactionMode() const;
    void setInteractionMode(InteractionMode interactionMode);

    // Returns the zoom level where, for example, 1.0=100% and 2.0=200%.  Returns 0 if the view is zoomed to one of the
    // preset zoom levels.
	GLfloat customZoom() const;
	// Returns the current preset zoom level index, or -1 if the view is zoomed to a custom level
	int zoomIndex() const;
	void setCustomZoom(GLfloat customZoom);
	void setZoomIndex(int zoomIndex);
    bool zoomToFit() const;
    void setZoomToFit(bool zoomToFit);

signals:
    void interactionModeChanged(InteractionMode interactionMode, InteractionMode previousInteractionMode);
    void pointerMovedToDifferentPixel(bool isOnPixel, QPoint pixelCoord, GLushort pixelValue);
    void zoomChanged(int zoomIndex, GLfloat customZoom);

protected:
    InteractionMode m_interactionMode;
    int m_zoomIndex;
    GLfloat m_customZoom;
    bool m_zoomToFit;
    QPoint m_pan;
    QSize m_imageSize;
    ImageData m_imageData;

    virtual void makeView(bool doAddWidget = true) override;
    virtual View* instantiateView() override;
    void updateImageSizeAndData(const QSize& imageSize, const ImageData& imageData);
    virtual void resizeEventInView(QResizeEvent* ev) override;
    void updateScrollerRanges();

protected slots:
    void scrollViewContentsBy(int dx, int dy);
    void mousePressEventInView(QMouseEvent* ev);
    void mouseMoveEventInView(QMouseEvent* ev);
    void mouseEnterExitView(bool entered);
    void wheelEventInView(QWheelEvent* ev);
};
