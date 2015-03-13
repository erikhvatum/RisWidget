#include "GV.h"
#include "ImageWidget.h"
#include <iostream>

GV::GV(QGraphicsScene* scene, QWidget* parent, ImageWidget* image_widget)
  : QGraphicsView(scene, parent),
    m_image_widget(image_widget)
{
    setDragMode(ScrollHandDrag);
    setTransformationAnchor(AnchorUnderMouse);
    setResizeAnchor(AnchorViewCenter);
}

void GV::drawBackground(QPainter * painter, const QRectF & rect)
{
    m_image_widget->paint(painter, rect);
    QGraphicsView::drawBackground(painter, rect);
}

void GV::resizeEvent(QResizeEvent* event)
{
    QGraphicsView::resizeEvent(event);
    std::cout << "GV::resizeEvent(..) w: " << m_image_widget->size().width() << " h: " << m_image_widget->size().height() << std::endl;
}
