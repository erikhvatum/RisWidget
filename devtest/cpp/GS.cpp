#include <iostream>
#include "GS.h"

GS::GS(QObject* parent) : QGraphicsScene(parent)
{
}

bool GS::event(QEvent* e)
{
    /*if(e->type() == QEvent::GraphicsSceneMouseMove)
    {
        QGraphicsSceneMouseEvent* se = dynamic_cast<QGraphicsSceneMouseEvent*>(e);
        QPointF np(se->scenePos());
        np.rx() += 15;
        np.ry() += 25;
        se->setScenePos(np);
    }*/
    return QGraphicsScene::event(e);
}

void GS::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    std::cout << "event->scenePos(): " << event->scenePos().x() << ", " << event->scenePos().y() << std::endl;
    QGraphicsScene::mouseMoveEvent(event);
}