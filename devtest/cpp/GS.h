#pragma once

#include <QPointer>
#include <QtCore>
#include <QtWidgets>

class GS : public QGraphicsScene
{
    Q_OBJECT;

public:
    GS(QObject* parent=0);
    bool event(QEvent* e);

protected:
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event);
};