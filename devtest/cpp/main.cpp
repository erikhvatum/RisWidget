#include <QApplication>
#include <QPointer>
#include <QtCore>
#include <QtWidgets>
#include "GV.h"
#include "ImageWidget.h"

int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    QMainWindow* mw(new QMainWindow);
    ImageWidget* iw(new ImageWidget(mw));
    QGraphicsScene* gs(new QGraphicsScene);
    QColor c = QColor(Qt::red);
    c.setAlphaF(0.5);
    QBrush b = QBrush(c);
    gs->addRect(10, 10, 100, 100, QPen(Qt::blue), b);
    GV* gv(new GV(gs, mw, iw));
    gv->setCacheMode(QGraphicsView::CacheNone);
    gv->setViewport(iw);
    mw->setCentralWidget(gv);
    mw->show();
    return app.exec();
}
