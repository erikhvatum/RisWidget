#include <QPointer>
#include <QtCore>
#include <QtWidgets>

class ImageWidget;

class GV
  : public QGraphicsView
{
public:
    GV(QGraphicsScene* scene, QWidget* parent, ImageWidget* image_widget);

protected:
    ImageWidget* m_image_widget;
    virtual void drawBackground(QPainter * painter, const QRectF & rect);
    virtual void resizeEvent(QResizeEvent* event);
};
