#include <QOpenGLFunctions_2_0>
#include <QPointer>
#include <QtCore>
#include <QtWidgets>

class ImageWidget
  : public QOpenGLWidget,
    public QOpenGLFunctions_2_0
{
public:
    ImageWidget(QWidget* parent);

    virtual void initializeGL();
    void paint(QPainter* painter, const QRectF& rect);
    virtual void resizeEvent(QResizeEvent* event);
    virtual void resizeGL(int w, int h);
};
