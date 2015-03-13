#include <iostream>
#include <QApplication>
#include <QOpenGLFunctions_2_0>
#include <QPointer>
#include <QtCore>
#include <QtWidgets>

class GLW
  : public QOpenGLWidget,
    public QOpenGLFunctions_2_0
{
public:
    GLW(QWidget* parent=0)
      : QOpenGLWidget(parent)
    {
    }

    virtual void initializeGL()
    {
        if(!initializeOpenGLFunctions())
        {
            std::cerr << "initializeOpenGLFunctions failed.\n";
        }
    }

    virtual void paintGL()
    {
        QPainter p;
        p.begin(this);

        p.beginNativePainting();
        glClearColor(0,0,0,1);
        glClearDepth(1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        p.endNativePainting();

        QColor color(Qt::red);
        color.setAlphaF(0.5);
        QBrush brush(color);
        p.setBrush(brush);
        p.drawRect(10, 10, 100, 100);

        p.end();
    }

    virtual void resizeGL(int, int)
    {
    }
};

int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    GLW* glw(new GLW());
    glw->show();
    return app.exec();
}
