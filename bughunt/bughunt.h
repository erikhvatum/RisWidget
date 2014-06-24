#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/import.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/module.hpp>
#include <boost/python/manage_new_object.hpp>
#include <iostream>
#include <Python.h>
#include <QAbstractScrollArea>
#include <QApplication>
#include <QComboBox>
#include <QDoubleValidator>
#include <QFile>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QMainWindow>
#include <QMessageBox>
#include <QMutex>
#include <QMutexLocker>
#include <QPointer>
#include <QPushButton>
#include <QResizeEvent>
#include <QScreen>
#include <QScrollBar>
#include <QSharedPointer>
#include <QStatusBar>
#include <QString>
#include <QSurface>
#include <QThread>
#include <QTimer>
#include <QToolBar>
#include <QVector>
#include <QWindow>

namespace py = boost::python;

class BugHunt
  : public QDialog
{
    Q_OBJECT;

public:
    explicit BugHunt(QWidget* parent=nullptr);

    virtual ~BugHunt();

protected:
    QPushButton* m_leftButton;
    QPushButton* m_rightButton;
    py::object main;
    py::object mainNamespace;

protected slots:
    void onLeftButton();

    void onRightButton();
};
