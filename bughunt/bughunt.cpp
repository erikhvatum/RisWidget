#include "bughunt.h"


BugHunt::BugHunt(QWidget* parent)
  : QDialog(parent)
{
    PyGILState_STATE s = PyGILState_Ensure();
//  main = py::object(( py::handle<>(py::borrowed(PyImport_AddModule("__main__")))));
    PyImport_AddModule("__main__");
    PyErr_Print();
//  mainNamespace = main.attr("__dict__");
//  main = py::import("__main__");
//  mainNamespace = main.attr("__dict__");
    PyGILState_Release(s);
    setLayout(new QHBoxLayout);
    m_leftButton = new QPushButton("left button");
    m_rightButton = new QPushButton("right button");
    layout()->addWidget(m_leftButton);
    layout()->addWidget(m_rightButton);
    connect(m_leftButton, &QPushButton::clicked, this, &BugHunt::onLeftButton);
    connect(m_rightButton, &QPushButton::clicked, this, &BugHunt::onRightButton);
}

BugHunt::~BugHunt()
{
}

void BugHunt::onLeftButton()
{
    std::cerr << "left button pressed\n";
}

void BugHunt::onRightButton()
{
    py::eval("print('hello world')", mainNamespace);
}

void foo()
{
    BugHunt* b = new BugHunt;
    delete b;
}
