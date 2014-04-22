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

#include "Common.h"
#include "RisWidget.h"

namespace py = boost::python;
namespace np = boost::numpy;

RisWidget::RisWidget(QString windowTitle_,
                     QWidget* parent,
                     Qt::WindowFlags flags)
  : QMainWindow(parent, flags)
{
    static bool oneTimeInitDone{false};
    if(!oneTimeInitDone)
    {
        Q_INIT_RESOURCE(RisWidget);
        np::initialize();
        oneTimeInitDone = true;
    }

    setWindowTitle(windowTitle_);
    setupUi(this);
    Renderer::staticInit();
    setupActions();
    makeToolBars();
    makeViews();
    makeRenderer();

    try
    {
        m_numpy = py::import("numpy");
    }
    catch(py::error_already_set const&)
    {
        PyErr_Print();
        throw RisWidgetException("RisWidget constructor: Failed to import numpy Python module.");
    }

#ifdef STAND_ALONE_EXECUTABLE
    setAttribute(Qt::WA_DeleteOnClose, true);
    QApplication::setQuitOnLastWindowClosed(true);
#else
    Py_Initialize();
    PyEval_InitThreads();
#endif
}

RisWidget::~RisWidget()
{
}

void RisWidget::setupActions()
{
    m_modeGroup = new QActionGroup(this);
    m_modeGroup->addAction(m_actionPanMode);
    m_modeGroup->addAction(m_actionZoomMode);

    connect(m_actionPanMode,  &QAction::triggered, [&](){setViewMode(ViewMode::Pan );});
    connect(m_actionZoomMode, &QAction::triggered, [&](){setViewMode(ViewMode::Zoom);});

    m_actionPanMode->setChecked(true);
}

void RisWidget::makeToolBars()
{
    m_viewToolBar = addToolBar("View");
    m_zoomCombo = new QComboBox(this);
    m_viewToolBar->addWidget(m_zoomCombo);
    m_zoomCombo->setEditable(true);
    m_zoomCombo->setInsertPolicy(QComboBox::NoInsert);
    m_zoomCombo->setDuplicatesEnabled(true);
    m_zoomCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    for(const GLfloat& z : sm_zoomPresets)
    {
        m_zoomCombo->addItem(formatZoom(z * 100.0f) + '%');
    }
    m_zoomCombo->setCurrentIndex(1);
    m_zoomComboValidator = new QDoubleValidator(sm_zoomMinMax[0], sm_zoomMinMax[1], 4, m_zoomCombo);
    connect(m_zoomCombo, SIGNAL(activated(int)), this, SLOT(zoomComboChanged(int)));
    connect(m_zoomCombo->lineEdit(), SIGNAL(returnPressed()), this, SLOT(zoomComboCustomValueEntered()));
    m_viewToolBar->addSeparator();
    m_viewToolBar->addAction(m_actionPanMode);
    m_viewToolBar->addAction(m_actionZoomMode);
}

void RisWidget::makeViews()
{
    m_imageWidget->makeView();
    connect(m_imageWidget->imageView(), &ImageView::mouseMoveEventSignal, this, &RisWidget::mouseMoveEventInImageView);
    m_histogramWidget->makeView();

    m_imageWidget->imageView()->setClearColor(glm::vec4(1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 0.0f));
    m_histogramWidget->histogramView()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));
}

void RisWidget::makeRenderer()
{
    m_rendererThread = new QThread(this);
    m_renderer.reset( new Renderer(m_imageWidget->imageView(),
                                   m_histogramWidget->histogramView()) );
    m_renderer->moveToThread(m_rendererThread);
    connect(m_rendererThread.data(), &QThread::started, m_renderer.get(), &Renderer::threadInitSlot, Qt::QueuedConnection);
    m_rendererThread->start();
}

ImageWidget* RisWidget::imageWidget()
{
    return m_imageWidget;
}

HistogramWidget* RisWidget::histogramWidget()
{
    return m_histogramWidget;
}

void RisWidget::showImage(const GLushort* imageDataRaw, const QSize& imageSize, bool filterTexture)
{
    std::size_t byteCount = sizeof(GLushort) *
                            static_cast<std::size_t>(imageSize.width()) *
                            static_cast<std::size_t>(imageSize.height());
    ImageData imageData(byteCount);
    memcpy(reinterpret_cast<void*>(imageData.data()),
           reinterpret_cast<const void*>(imageDataRaw),
           byteCount);
    m_renderer->showImage(imageData, imageSize, filterTexture);
}

void RisWidget::showImage(PyObject* image, bool filterTexture)
{
    py::object imagepy{py::handle<>(py::borrowed(image))};
    np::ndarray imagenp{np::from_object(imagepy, np::dtype::get_builtin<GLushort>(), 2, 2, np::ndarray::CARRAY_RO)};
    if(imagenp.is_none())
    {
        throw RisWidgetException("RisWidget::showImage(PyObject* image): image argument must be an "
                                 "array-like object convertable to a 2d uint16 numpy array.");
    }
    const Py_intptr_t* shape = imagenp.get_shape();
    showImage(reinterpret_cast<const GLushort*>(imagenp.get_data()), QSize(shape[1], shape[0]), filterTexture);
}

PyObject* RisWidget::getHistogram()
{
    auto histogramData = m_renderer->getHistogram();
    std::unique_ptr<np::ndarray> ret;

    Py_intptr_t size = histogramData->ref().size();
    if(size != 0)
    {
        ret.reset(new np::ndarray(np::empty(1, &size, np::dtype::get_builtin<GLuint>())));
        memcpy(reinterpret_cast<void*>(ret->get_data()),
               reinterpret_cast<const void*>(histogramData->ref().data()),
               sizeof(GLuint) * histogramData->ref().size());
    }

    // boost::python is not managing the copy of the pointer we are returning; we must incref so that it is not garbage
    // collected when ret goes out of scope and its destructor decrefs its internal copy of the pointer.
    Py_INCREF(ret->ptr());
    return ret->ptr();
}

RisWidget::ViewMode RisWidget::viewMode() const
{
    return m_viewMode;
}

void RisWidget::setViewMode(RisWidget::ViewMode viewMode)
{
    m_viewMode = viewMode;
    statusBar()->showMessage(viewMode == ViewMode::Pan ? "pan" : "zoom");
}

QString RisWidget::formatZoom(const GLfloat& z)
{
    QString ret;
    if(z == floor(z))
    {
        ret = QString::number(z, 'f', 0);
    }
    else
    {
        ret = QString::number(z, 'f', 2);
        if(ret.endsWith('0'))
        {
            ret.chop(1);
        }
    }
    return std::move(ret);
}

void RisWidget::loadFile()
{
    QString fnqstr(QFileDialog::getOpenFileName(this, "Open Image or Numpy Array File", QString(), "Numpy Array Files (*.npy)"));
    if(!fnqstr.isNull())
    {
        if(m_numpyLoad.is_none())
        {
            m_numpyLoad = m_numpy.attr("load");
            if(m_numpyLoad.is_none())
            {
                throw RisWidgetException("RisWidget::loadFile(): Failed to resolve Python function numpy.load(..).");
            }
        }
        std::string fnstdstr{fnqstr.toStdString()};
        try
        {
            py::object ret{m_numpyLoad(fnstdstr.c_str())}; 
            showImage(ret.ptr());
        }
        catch(py::error_already_set const&)
        {
            PyErr_Print();
        }
    }
}

void RisWidget::mouseMoveEventInImageView(QMouseEvent* event)
{
    statusBar()->showMessage(QString("%1, %2").arg(event->x()).arg(event->y()));
}

void RisWidget::zoomComboCustomValueEntered()
{
    QString text(m_zoomCombo->lineEdit()->text());
    QString scaleText(text.left(text.indexOf("%")));

    bool valid(true);
    int zero(0);
    switch(m_zoomComboValidator->validate(scaleText, zero))
    {
    case QValidator::Intermediate:
    case QValidator::Invalid:
        QMessageBox::information(this, "RisWidget", QString("Please enter a number between %1 and %2.").arg(formatZoom(sm_zoomMinMax[0])).arg(formatZoom(sm_zoomMinMax[1])));
        m_zoomCombo->setFocus();
        m_zoomCombo->lineEdit()->selectAll();
        valid = false;
        break;
    }

    if(valid)
    {
        bool converted;
        double scalePercent(scaleText.toDouble(&converted));
        if(!converted)
        {
            throw RisWidgetException(std::string("RisWidget::zoomComboCustomValueEntered(): scaleText.toDouble(..) failed for \"") +
                                     scaleText.toStdString() + "\".");
        }
        setCustomZoom(scalePercent * .01);
    }
}

void RisWidget::zoomComboChanged(int index)
{
    setZoomIndex(index);
}

void RisWidget::zoomChanged(int zoomIndex, GLfloat customZoom)
{
    if(zoomIndex != -1 && customZoom != 0)
    {
        throw RisWidgetException("RisWidget::zoomChanged(..): Both zoomIndex and customZoom specified.");
    }
    if(zoomIndex == -1 && customZoom == 0)
    {
        throw RisWidgetException("RisWidget::zoomChanged(..): Neither zoomIndex nor customZoom specified.");
    }

    if(zoomIndex != -1)
    {
        if(zoomIndex < 0 || zoomIndex >= m_zoomCombo->count())
        {
            throw RisWidgetException("RisWidget::zoomChanged(..): Invalid value for zoomIndex.");
        }
        m_zoomCombo->setCurrentIndex(zoomIndex);
    }
    else
    {
        m_zoomCombo->lineEdit()->setText(formatZoom(customZoom * 100.0f) + '%');
    }
}

#ifdef STAND_ALONE_EXECUTABLE

void RisWidget::closeEvent(QCloseEvent* event)
{
    event->accept();
    if(m_rendererThread)
    {
        m_rendererThread->quit();
        m_rendererThread->wait();
    }
}

#include <QApplication>

int main(int argc, char** argv)
{
    int ret{0};
    QString argv0(argv[0]);
    std::wstring argv0std(argv0.toStdWString());
    Py_SetProgramName(const_cast<wchar_t*>(argv0std.c_str()));
    Py_Initialize();
    PyEval_InitThreads();
    PyObject* mainModule = PyImport_AddModule("__main__");
    if(mainModule == nullptr)
    {
        ret = -1;
        std::cerr << "int main(int argc, char** argv): PyImport_AddModule(\"__main__\") failed." << std::endl;
    }
    else
    {
        QApplication app(argc, argv);
        RisWidget* risWidget{new RisWidget};
        risWidget->show();
        ret = app.exec();
    }
    Py_Finalize();
    return ret;
}

#endif

