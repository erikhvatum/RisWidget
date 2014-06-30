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
#include "GilStateScopeOperators.h"
#include "RisWidget.h"
#include "ShowCheckerDialog.h"

RisWidget::RisWidget(QString windowTitle_,
                     QWidget* parent,
                     Qt::WindowFlags flags)
  : QMainWindow(parent, flags),
    m_showStatusBarPixelInfo(true),
    m_showStatusBarFps(false),
    m_previousFrameTimestampValid(false)
{
    static bool oneTimeInitDone{false};
    if(!oneTimeInitDone)
    {
        Q_INIT_RESOURCE(RisWidget);
#ifndef STAND_ALONE_EXECUTABLE
        Py_Initialize();
        PyEval_InitThreads();
        if(Py_IsInitialized() == 0)
        {
            std::cerr << "RisWidget Python module attempted to load while Python interpreter is not initialized (Py_IsInitialized() == 0).\n";
            Py_Exit(-1);
        }
#endif
        GilLocker gilLocker;
        oneTimeInitDone = true;
    }

    setWindowTitle(windowTitle_);
    setupUi(this);
    Renderer::staticInit();
    setupActions();
    makeToolBars();
    makeViews();
    makeRenderer();
    updateStatusBarPixelInfoPresence();
    updateStatusBarFpsPresence();

    {
        GilLocker gilLocker;
        m_numpyModule = PyImport_ImportModule("numpy");
        if(PyErr_Occurred() != nullptr)
        {
            PyErr_Print();
            Py_FatalError("Failed to import Python numpy module.  This program requires numpy.");
        }
        PyObject* loadpystr{PyUnicode_FromString("load")};
        m_numpyLoadFunction = PyObject_GetAttr(m_numpyModule, loadpystr);
        Py_XDECREF(loadpystr);
        if(PyErr_Occurred() != nullptr)
        {
            PyErr_Print();
            Py_FatalError("Imported numpy, but failed to get the numpy.load(..) function.  Your numpy installation is probably broken.");
        }
    }

#ifdef STAND_ALONE_EXECUTABLE
    setAttribute(Qt::WA_DeleteOnClose, true);
    QApplication::setQuitOnLastWindowClosed(true);
#endif
}

RisWidget::~RisWidget()
{
    GilLocker gilLocker;
    Py_XDECREF(m_numpyLoadFunction);
    Py_XDECREF(m_numpyModule);
}

void RisWidget::setupActions()
{
    m_imageViewInteractionModeGroup = new QActionGroup(this);
    m_imageViewInteractionModeGroup->addAction(m_actionImageViewPointerInteractionMode);
    m_imageViewInteractionModeGroup->addAction(m_actionImageViewPanInteractionMode);
    m_imageViewInteractionModeGroup->addAction(m_actionImageViewZoomInteractionMode);

    connect(m_actionImageViewPanInteractionMode, &QAction::triggered,
            [&](){m_imageWidget->setInteractionMode(ImageWidget::InteractionMode::Pan);});
    connect(m_actionImageViewZoomInteractionMode, &QAction::triggered,
            [&](){m_imageWidget->setInteractionMode(ImageWidget::InteractionMode::Zoom);});
    connect(m_actionImageViewPointerInteractionMode, &QAction::triggered,
            [&](){m_imageWidget->setInteractionMode(ImageWidget::InteractionMode::Pointer);});
}

void RisWidget::makeToolBars()
{
    m_imageViewToolBar = addToolBar("View");
    m_imageViewZoomCombo = new QComboBox(this);
    m_imageViewToolBar->addWidget(m_imageViewZoomCombo);
    m_imageViewZoomCombo->setEditable(true);
    m_imageViewZoomCombo->setInsertPolicy(QComboBox::NoInsert);
    m_imageViewZoomCombo->setDuplicatesEnabled(true);
    m_imageViewZoomCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    for(const GLfloat& z : ImageWidget::sm_zoomPresets)
    {
        m_imageViewZoomCombo->addItem(formatZoom(z * 100.0f) + '%');
    }
    m_imageViewZoomCombo->setCurrentIndex(ImageWidget::sm_defaultZoomPreset);
    m_imageViewZoomComboValidator = new QDoubleValidator(m_imageWidget->sm_zoomMinMax.first, m_imageWidget->sm_zoomMinMax.second, 4, m_imageViewZoomCombo);
    m_imageViewZoomComboValidator->setNotation(QDoubleValidator::StandardNotation);
    connect(m_imageViewZoomCombo, SIGNAL(activated(int)), this, SLOT(imageViewZoomComboChanged(int)));
    connect(m_imageViewZoomCombo->lineEdit(), SIGNAL(returnPressed()), this, SLOT(imageViewZoomComboCustomValueEntered()));
    connect(m_imageWidget, &ImageWidget::zoomChanged, this, &RisWidget::imageViewZoomChanged);
    m_imageViewToolBar->addAction(m_actionImageViewZoomToFit);
    m_imageViewToolBar->addSeparator();
    m_imageViewToolBar->addAction(m_actionImageViewPointerInteractionMode);
    m_imageViewToolBar->addAction(m_actionImageViewPanInteractionMode);
    m_imageViewToolBar->addAction(m_actionImageViewZoomInteractionMode);
    m_imageViewToolBar->addSeparator();
    m_imageViewToolBar->addAction(m_actionHighlightImagePixelUnderMouse);
    m_imageViewToolBar->addSeparator();
    m_imageViewToolBar->addAction(m_actionStoreImage);
}

void RisWidget::makeViews()
{
    m_imageWidget->makeView();
    m_histogramWidget->makeView();

    m_imageWidget->setClearColor(glm::vec4(1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 0.0f));
    m_histogramWidget->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));

    connect(m_imageWidget, &ImageWidget::pointerMovedToDifferentPixel, this, &RisWidget::imageViewPointerMovedToDifferentPixel);
    connect(m_histogramWidget, &HistogramWidget::gtpChanged, [&](){m_imageWidget->view()->update();});
}

void RisWidget::makeRenderer()
{
    m_rendererThread = new QThread(this);
    m_renderer.reset(new Renderer(m_imageWidget, m_histogramWidget));
    m_renderer->moveToThread(m_rendererThread);
    connect(m_rendererThread.data(), &QThread::started, m_renderer.get(), &Renderer::threadInitSlot, Qt::QueuedConnection);
    // Note: Connection must be direct or Renderer thread will terminate before finished signal is received
    connect(m_rendererThread.data(), &QThread::finished, m_renderer.get(), &Renderer::threadDeInitSlot, Qt::DirectConnection);
    m_rendererThread->start();
    connect(m_renderer.get(), &Renderer::newImageExtrema, m_histogramWidget, &HistogramWidget::newImageExtremaFoundByRenderer, Qt::QueuedConnection);
}


void RisWidget::updateStatusBarPixelInfoPresence()
{
    if(m_showStatusBarPixelInfo == m_statusBarPixelInfoWidget.isNull())
    {
        if(m_showStatusBarPixelInfo)
        {
            m_statusBarPixelInfoWidget = new QWidget;
            QHBoxLayout* layout = new QHBoxLayout;
            m_statusBarPixelInfoWidget->setLayout(layout);
            layout->addWidget(new QLabel(tr("X: ")));
            layout->addWidget(m_statusBarPixelInfoWidget_x = new QLabel("-"));
            layout->addWidget(new QLabel(tr("Y: ")));
            layout->addWidget(m_statusBarPixelInfoWidget_y = new QLabel("-"));
            layout->addWidget(new QLabel(tr("Intensity: ")));
            layout->addWidget(m_statusBarPixelInfoWidget_intensity = new QLabel("-"));
            statusBar()->addPermanentWidget(m_statusBarPixelInfoWidget.data());
        }
        else
        {
            statusBar()->removeWidget(m_statusBarPixelInfoWidget.data());
            m_statusBarFpsWidget->deleteLater();
        }
    }
}

void RisWidget::updateStatusBarFpsPresence()
{
    if(m_showStatusBarFps == m_statusBarFpsWidget.isNull())
    {
        if(m_showStatusBarFps)
        {
            m_statusBarFpsWidget = new QWidget;
            QHBoxLayout* layout = new QHBoxLayout;
            m_statusBarFpsWidget->setLayout(layout);
            layout->addWidget(new QLabel(tr("FPS: ")));
            layout->addWidget(m_statusBarFpsWidget_fps = new QLabel("-"));
            statusBar()->addPermanentWidget(m_statusBarFpsWidget.data());
        }
        else
        {
            statusBar()->removeWidget(m_statusBarFpsWidget.data());
            m_statusBarFpsWidget->deleteLater();
        }
    }
}

ImageWidget* RisWidget::imageWidget()
{
    return m_imageWidget;
}

HistogramWidget* RisWidget::histogramWidget()
{
    return m_histogramWidget;
}

QAction* RisWidget::getStoreImageAction() const
{
    return m_actionStoreImage;
}

void RisWidget::showCheckerPatternSlot()
{
    ShowCheckerDialog showCheckerDialog(this);
    if(showCheckerDialog.exec() == QDialog::Accepted)
    {
        showCheckerPattern(showCheckerDialog.checkerboardWidth(), showCheckerDialog.filter());
    }
}

void RisWidget::showCheckerPattern(int width, bool filterTexture)
{
    ImageData imageData;
    QSize imageSize;
    if(width < 0)
    {
        throw RisWidgetException("RisWidget::showCheckerPattern(int width, bool filterTexture): Negative value supplied for width.");
    }
    if(width <= 1)
    {
        imageSize.setWidth(1);
        imageSize.setHeight(1);
        imageData.push_back(width == 0 ? 0x0000 : 0xffff);
    }
    else
    {
        imageSize.setWidth(width);
        imageSize.setHeight(width);
        std::size_t pixelCount = static_cast<std::size_t>(width);
        pixelCount *= width;
        imageData.resize(pixelCount);

        std::size_t i = 0;
        bool a = true;
        float f = 65535.0f / (pixelCount - 1);
        GLushort *p = imageData.data();
        for(std::uint16_t r(0), c; r < width; ++r)
        {
            for(c = 0; c < width; ++c, ++p, ++i)
            {
                *p = a ? static_cast<GLushort>(std::nearbyint(f * i)) : 0x0000;
                a = !a;
            }
            a = !a;
        }
    }
    m_histogramWidget->updateImageLoaded(true);
    m_renderer->showImage(imageData, imageSize, filterTexture);
    m_imageWidget->updateImageSizeAndData(imageSize, imageData);
}

void RisWidget::risImageAcquired(PyObject* /*stream*/, PyObject* image)
{
    showImage(image);
}

void RisWidget::showImage(const GLushort* imageDataRaw, const QSize& imageSize, bool filterTexture)
{
    if(imageDataRaw == nullptr)
    {
        clearCanvasSlot();
    }
    else
    {
        std::size_t byteCount = sizeof(GLushort) *
                                static_cast<std::size_t>(imageSize.width()) *
                                static_cast<std::size_t>(imageSize.height());
        ImageData imageData(byteCount);
        memcpy(reinterpret_cast<void*>(imageData.data()),
               reinterpret_cast<const void*>(imageDataRaw),
               byteCount);
        m_renderer->showImage(imageData, imageSize, filterTexture);
        m_imageWidget->updateImageSizeAndData(imageSize, imageData);
        m_histogramWidget->updateImageLoaded(true);
    }

    std::chrono::steady_clock::time_point currentFrameTimestamp(std::chrono::steady_clock::now());
    if(m_showStatusBarFps && m_previousFrameTimestampValid)
    {
        std::chrono::duration<float> delta(std::chrono::duration_cast<std::chrono::duration<float>>(currentFrameTimestamp - m_previousFrameTimestamp));
        m_statusBarFpsWidget_fps->setText(QString::number(1.0f / delta.count()));
    }
    m_previousFrameTimestamp = currentFrameTimestamp;
    m_previousFrameTimestampValid = true;
}

void RisWidget::showImage(PyObject* image, bool filterTexture)
{
//  py::object imagepy{py::handle<>(py::borrowed(image))};
//  if(imagepy.is_none())
//  {
//      showImage(nullptr, QSize(), false);
//  }
//  else
//  {
//      np::ndarray imagenp{np::from_object(imagepy, np::dtype::get_builtin<GLushort>(), 2, 2, np::ndarray::CARRAY_RO)};
//      if(imagenp.is_none())
//      {
//          throw RisWidgetException("RisWidget::showImage(PyObject* image): image argument must be an "
//                                   "array-like object convertable to a 2d uint16 numpy array.");
//      }
//      const Py_intptr_t* shape = imagenp.get_shape();
//      showImage(reinterpret_cast<const GLushort*>(imagenp.get_data()), QSize(shape[1], shape[0]), filterTexture);
//  }
}

PyObject* RisWidget::getCurrentImage()
{
//     ImageData imageData;
//     QSize imageSize;
//     m_renderer->getImageDataAndSize(imageData, imageSize);
//     std::unique_ptr<np::ndarray> ret;
// 
//     if(!imageData.isEmpty())
//     {
//         Py_intptr_t imageSizePy = imageData.size();
//         ret.reset(new np::ndarray(np::empty(1, &imageSizePy, np::dtype::get_builtin<GLushort>())));
// //      std::cerr << "alloc\n";
//         std::copy(imageData.begin(), imageData.end(), reinterpret_cast<GLushort*>(ret->get_data()));
// //      memcpy(reinterpret_cast<void*>(ret->get_data()),
// //             reinterpret_cast<const void*>(imageData.data()),
// //             100/*sizeof(GLushort) * imageData.size()*/);
// //      std::cerr << "copy\n" << ret->get_nd() << "\t" << *ret->get_shape() << "\t" << imageSize.width() << "\t" << imageSize.height() << "\n";
//         /*try
//         {
//             ret->reshape(py::make_tuple(imageSize.height(), imageSize.width()));
//         }
//         catch(py::error_already_set const&)
//         {
//             PyErr_Print();
//         }
//         std::cerr << "rehape\n";*/
//     }
//     Py_INCREF(ret->ptr());
//     return ret->ptr();
    return nullptr;
}

PyObject* RisWidget::getHistogram()
{
//  auto histogramData = m_renderer->getHistogram();
//  std::unique_ptr<np::ndarray> ret;
// 
//  Py_intptr_t size = histogramData->ref().size();
//  if(size != 0)
//  {
//      ret.reset(new np::ndarray(np::empty(1, &size, np::dtype::get_builtin<GLuint>())));
//      memcpy(reinterpret_cast<void*>(ret->get_data()),
//             reinterpret_cast<const void*>(histogramData->ref().data()),
//             sizeof(GLuint) * histogramData->ref().size());
//  }
// 
//  // boost::python is not managing the copy of the pointer we are returning; we must incref so that it is not garbage
//  // collected when ret goes out of scope and its destructor decrefs its internal copy of the pointer.
//  Py_INCREF(ret->ptr());
//  return ret->ptr();
    return nullptr;
}

void RisWidget::setGtpEnabled(bool gtpEnabled)
{
    m_histogramWidget->setGtpEnabled(gtpEnabled);
}

void RisWidget::setGtpAutoMinMax(bool gtpAutoMinMax)
{
    m_histogramWidget->setGtpAutoMinMax(gtpAutoMinMax);
}

void RisWidget::setGtpMin(GLushort gtpMin)
{
    m_histogramWidget->setGtpMin(gtpMin);
}

void RisWidget::setGtpMax(GLushort gtpMax)
{
    m_histogramWidget->setGtpMax(gtpMax);
}

void RisWidget::setGtpGamma(GLfloat gtpGamma)
{
    m_histogramWidget->setGtpGamma(gtpGamma);
}

void RisWidget::setGtpGammaGamma(GLfloat gtpGammaGamma)
{
    m_histogramWidget->setGtpGammaGamma(gtpGammaGamma);
}

bool RisWidget::getGtpEnabled() const
{
    return m_histogramWidget->getGtpEnabled();
}

bool RisWidget::getGtpAutoMinMax() const
{
    return m_histogramWidget->getGtpAutoMinMax();
}

GLushort RisWidget::getGtpMin() const
{
    return m_histogramWidget->getGtpMin();
}

GLushort RisWidget::getGtpMax() const
{
    return m_histogramWidget->getGtpMax();
}

GLfloat RisWidget::getGtpGamma() const
{
    return m_histogramWidget->getGtpGamma();
}

GLfloat RisWidget::getGtpGammaGamma() const
{
    return m_histogramWidget->getGtpGammaGamma();
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
        GilLocker gilLock;
        std::string fnstdstr{fnqstr.toStdString()};
        PyObject* fnpystr = PyUnicode_FromString(fnstdstr.c_str());
        PyObject* image = PyObject_CallFunctionObjArgs(m_numpyLoadFunction, fnpystr, nullptr);
        if(image == nullptr)
        {
            PyObject *ptype(nullptr), *pvalue(nullptr), *ptraceback(nullptr);
            PyErr_Fetch(&ptype, &pvalue, &ptraceback);
            if(ptype != nullptr && pvalue != nullptr)
            {
                PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
                PyObject* e{PyObject_Str(pvalue)};
                QMessageBox::warning(this, "Failed to Open File", PyUnicode_AsUTF8(e));
                Py_DECREF(e);
            }
            else
            {
                QMessageBox::warning(this, "Failed to Open File", "(Failed to retrieve error information.)");
            }
            Py_XDECREF(ptype);
            Py_XDECREF(pvalue);
            Py_XDECREF(ptraceback);
        }
        else
        {
            showImage(image);
        }
        Py_XDECREF(fnpystr);
        Py_XDECREF(image);
    }
}

void RisWidget::clearCanvasSlot()
{
    m_renderer->showImage(ImageData(), QSize(), false);
    m_imageWidget->updateImageSizeAndData(QSize(), ImageData());
    m_histogramWidget->updateImageLoaded(false);
}

void RisWidget::imageViewPointerMovedToDifferentPixel(bool isOnPixel, QPoint pixelCoord, GLushort pixelValue)
{
    if(m_showStatusBarPixelInfo)
    {
        if(isOnPixel)
        {
            m_statusBarPixelInfoWidget_x->setText(QString::number(pixelCoord.x()));
            m_statusBarPixelInfoWidget_y->setText(QString::number(pixelCoord.y()));
            m_statusBarPixelInfoWidget_intensity->setText(QString::number(pixelValue));
        }
        else
        {
            m_statusBarPixelInfoWidget_x->setText("-");
            m_statusBarPixelInfoWidget_y->setText("-");
            m_statusBarPixelInfoWidget_intensity->setText("-");
        }
    }
}

void RisWidget::imageViewZoomComboCustomValueEntered()
{
    QString text(m_imageViewZoomCombo->lineEdit()->text());
    QString scaleText(text.left(text.indexOf("%")));

    bool valid;
    int zero(0);
    switch(m_imageViewZoomComboValidator->validate(scaleText, zero))
    {
    case QValidator::Acceptable:
        valid = true;
        break;
    case QValidator::Intermediate:
    case QValidator::Invalid:
    default:
        QMessageBox::information(this, "RisWidget", QString("Please enter a number between %1 and %2.").arg(formatZoom(ImageWidget::sm_zoomMinMax.first)).arg(formatZoom(ImageWidget::sm_zoomMinMax.second)));
        m_imageViewZoomCombo->setFocus();
        m_imageViewZoomCombo->lineEdit()->selectAll();
        valid = false;
        break;
    }

    if(valid)
    {
        bool converted;
        double scalePercent(scaleText.toDouble(&converted));
        if(!converted)
        {
            throw RisWidgetException(std::string("RisWidget::imageViewZoomComboCustomValueEntered(): scaleText.toDouble(..) failed for \"") +
                                     scaleText.toStdString() + "\".");
        }
        m_imageWidget->setCustomZoom(scalePercent * .01);
    }
}

void RisWidget::imageViewZoomComboChanged(int index)
{
    m_imageWidget->setZoomIndex(index);
}

void RisWidget::imageViewZoomChanged(int zoomIndex, GLfloat customZoom)
{
    if(zoomIndex != -1 && customZoom != 0)
    {
        throw RisWidgetException("RisWidget::imageViewZoomChanged(..): Both zoomIndex and customZoom specified.");
    }
    if(zoomIndex == -1 && customZoom == 0)
    {
        throw RisWidgetException("RisWidget::imageViewZoomChanged(..): Neither zoomIndex nor customZoom specified.");
    }

    if(zoomIndex != -1)
    {
        if(zoomIndex < 0 || zoomIndex >= m_imageViewZoomCombo->count())
        {
            throw RisWidgetException("RisWidget::imageViewZoomChanged(..): Invalid value for zoomIndex.");
        }
        m_imageViewZoomCombo->setCurrentIndex(zoomIndex);
    }
    else
    {
        m_imageViewZoomCombo->lineEdit()->setText(formatZoom(customZoom * 100.0f) + '%');
    }
}

void RisWidget::imageViewZoomToFitToggled(bool zoomToFit)
{
    m_imageViewZoomCombo->setEnabled(!zoomToFit);
    m_imageWidget->setZoomToFit(zoomToFit);
}

void RisWidget::highlightImagePixelUnderMouseToggled(bool highlight)
{
    m_imageWidget->setHighlightPointer(highlight);
}

void RisWidget::statusBarPixelInfoToggled(bool showStatusBarPixelInfo)
{
    if(showStatusBarPixelInfo != m_showStatusBarPixelInfo)
    {
        m_showStatusBarPixelInfo = showStatusBarPixelInfo;
        updateStatusBarPixelInfoPresence();
    }
}

void RisWidget::statusBarFpsToggled(bool showStatusBarFps)
{
    if(showStatusBarFps != m_showStatusBarFps)
    {
        m_showStatusBarFps = showStatusBarFps;
        updateStatusBarFpsPresence();
    }
}

// bool RisWidget::event(QEvent* event)
// {
//     bool ret{QMainWindow::event(event)};
//     if(event->type() == QEvent::PolishRequest)
//     {
//         emit polishRequestReceived;
//     }
//     return ret;
// }

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
    PyObject* mainModule;
    {
        GilLocker gilLock;
        mainModule = PyImport_AddModule("__main__");
    }
    if(mainModule == nullptr)
    {
        ret = -1;
        std::cerr << "int main(int argc, char** argv): PyImport_AddModule(\"__main__\") failed." << std::endl;
    }
    else
    {
        QApplication app(argc, argv);
        RisWidget* risWidget{new RisWidget("RisWidget Standalone")};
        risWidget->show();
        ret = app.exec();
    }
    Py_Finalize();
    return ret;
}

#endif

