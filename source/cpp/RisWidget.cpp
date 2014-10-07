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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL RisWidget_ARRAY_API
#include <numpy/arrayobject.h>

static void* do_import_array()
{
    // import_array() is actually a macro that returns NULL if it fails, so it has to be wrapped in order to be called
    // from a constructor which necessarily does not return anything
    import_array();
    return reinterpret_cast<void*>(1);
}

RisWidget::RisWidget(QString windowTitle_,
                     QWidget* parent,
                     Qt::WindowFlags flags)
  : QMainWindow(parent, flags),
    m_nextFlipperId(0),
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
        do_import_array();
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

    setAcceptDrops(true);

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
    m_imageViewToolBar->addAction(m_actionMakeFlipper);
}

void RisWidget::makeViews()
{
    m_imageWidget->makeView();
    m_histogramWidget->makeView();

    m_imageWidget->setClearColor(glm::vec4(1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 0.0f));
    m_histogramWidget->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));

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
    connect(m_renderer.get(), &Renderer::openClDeviceListChanged, this, &RisWidget::openClDeviceListChangedSlot, Qt::QueuedConnection);
    connect(m_renderer.get(), &Renderer::currentOpenClDeviceListIndexChanged, this, &RisWidget::currentOpenClDeviceListIndexChangedSlot, Qt::QueuedConnection);
    connect(m_renderer.get(), &Renderer::imageViewPointerMovedToDifferentPixel, this, &RisWidget::imageViewPointerMovedToDifferentPixel, Qt::QueuedConnection);
    connect(m_renderer.get(), &Renderer::histogramBinCountChanged, this, &RisWidget::histogramBinCountChanged, Qt::QueuedConnection);
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
    PyArrayObject* imageao = reinterpret_cast<PyArrayObject*>(PyArray_FromAny(image, PyArray_DescrFromType(NPY_USHORT),
                                                                              2, 2, NPY_ARRAY_CARRAY_RO, nullptr));
    if(imageao == nullptr)
    {
        throw RisWidgetException("RisWidget::showImage(PyObject* image): image argument must be an "
                                 "array-like object convertable to a 2d uint16 numpy array.");
    }
    npy_intp* shape = PyArray_DIMS(imageao);
    showImage(reinterpret_cast<const GLushort*>(PyArray_DATA(imageao)), QSize(shape[1], shape[0]), filterTexture);
    Py_DECREF(imageao);
}

void RisWidget::showImageFromNpyFile(const std::string& npyFileName)
{
    GilLocker gilLock;
    PyObject* fnpystr = PyUnicode_FromString(npyFileName.c_str());
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

PyObject* RisWidget::getCurrentImage()
{
    GilLocker gilLocker;
    ImageData imageData;
    QSize imageSize;
    m_renderer->getImageDataAndSize(imageData, imageSize);
    PyObject* ret;

    if(imageData.isEmpty())
    {
        ret = Py_None;
        Py_XINCREF(ret);
    }
    else
    {
        npy_intp shape[] = {imageSize.height(), imageSize.width()};
        ret = PyArray_EMPTY(2, shape, NPY_USHORT, false);
        // Note that if ret is null, it is because PyArray_EMPTY failed and a Python exception was thrown.  We don't
        // clear the exception, so it will propagate back to the Python code that called this function.
        if(ret != nullptr)
        {
            PyArrayObject* retnp{reinterpret_cast<PyArrayObject*>(ret)};
            npy_intp ystride{PyArray_STRIDE(retnp, 0)}, xstride{PyArray_STRIDE(retnp, 1)};
            npy_uintp retdataddr{reinterpret_cast<npy_uintp>(PyArray_DATA(retnp))}, retdatColaddr;
            const GLushort* srcdat{imageData.data()};
            npy_intp y{0}, x;
            for(;;)
            {
                x = 0;
                retdatColaddr = retdataddr;
                for(;;)
                {
                    *reinterpret_cast<GLushort*>(retdatColaddr) = *srcdat;
                    ++srcdat;
                    ++x;
                    if(x == shape[1]) break;
                    retdatColaddr += xstride;
                }
                ++y;
                if(y == shape[0]) break;
                retdataddr += ystride;
            }
        }
    }
    return ret;
}

PyObject* RisWidget::getHistogram()
{
    GilLocker gilLocker;
    std::shared_ptr<LockedRef<const HistogramData>> histogramData(m_renderer->getHistogram());
    PyObject* ret;
    npy_intp size{static_cast<npy_intp>(histogramData->ref().size())};

    if(size == 0)
    {
        ret = Py_None;
        Py_XINCREF(ret);
    }
    else
    {
        ret = PyArray_EMPTY(1, &size, NPY_UINT, false);
        if(ret != nullptr)
        {
            PyArrayObject* retnp{reinterpret_cast<PyArrayObject*>(ret)};
            npy_intp stride{PyArray_STRIDE(retnp, 0)};
            npy_uintp retdataddr{reinterpret_cast<npy_uintp>(PyArray_DATA(retnp))};
            const GLuint* srcdat{histogramData->ref().data()};
            const GLuint* srcdatEnd{srcdat + size};
            for(;;)
            {
                *reinterpret_cast<GLuint*>(retdataddr) = *srcdat;
                ++srcdat;
                if(srcdat == srcdatEnd) break;
                retdataddr += stride;
            }
        }
    }

    return ret;
}

GLuint RisWidget::getHistogramBinCount() const
{
    return m_renderer->getHistogramBinCount();
}

void RisWidget::setHistogramBinCount(GLuint histogramBinCount)
{
    m_renderer->setHistogramBinCount(histogramBinCount);
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
        showImageFromNpyFile(fnqstr.toStdString());
    }
}

void RisWidget::clearCanvasSlot()
{
    m_renderer->showImage(ImageData(), QSize(), false);
    m_imageWidget->updateImageSizeAndData(QSize(), ImageData());
    m_histogramWidget->updateImageLoaded(false);
}

bool RisWidget::hasFlipper(const QString& flipperName) const
{
    return m_flippers.find(flipperName) != m_flippers.end();
}

Flipper* RisWidget::getFlipper(const QString& flipperName)
{
    auto flipIt = m_flippers.find(flipperName);
    if(flipIt == m_flippers.end())
    {
        std::ostringstream o;
        o << "RisWidget::getFlipper(const QString& flipperName): Failed to find flipper with the name specified (\"";
        o << flipperName.toStdString() << "\").";
        throw RisWidgetException(o.str());
    }
    return flipIt->second;
}

QVector<QString> RisWidget::getFlipperNames() const
{
    QVector<QString> flipperNames;
    for(std::map<QString, Flipper*>::const_iterator flipIt{m_flippers.begin()}; flipIt != m_flippers.end(); ++flipIt)
    {
        flipperNames.append(flipIt->first);
    }
    return flipperNames;
}

Flipper* RisWidget::makeFlipper()
{
    uint64_t flipperId;
    // It is possible, if somewhat pathological, for the user to specify integer names that would conflict with the
    // default naming scheme.  The following loop skips to the first non-conflicting integer in this case.
    do
    {
        flipperId = m_nextFlipperId++;
    } while(m_flippers.find(QString::number(flipperId)) != m_flippers.end());

    QDockWidget* dw{new QDockWidget(QString("Flipbook (%1)").arg(flipperId), this)};
    dw->setAttribute(Qt::WA_DeleteOnClose, true);
    dw->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    Flipper* flipper{new Flipper(dw, this, QString::number(flipperId))};
    m_flippers[QString::number(flipperId)] = flipper;
    connect(flipper, &Flipper::flipperNameChanged, this, &RisWidget::flipperNameChanged);
    connect(flipper, &Flipper::closing, this, &RisWidget::flipperClosing);
    dw->setWidget(flipper);
    addDockWidget(Qt::RightDockWidgetArea, dw);
    return flipper;
}

void RisWidget::flipperNameChanged(Flipper* flipper, QString oldName)
{
    QString newName{flipper->getFlipperName()};
#ifdef DEBUG
    auto flipIt = m_flippers.find(oldName);
    if(flipIt == m_flippers.end())
    {
        throw RisWidgetException("RisWidget::flipperNameChanged(Flipper* flipper, QString oldName): Failed to find flipper with name "
                                 "specified by oldName argument.");
    }
    flipIt = m_flippers.find(newName);
    if(flipIt != m_flippers.end())
    {
        throw RisWidgetException("RisWidget::flipperNameChanged(Flipper* flipper, QString oldName): A Flipper with the new name already "
                                 "exists.");
    }
    if(flipIt->second != flipper)
    {
        throw RisWidgetException("RisWidget::flipperNameChanged(Flipper* flipper, QString oldName): m_flippers[oldName] != flipper.");
    }
#endif
    m_flippers.erase(oldName);
    m_flippers[newName] = flipper;
}

void RisWidget::flipperClosing(Flipper* flipper)
{
    QString flipperName{flipper->getFlipperName()};
    auto flipIt = m_flippers.find(flipperName);
    if(flipIt == m_flippers.end())
    {
        throw RisWidgetException("RisWidget::flipperDestroyed(QObject* flipperQObj): m_flippers does not seem to contain the Flipper "
                                 "being destroyed.");
    }
    m_flippers.erase(flipIt);
    disconnect(flipper, &Flipper::flipperNameChanged, this, &RisWidget::flipperNameChanged);
    disconnect(flipper, &Flipper::closing, this, &RisWidget::flipperClosing);
    // NB: Because Flipper's Qt::WA_DeleteOnClose attribute is set to true, flipper will be automatically deallocated,
    // and we must not delete flipper manually
    std::cerr << "flipper \"" << flipperName.toStdString() << "\" destroyed.\n";
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

void RisWidget::refreshOpenClDeviceList()
{
}

QVector<QString> RisWidget::getOpenClDeviceList() const
{
    return m_renderer->getOpenClDeviceList();
}

int RisWidget::getCurrentOpenClDeviceListIndex() const
{
    return m_renderer->getCurrentOpenClDeviceListIndex();
}

void RisWidget::setCurrentOpenClDeviceListIndex(int newOpenClDeviceListIndex)
{
    m_renderer->setCurrentOpenClDeviceListIndex(newOpenClDeviceListIndex);
}

void RisWidget::openClDeviceListChangedSlot(QVector<QString> openClDeviceList)
{
    if(!m_openClDevicesGroup.isNull())
    {
        m_menuOpenClDevices->clear();
        for(QAction* action : m_actionsOpenClDevices)
        {
            disconnect(action, SIGNAL(triggered()), nullptr, nullptr);
            action->deleteLater();
        }
        m_actionsOpenClDevices.clear();
        m_openClDevicesGroup->deleteLater();
    }
    m_openClDevicesGroup.reset(new QActionGroup(this));
    m_actionsOpenClDevices.reserve(openClDeviceList.size());
    int i = 0;
    for(const QString& description : openClDeviceList)
    {
        QAction* action{m_menuOpenClDevices->addAction(description)};
        action->setCheckable(true);
        action->setChecked(false);
        connect(action, &QAction::triggered, [&, i](){setCurrentOpenClDeviceListIndex(i);});
        m_openClDevicesGroup->addAction(action);
        m_actionsOpenClDevices.append(action);
        ++i;
    }
    emit openClDeviceListChanged(openClDeviceList);
}

void RisWidget::currentOpenClDeviceListIndexChangedSlot(int currentOpenClDeviceListIndex)
{
    if(currentOpenClDeviceListIndex < 0 || currentOpenClDeviceListIndex >= m_actionsOpenClDevices.size())
    {
        std::ostringstream o;
        o << "RisWidget::currentOpenClDeviceListIndexChangedSlot(int currentOpenClDeviceListIndex): Value for ";
        o << "currentOpenClDeviceListIndex must be in the range [0, " << m_actionsOpenClDevices.size() << "), not ";
        o << currentOpenClDeviceListIndex << ".  Note that the right end of this is a function of the number of OpenCL ";
        o << "devices available.";
        throw RisWidgetException(o.str());
    }
    m_actionsOpenClDevices[currentOpenClDeviceListIndex]->setChecked(true);
    emit currentOpenClDeviceListIndexChanged(currentOpenClDeviceListIndex);
}

void RisWidget::dragEnterEvent(QDragEnterEvent* event)
{
    event->acceptProposedAction();
}

void RisWidget::dragMoveEvent(QDragMoveEvent* event)
{
    event->acceptProposedAction();
}

void RisWidget::dragLeaveEvent(QDragLeaveEvent* event)
{
    event->accept();
}

void RisWidget::dropEvent(QDropEvent* event)
{
    const QMimeData* md{event->mimeData()};
    bool accept{false};

    if(md->hasImage())
    {
        // Raw image data is preferred in the case where both image data and source URL are present.  This is the case,
        // for example, on OS X when an image is dragged from Firefox.
        accept = true;
        QImage rgbImage(md->imageData().value<QImage>().convertToFormat(QImage::Format_RGB888));
        std::vector<GLushort> gsImage(rgbImage.width() * rgbImage.height(), 0);
        const GLubyte* rgbIt{rgbImage.bits()};
        const GLubyte* rgbItE{rgbIt + rgbImage.width() * rgbImage.height() * 3};
        for(GLushort* gsIt{gsImage.data()}; rgbIt != rgbItE; ++gsIt, rgbIt += 3)
        {
            *gsIt = GLushort(256) * static_cast<GLushort>(0.2126f * rgbIt[0] + 0.7152f * rgbIt[1] + 0.0722f * rgbIt[2]);
        }
        showImage(gsImage.data(), rgbImage.size());
    }
    else if(md->hasUrls())
    {
        QUrl url(md->urls()[0]);
        if(url.isLocalFile())
        {
            QString fn(url.toLocalFile());
            if(fn.endsWith(".npy", Qt::CaseInsensitive))
            {
                accept = true;
                showImageFromNpyFile(fn.toStdString());
            }
            else
            {
                fipImage image;
                std::string fnstdstr(fn.toStdString());
                if(image.load(fnstdstr.c_str()) && image.convertToUINT16())
                {
                    accept = true;
                    showImage((GLushort*)image.accessPixels(), QSize(image.getWidth(), image.getHeight()));
                }
            }
        }
    }

    if(accept)
    {
        event->acceptProposedAction();
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

