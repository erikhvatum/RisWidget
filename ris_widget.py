# The MIT License (MIT)
#
# Copyright (c) 2014-2015 WUSTL ZPLAB
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Authors: Erik Hvatum <ice.rikh@gmail.com>

from PyQt5 import Qt
import numpy
import sys

from .histogram_scene import HistogramScene
from .histogram_view import HistogramView
from .image import Image
from .image_scene import ImageScene
from .image_view import ImageView
from .shared_resources import FREEIMAGE
import ctypes

class RisWidget(Qt.QMainWindow):
    # The image_changed signal is emitted immediately after a new value is successfully assigned to the
    # RisWidget_instance.image property
    image_changed = Qt.pyqtSignal(object)

    def __init__(self, window_title='RisWidget', parent=None, window_flags=Qt.Qt.WindowFlags(0)):
        super().__init__(parent, window_flags)
#       if sys.platform == 'darwin': # workaround for https://bugreports.qt.io/browse/QTBUG-44230
#           hs = Qt.QSlider(Qt.Qt.Horizontal)
#           hs.show()
#           hs.hide()
#           hs.destroy()
#           del hs
        if window_title is not None:
            self.setWindowTitle(window_title)
        self.setAcceptDrops(True)
        self._init_scenes_and_views()
        self._init_actions()
        self._init_toolbars()
        self._image = None

    def _init_actions(self):
        self._histogram_view_reset_min_max_action = Qt.QAction(self)
        self._histogram_view_reset_min_max_action.setText('Reset Min/Max')
        self._histogram_view_reset_min_max_action.triggered.connect(self._on_reset_min_max)
        self._histogram_view_reset_gamma_action = Qt.QAction(self)
        self._histogram_view_reset_gamma_action.setText('Reset \u03b3')
        self._histogram_view_reset_gamma_action.triggered.connect(self._on_reset_gamma)

    @staticmethod
    def _format_zoom(zoom):
        if int(zoom) == zoom:
            return '{}'.format(int(zoom))
        else:
            txt = '{:.2f}'.format(zoom)
            if txt[-2:] == '00':
                return txt[:-3]
            if txt[-1:] == '0':
                return txt[:-1]
            return txt

    def _init_toolbars(self):
        self._image_view_toolbar = self.addToolBar('Image View')
        self._image_view_zoom_combo = Qt.QComboBox(self)
        self._image_view_toolbar.addWidget(self._image_view_zoom_combo)
        self._image_view_zoom_combo.setEditable(True)
        self._image_view_zoom_combo.setInsertPolicy(Qt.QComboBox.NoInsert)
        self._image_view_zoom_combo.setDuplicatesEnabled(True)
        self._image_view_zoom_combo.setSizeAdjustPolicy(Qt.QComboBox.AdjustToContents)
        for zoom in ImageView._ZOOM_PRESETS:
            self._image_view_zoom_combo.addItem(self._format_zoom(zoom * 100) + '%')
        self._image_view_zoom_combo.setCurrentIndex(ImageView._ZOOM_DEFAULT_PRESET_IDX)
        self._image_view_zoom_combo.activated[int].connect(self._image_view_zoom_combo_changed)
        self._image_view_zoom_combo.lineEdit().returnPressed.connect(self._image_view_zoom_combo_custom_value_entered)
        self.image_view.zoom_changed.connect(self._image_view_zoom_changed)
        self._image_view_toolbar.addAction(self.image_view.zoom_to_fit_action)
        self._histogram_view_toolbar = self.addToolBar('Histogram View')
        self._histogram_view_toolbar.addAction(self._histogram_dock_widget.toggleViewAction())
        self._histogram_view_toolbar.addAction(self._histogram_view_reset_min_max_action)
        self._histogram_view_toolbar.addAction(self._histogram_view_reset_gamma_action)
        self._histogram_view_toolbar.addAction(self.histogram_scene.auto_min_max_enabled_action)
#       self._image_name_toolbar = self.addToolBar('Image Name')
#       self._image_name_toolbar.addAction(self.image_view.show_image_name_action)

    def _init_scenes_and_views(self):
        self.image_scene = ImageScene(self)
        self.image_view = ImageView(self.image_scene, self)
        self.setCentralWidget(self.image_view)
        self.histogram_scene = HistogramScene(self)
        self.image_scene.histogram_scene = self.histogram_scene
        self._histogram_dock_widget = Qt.QDockWidget('Histogram', self)
        self.histogram_view, self._histogram_frame = HistogramView.make_histogram_view_and_frame(self.histogram_scene, self._histogram_dock_widget)
        self._histogram_dock_widget.setWidget(self._histogram_frame)
        self._histogram_dock_widget.setAllowedAreas(Qt.Qt.BottomDockWidgetArea | Qt.Qt.TopDockWidgetArea)
        self._histogram_dock_widget.setFeatures(Qt.QDockWidget.DockWidgetClosable | Qt.QDockWidget.DockWidgetFloatable | \
                                                Qt.QDockWidget.DockWidgetMovable | Qt.QDockWidget.DockWidgetVerticalTitleBar)
        self.addDockWidget(Qt.Qt.BottomDockWidgetArea, self._histogram_dock_widget)

    def dragEnterEvent(self, event):
        event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        mime_data = event.mimeData()
        if mime_data.hasImage():
            qimage = mime_data.imageData()
            if not qimage.isNull() and qimage.format() != Qt.QImage.Format_Invalid:
                if qimage.hasAlphaChannel():
                    desired_format = Qt.QImage.Format_RGBA8888
                    channel_count = 4
                else:
                    desired_format = Qt.QImage.Format_RGB888
                    channel_count = 3
                if qimage.format() != desired_format:
                    qimage = qimage.convertToFormat(desired_format)
                npyimage = numpy.ctypeslib.as_array(ctypes.cast(int(qimage.bits()), ctypes.POINTER(ctypes.c_uint8)),
                                                    shape=(qimage.height(), qimage.width(), channel_count))
                if qimage.isGrayscale():
                    npyimage=npyimage[...,0]
                image = Image(npyimage, mime_data.urls()[0] if mime_data.hasUrls() else None, shape_is_width_height=False)
                if image.data.ctypes.data == npyimage.ctypes.data:
                    # Retain reference to prevent deallocation of underlying buffer owned by Qt and wrapped by numpy.  This does happen,
                    # indicating that the various transponse operations just shift around elements of shape and strides rather than
                    # causing memcpys.
                    image.qimage = qimage
                self.image = image
                event.accept()
        elif mime_data.hasUrls():
            # Note: if the URL is a "file://..." representing a local file, toLocalFile returns a string
            # appropriate for feeding to Python's open() function.  If the URL does not refer to a local file,
            # toLocalFile returns None.
            fpaths = list(map(lambda url: url.toLocalFile(), mime_data.urls()))
            if len(fpaths) > 0 and fpaths[0].startswith('file:///.file/id=') and sys.platform == 'darwin':
                e = 'In order for image file drag & drop to work on OS X >=10.10 (Yosemite), please upgrade to at least Qt 5.4.1.'
                Qt.QMessageBox.information(self, 'Qt Upgrade Required', e)
                return
            freeimage = FREEIMAGE(show_messagebox_on_error=True, error_messagebox_owner=self)
            if freeimage is None:
                return
            if len(fpaths) == 1:
                self.image = Image(freeimage.read(fpaths[0]), fpaths[0])
                event.accept()
            else:
                # TODO: if more than one file is dropped, open them all in a new flipper
                for fpath in fpaths:
                    print(fpath)

    @property
    def image_data(self):
        """image_data property:
        The input assigned to this property may be None, in which case the current image and histogram views are cleared,
        and otherwise must be convertable to a 2D or 3D numpy array of shape (w, h) or (w, h, c), respectively*.  2D input
        is interpreted as grayscale.  3D input, depending on the value of c, is iterpreted as grayscale & alpha (c of 2),
        red & blue & green (c of 3), or red & blue & green & alpha (c of 4).

        The following dtypes are directly supported (data of any other type is converted to 32-bit floating point,
        and an exception is thrown if conversion fails):
        numpy.uint8
        numpy.uint16
        numpy.float32

        Supplying a numpy array of one of the above types as input may avoid an intermediate copy step by allowing RisWidget
        to keep a reference to the supplied array, allowing its data to be accessed directly.


        * IE, the iterable assigned to the image property is interpreted as an iterable of columns (image left to right), each
        containing an iterable of rows (image top to bottom), each of which is either a grayscale intensity value or an
        iterable of color channel intensity values (gray & alpha, or red & green & blue, or red & green & blue & alpha)."""
        return None if self._image is None else self._image.data

    @image_data.setter
    def image_data(self, image_data):
        self.image = None if image_data is None else Image(image_data)

    @property
    def image_data_T(self):
        """image_data_T property:
        The input assigned to this property may be None, in which case the current image and histogram views are cleared,
        and otherwise must be convertable to a 2D or 3D numpy array of shape (h, w) or (h, w, c), respectively*.  2D input
        is interpreted as grayscale.  3D input, depending on the value of c, is iterpreted as grayscale & alpha (c of 2),
        red & blue & green (c of 3), or red & blue & green & alpha (c of 4).

        The following dtypes are directly supported (data of any other type is converted to 32-bit floating point,
        and an exception is thrown if conversion fails):
        numpy.uint8
        numpy.uint16
        numpy.float32

        Supplying a numpy array of one of the above types as input may avoid an intermediate copy step by allowing RisWidget
        to keep a reference to the supplied array, allowing its data to be accessed directly.


        * IE, the iterable assigned to the image property is interpreted as an iterable of columns (image left to right), each
        containing an iterable of rows (image top to bottom), each of which is either a grayscale intensity value or an
        iterable of color channel intensity values (gray & alpha, or red & green & blue, or red & green & blue & alpha)."""
        if self._image is not None:
            return self._image.data_T

    @image_data_T.setter
    def image_data_T(self, image_data_T):
        self.image = None if image_data_T is None else Image(image_data_T, shape_is_width_height=False)

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        if image is not None and not issubclass(type(image), Image):
            raise ValueError('The value assigned to the image property must either be derived from ris_widget.image.Image or must be None.  Did you mean to assign to the image_data property?')
        self.histogram_scene.on_image_changing(image)
        self.image_scene.on_image_changing(image)
        self._image = image
        self.image_changed.emit(image)

    def _image_view_zoom_changed(self, zoom_preset_idx, custom_zoom):
        assert zoom_preset_idx == -1 and custom_zoom != 0 or zoom_preset_idx != -1 and custom_zoom == 0, 'zoom_preset_idx XOR custom_zoom must be set.'
        if zoom_preset_idx == -1:
            self._image_view_zoom_combo.lineEdit().setText(self._format_zoom(custom_zoom * 100) + '%')
        else:
            self._image_view_zoom_combo.setCurrentIndex(zoom_preset_idx)

    def _image_view_zoom_combo_changed(self, idx):
        self.image_view.zoom_preset_idx = idx

    def _image_view_zoom_combo_custom_value_entered(self):
        txt = self._image_view_zoom_combo.lineEdit().text()
        percent_pos = txt.find('%')
        scale_txt = txt if percent_pos == -1 else txt[:percent_pos]
        try:
            self.image_view.custom_zoom = float(scale_txt) * 0.01
        except ValueError:
            Qt.QMessageBox.information(self, self.windowTitle(), 'Please enter a number between {} and {}.'.format(self._format_zoom(ImageView._ZOOM_MIN_MAX[0] * 100),
                                                                                                                   self._format_zoom(ImageView._ZOOM_MIN_MAX[1] * 100)))
            self._image_view_zoom_combo.setFocus()
            self._image_view_zoom_combo.lineEdit().selectAll()

    def _on_reset_min_max(self):
        self.histogram_scene.auto_min_max_enabled = False
        del self.histogram_scene.min
        del self.histogram_scene.max

    def _on_reset_gamma(self):
        del self.histogram_scene.gamma

if __name__ == '__main__':
    import sys
    app = Qt.QApplication(sys.argv)
    rw = RisWidget()
    rw.show()
    app.exec_()
