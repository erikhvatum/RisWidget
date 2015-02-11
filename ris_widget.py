# The MIT License (MIT)
#
# Copyright (c) 2014 WUSTL ZPLAB
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
import pyagg
import sys

from . import image_widget
from . import histogram_widget
from .image import Image

class RisWidget(Qt.QMainWindow):
    # The image_changed signal is emitted when a new value is successfully assigned to the
    # RisWidget.image property
    image_changed = Qt.pyqtSignal()

    def __init__(self, window_title='RisWidget', parent=None, window_flags=Qt.Qt.WindowFlags(0)):
        super().__init__(parent, window_flags)
        if sys.platform == 'darwin': # workaround for https://bugreports.qt.io/browse/QTBUG-44230
            hs = Qt.QSlider(Qt.Qt.Horizontal)
            hs.show()
            hs.hide()
            hs.destroy()
            del hs
        if window_title is not None:
            self.setWindowTitle(window_title)
        self.setAcceptDrops(True)
        self._init_actions()
        self._init_toolbars()
        self._init_views()
        self._image = None

    def _init_actions(self):
        pass

    def _init_toolbars(self):
        pass

    def _init_views(self):
        qsurface_format = Qt.QSurfaceFormat()
        qsurface_format.setRenderableType(Qt.QSurfaceFormat.OpenGL)
        qsurface_format.setVersion(2, 1)
        qsurface_format.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
        qsurface_format.setSwapBehavior(Qt.QSurfaceFormat.DoubleBuffer)
        qsurface_format.setStereo(False)
        qsurface_format.setSwapInterval(1)
        self.image_widget_scroller = image_widget.ImageWidgetScroller(self, qsurface_format)
        self.image_widget = self.image_widget_scroller.image_widget
        self.setCentralWidget(self.image_widget_scroller)
        self._histogram_dock_widget = Qt.QDockWidget('Histogram', self)
        self.histogram_widget, self._histogram_container_widget = histogram_widget.HistogramWidget.make_histogram_and_container_widgets(self._histogram_dock_widget, qsurface_format)
        self.image_widget.histogram_widget = self.histogram_widget
        self._histogram_dock_widget.setWidget(self._histogram_container_widget)
        self._histogram_dock_widget.setAllowedAreas(Qt.Qt.BottomDockWidgetArea | Qt.Qt.TopDockWidgetArea)
        self._histogram_dock_widget.setFeatures(Qt.QDockWidget.DockWidgetFloatable | Qt.QDockWidget.DockWidgetMovable | Qt.QDockWidget.DockWidgetVerticalTitleBar)
        self.addDockWidget(Qt.Qt.BottomDockWidgetArea, self._histogram_dock_widget)
        self.histogram_widget.gamma_or_min_max_changed.connect(self.image_widget.update)
        self.histogram_widget.request_mouseover_info_status_text_change.connect(self.histogram_widget._on_request_mouseover_info_status_text_change)
        self.image_widget.request_mouseover_info_status_text_change.connect(self.histogram_widget._on_request_mouseover_info_status_text_change)

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
        self.image = Image(image_data)

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        if image is not None and not issubclass(type(image), Image):
            raise ValueError('The value assigned to the image property must either be derived from ris_widget.image.Image or must be None.  Did you mean to assign to the image_data property?')
        self.image_widget._on_image_changed(image)
        self.histogram_widget._on_image_changed(image)
        self._image = image
        self.image_changed.emit()

if __name__ == '__main__':
    import sys
    app = Qt.QApplication(sys.argv)
    rw = RisWidget()
    rw.show()
    app.exec_()
