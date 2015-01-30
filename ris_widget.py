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
        self._next_unnamed_image_idx = 0

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
#       self.image_widget_scroller = image_widget.ImageWidgetScroller(self, qsurface_format)
#       self.image_widget = self.image_widget_scroller.image_widget
#       self.setCentralWidget(self.image_widget_scroller)
        self.image_widget = image_widget.ImageWidget(self, qsurface_format)
        self.setCentralWidget(self.image_widget)
        self._histogram_dock_widget = Qt.QDockWidget('Histogram', self)
        self.histogram_widget, self._histogram_container_widget = histogram_widget.HistogramWidget.make_histogram_and_container_widgets(self._histogram_dock_widget, qsurface_format)
        self._histogram_dock_widget.setWidget(self._histogram_container_widget)
        self._histogram_dock_widget.setAllowedAreas(Qt.Qt.BottomDockWidgetArea | Qt.Qt.TopDockWidgetArea)
        self._histogram_dock_widget.setFeatures(Qt.QDockWidget.DockWidgetFloatable | Qt.QDockWidget.DockWidgetMovable | Qt.QDockWidget.DockWidgetVerticalTitleBar)
        self.addDockWidget(Qt.Qt.BottomDockWidgetArea, self._histogram_dock_widget)

    @property
    def image(self):
        """image property:
        None, or a 2D (grayscale), or a 3D (grayscale with alpha, rgb, or rgba) iterable of floats or ints may be
        assigned to or read from this property.  Assigning a 2D or 3D numpy array or buffer protocol object of the
        dtype uint8, uint16, or uint32, in C order, avoids an intermediate copy operation, keeping a reference to supplied
        data.  Assigning None clears the image and histogram views.

        The following dtypes are directly supported (data of any other type is converted to 32-bit floating point,
        and an exception is thrown if conversion fails):
        numpy.uint8
        numpy.uint16
        numpy.float32

        The following container layouts are supported (image data supplied in any other arrangement results in an exception):
        * A container of rows, each of which is a container of scalars.  This is displayed as a grayscale image, with
        each scalar representing a pixel.
        * A container of rows, each of which is a container of N scalars, with each scalar representing the intensity of a
        color channel and each container of N scalars representing a pixel.  For N of 2, image_data[:,:,0] represents grayscale
        intensity, while image_data[:,:,1] represents alpha intensity (transparency).  For N of 3, image_data[:,:,0] is red, image_data[:,:,1]
        is green, and image_data[:,:,2] is blue.  For N of 4, the situation is the same as for N of 3, with the addition of
        alpha intensity as image_data[:,:,3].

        A name may be associated with an image by assigning image data as described above in a tuple along with the desired
        image name.  For example, riswidget.image = (numpy.random.rand(600,800), 'this is random data')."""
        return self._image.data

    @image.setter
    def image(self, image_data):
        if len(image_data) == 2 and type(image_data[2]) == str:
            image = Image(image_data[0], image_data[1])
        else:
            image = Image(image_data, str(self._next_unnamed_image_idx))
            self._next_unnamed_image_idx += 1
        self.image_widget.image = image
        self.histogram_widget.image = image
        self._image = image

    @property
    def image_name(self):
        if self._image is not None:
            return self._image.name

if __name__ == '__main__':
    import sys
    app = Qt.QApplication(sys.argv)
    rw = RisWidget()
    rw.show()
    app.exec_()
