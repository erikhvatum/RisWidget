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

import ctypes
from PyQt5 import Qt
import numpy
import sys
from .flipbook import Flipbook
from .histogram_scene import HistogramScene, HistogramItem
from .histogram_view import HistogramView
from .display_image import DisplayImage
from .image_stack import ImageStack
from .main_scene import MainScene
from .main_view import MainView
from .shader_scene import ContextualInfoItem
from .shared_resources import FREEIMAGE, GL_QSURFACE_FORMAT

class RisWidget(Qt.QMainWindow):
    def __init__(self, window_title='RisWidget', parent=None, window_flags=Qt.Qt.WindowFlags(0), msaa_sample_count=2,
                 DisplayImageClass=DisplayImage, ImageStackClass=ImageStack, MainSceneClass=MainScene, MainViewClass=MainView,
                 MainViewContextualInfoItemClass=ContextualInfoItem,
                 HistogramItemClass=HistogramItem, HistogramSceneClass=HistogramScene, HistogramViewClass=HistogramView,
                 HistgramViewContextualInfoItemClass=ContextualInfoItem):
        super().__init__(parent, window_flags)
        GL_QSURFACE_FORMAT(msaa_sample_count)
        if window_title is not None:
            self.setWindowTitle(window_title)
        self.setAcceptDrops(True)
        self._init_scenes_and_views(
            DisplayImageClass, ImageStackClass, MainSceneClass, MainViewClass,
            MainViewContextualInfoItemClass,
            HistogramItemClass, HistogramSceneClass, HistogramViewClass,
            HistgramViewContextualInfoItemClass)
        self._init_actions()
        self._init_toolbars()
        self._init_menus()
        # Flipbook names -> Flipbook widget instances
        self._flipbooks = dict()

    def _init_actions(self):
        self._main_view_reset_min_max_action = Qt.QAction(self)
        self._main_view_reset_min_max_action.setText('Reset Min/Max')
#       self._main_view_reset_min_max_action.triggered.connect(self._on_reset_min_max)
        self._main_view_reset_gamma_action = Qt.QAction(self)
        self._main_view_reset_gamma_action.setText('Reset \u03b3')
        self._main_view_reset_gamma_action.triggered.connect(self._on_reset_gamma)
        if sys.platform == 'darwin':
            self.exit_fullscreen_action = Qt.QAction(self)
            # If self.exit_fullscreen_action's text were "Exit Full Screen Mode" as we desire,
            # we would not be able to add it as a menu entry (http://doc.qt.io/qt-5/qmenubar.html#qmenubar-on-os-x).
            # "Leave Full Screen Mode" is a compromise.
            self.exit_fullscreen_action.setText('Leave Full Screen Mode')
            self.exit_fullscreen_action.triggered.connect(self.showNormal)
            self.exit_fullscreen_action.setShortcut(Qt.Qt.Key_Escape)
            self.exit_fullscreen_action.setShortcutContext(Qt.Qt.ApplicationShortcut)

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
        self._main_view_toolbar = self.addToolBar('Image View')
        self._main_view_zoom_combo = Qt.QComboBox(self)
        self._main_view_toolbar.addWidget(self._main_view_zoom_combo)
        self._main_view_zoom_combo.setEditable(True)
        self._main_view_zoom_combo.setInsertPolicy(Qt.QComboBox.NoInsert)
        self._main_view_zoom_combo.setDuplicatesEnabled(True)
        self._main_view_zoom_combo.setSizeAdjustPolicy(Qt.QComboBox.AdjustToContents)
        for zoom in MainView._ZOOM_PRESETS:
            self._main_view_zoom_combo.addItem(self._format_zoom(zoom * 100) + '%')
        self._main_view_zoom_combo.setCurrentIndex(MainView._ZOOM_ONE_TO_ONE_PRESET_IDX)
        self._main_view_zoom_combo.activated[int].connect(self._main_view_zoom_combo_changed)
        self._main_view_zoom_combo.lineEdit().returnPressed.connect(self._main_view_zoom_combo_custom_value_entered)
        self.main_view.zoom_changed.connect(self._main_view_zoom_changed)
        self._main_view_toolbar.addAction(self.main_view.zoom_to_fit_action)
        self._main_view_toolbar.addAction(self._main_view_reset_min_max_action)
        self._main_view_toolbar.addAction(self._main_view_reset_gamma_action)
#       self._main_view_toolbar.addAction(self.main_scene.image_item.auto_min_max_enabled_action)
        self._histogram_view_toolbar = self.addToolBar('Histogram View')
        self._histogram_view_toolbar.addAction(self._histogram_dock_widget.toggleViewAction())
#       self._image_name_toolbar = self.addToolBar('Image Name')
#       self._image_name_toolbar.addAction(self.main_view.show_image_name_action)

    def _init_menus(self):
        mb = self.menuBar()
        m = mb.addMenu('View')
        if sys.platform == 'darwin':
            m.addAction(self.exit_fullscreen_action)
            m.addSeparator()
        m.addAction(self.main_view.zoom_to_fit_action)
        m.addAction(self.main_view.zoom_one_to_one_action)

    def _init_scenes_and_views(self, DisplayImageClass, ImageStackClass, MainSceneClass, MainViewClass, MainViewContextualInfoItemClass,
                               HistogramItemClass, HistogramSceneClass, HistogramViewClass, HistgramViewContextualInfoItemClass):
        self.main_scene = MainSceneClass(self, DisplayImageClass, ImageStackClass, MainViewContextualInfoItemClass)
        self.main_view = MainViewClass(self.main_scene, self)
        self.setCentralWidget(self.main_view)
        self.histogram_scene = HistogramSceneClass(self, self.main_scene.image_stack, HistogramItemClass, HistgramViewContextualInfoItemClass)
        self._histogram_dock_widget = Qt.QDockWidget('Histogram', self)
        self.histogram_view, self._histogram_frame = HistogramViewClass.make_histogram_view_and_frame(self.histogram_scene, self._histogram_dock_widget)
        self._histogram_dock_widget.setWidget(self._histogram_frame)
        self._histogram_dock_widget.setAllowedAreas(Qt.Qt.BottomDockWidgetArea | Qt.Qt.TopDockWidgetArea)
        self._histogram_dock_widget.setFeatures(
            Qt.QDockWidget.DockWidgetClosable | Qt.QDockWidget.DockWidgetFloatable | \
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
                if channel_count == 3:
                    # 24-bit RGB QImage rows are padded to 32-bit chunks, which we must match
                    row_stride = qimage.width() * 3
                    row_stride += 4 - (row_stride % 4)
                    padded = numpy.ctypeslib.as_array(ctypes.cast(int(qimage.bits()), ctypes.POINTER(ctypes.c_uint8)), shape=(qimage.height(), row_stride))
                    padded = padded[:, qimage.width() * 3].reshape((qimage.height(), qimage.width(), 3))
                    npyimage = numpy.empty((qimage.height(), qimage.width(), 3), dtype=numpy.uint8)
                    npyimage.flat = padded.flat
                else:
                    npyimage = numpy.ctypeslib.as_array(
                        ctypes.cast(int(qimage.bits()), ctypes.POINTER(ctypes.c_uint8)),
                        shape=(qimage.height(), qimage.width(), channel_count))
                if qimage.isGrayscale():
                    npyimage=npyimage[...,0]
                self.main_scene.image_stack.replace_image_data(0, npyimage, keep_name=False, shape_is_width_height=False, name=str(mime_data.urls()[0]) if mime_data.hasUrls() else None)
                image_object = self.main_scene.image_stack.image_objects[0]
                if image_object.data.ctypes.data == npyimage.ctypes.data:
                    def del_qimage():
                        try:
                            del image_object.qimage
                            print('del image_object.qimage')
                        except AttributeError:
                            pass
                        try:
                            image_object.data_changed.disconnect(del_qimage)
                        except TypeError:
                            pass
                    image_object.data_changed.connect(del_qimage)
                    # Retain reference to prevent deallocation of underlying buffer owned by Qt and wrapped by numpy.  This does happen,
                    # indicating that the various transponse operations just shift around elements of shape and strides rather than
                    # causing memcpys.
                    image_object.qimage = qimage
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
                image_data = freeimage.read(fpaths[0])
                self.main_scene.image_stack.replace_image_data(0, image_data, keep_name=False, name=fpaths[0])
            else:
                # TODO: read images in background thread and display modal progress bar dialog with cancel button
                images = [DisplayImage(freeimage.read(fpath), name=fpath) for fpath in fpaths]
                self.make_flipbook(images)
            event.accept()

    @property
    def image_object(self):
        image_objects = self.main_scene.image_stack.image_objects
        return image_objects[0] if image_objects else None

    @image_object.setter
    def image_object(self, image_object):
        self.main_scene.image_stack.replace_image_object(0, image_object)

    @property
    def image_data(self):
        image_objects = self.main_scene.image_stack.image_objects
        if image_objects:
            return image_objects[0].data

    @image_data.setter
    def image_data(self, image_data):
        self.main_scene.image_stack.replace_image_data(0, image_data)

    @property
    def image_data_T(self):
        image_objects = self.main_scene.image_stack.image_objects
        if image_objects:
            return image_objects[0].data_T

    @image_data.setter
    def image_data_T(self, image_data_T):
        self.main_scene.image_stack.replace_image_data(0, image_data_T, shape_is_width_height=False)

    def make_flipbook(self, images=None, name=None):
        """The images argument may be any mixture of ris_widget.image.DisplayImage objects and raw data iterables of the sort that
        may be assigned to RisWidget.image_data or RisWidget.image_data_T.
        If None is supplied for images, an empty flipbook is created.
        If None is supplied for name, a unique name is generated.
        If the value supplied for name is not unique, a suffix is appended such that the resulting name is unique."""
        flipbook = Flipbook(self._uniqueify_flipbook_name, lambda image_object: RisWidget.image_object.fset(self, image_object), images, name)
        assert flipbook.name not in self._flipbooks
        self._flipbooks[flipbook.name] = flipbook
        flipbook.name_changed.connect(self._on_flipbook_name_changed)
        flipbook.destroyed.connect(self._on_flipbook_destroyed)
        dock_widget = Qt.QDockWidget(flipbook.name, self)
        dock_widget.setWidget(flipbook)
        dock_widget.setAllowedAreas(Qt.Qt.LeftDockWidgetArea | Qt.Qt.RightDockWidgetArea)
        dock_widget.setFeatures(Qt.QDockWidget.DockWidgetClosable | Qt.QDockWidget.DockWidgetFloatable | Qt.QDockWidget.DockWidgetMovable)
        dock_widget.setAttribute(Qt.Qt.WA_DeleteOnClose)
        self.addDockWidget(Qt.Qt.RightDockWidgetArea, dock_widget)

    def get_flipbook(self, name):
        return self._flipbooks[name]

    def close_flipbook(self, name):
        self._flipbooks[name].parent().deleteLater()

    def _main_view_zoom_changed(self, zoom_preset_idx, custom_zoom):
        assert zoom_preset_idx == -1 and custom_zoom != 0 or zoom_preset_idx != -1 and custom_zoom == 0, \
               'zoom_preset_idx XOR custom_zoom must be set.'
        if zoom_preset_idx == -1:
            self._main_view_zoom_combo.lineEdit().setText(self._format_zoom(custom_zoom * 100) + '%')
        else:
            self._main_view_zoom_combo.setCurrentIndex(zoom_preset_idx)

    def _main_view_zoom_combo_changed(self, idx):
        self.main_view.zoom_preset_idx = idx

    def _main_view_zoom_combo_custom_value_entered(self):
        txt = self._main_view_zoom_combo.lineEdit().text()
        percent_pos = txt.find('%')
        scale_txt = txt if percent_pos == -1 else txt[:percent_pos]
        try:
            self.main_view.custom_zoom = float(scale_txt) * 0.01
        except ValueError:
            e = 'Please enter a number between {} and {}.'.format(
                self._format_zoom(MainView._ZOOM_MIN_MAX[0] * 100),
                self._format_zoom(MainView._ZOOM_MIN_MAX[1] * 100))
            Qt.QMessageBox.information(self, 'self.windowTitle() Input Error', e)
            self._main_view_zoom_combo.setFocus()
            self._main_view_zoom_combo.lineEdit().selectAll()

#   def _on_reset_min_max(self):
#       self.main_scene.image_item.auto_min_max_enabled = False
#       del self.main_scene.image_item.min
#       del self.main_scene.image_item.max

    def _on_reset_gamma(self):
        del self.main_scene.image_item.gamma

    def _uniqueify_flipbook_name(self, name):
        if name not in self._flipbooks:
            return name
        dupe_count = 1
        try_name = name + str(dupe_count)
        while try_name in self._flipbooks:
            # This loop is not fast for large numbers of flipbooks, of which there should not be
            dupe_count += 1
            try_name = name + str(dupe_count)
        return try_name

    def _on_flipbook_name_changed(self, flipbook, old_name, name):
        assert name not in self._flipbooks
        del self._flipbooks[old_name]
        self._flipbooks[name] = flipbook

    def _on_flipbook_destroyed(self, flipbook_qobject):
        del self._flipbooks[flipbook_qobject.objectName()]

if __name__ == '__main__':
    import sys
    app = Qt.QApplication(sys.argv)
    rw = RisWidget()
    rw.show()
    app.exec_()
