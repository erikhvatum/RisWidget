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
from . import om
from .image import Image
from .layer import Layer
from .qwidgets.flipbook import Flipbook
from .qwidgets.layer_stack_table import InvertingProxyModel, LayerStackTableModel, LayerStackTableView
from .qwidgets import progress_thread_pool
from .qgraphicsitems.contextual_info_item import ContextualInfoItem
from .qgraphicsitems.histogram_items import HistogramItem
from .qgraphicsitems.layer_stack_item import LayerStackItem
from .qgraphicsscenes.general_scene import GeneralScene
from .qgraphicsviews.general_view import GeneralView
from .qgraphicsscenes.histogram_scene import HistogramScene
from .qgraphicsviews.histogram_view import HistogramView
from .shared_resources import FREEIMAGE, GL_QSURFACE_FORMAT, NV_PATH_RENDERING_AVAILABLE

def _atexit():
    #TODO: find a better way to do this or a way to avoid the need
    try:
        from IPython import Application
        Application.instance().shell.del_var('rw')
    except:
        pass
    import gc
    gc.collect()

if sys.platform == 'darwin':
    class NonTransientScrollbarsStyle(Qt.QProxyStyle):
        def styleHint(self, sh, option=None, widget=None, returnData=None):
            if sh == Qt.QStyle.SH_ScrollBar_Transient:
                return 0
            return self.baseStyle().styleHint(sh, option, widget, returnData)

class RisWidget(Qt.QMainWindow):
    def __init__(self, window_title='RisWidget', parent=None, window_flags=Qt.Qt.WindowFlags(0), msaa_sample_count=2,
                 layer_stack = tuple(),
                 LayerStackItemClass=LayerStackItem, GeneralSceneClass=GeneralScene, GeneralViewClass=GeneralView,
                 GeneralViewContextualInfoItemClass=None,
                 HistogramItemClass=HistogramItem, HistogramSceneClass=HistogramScene, HistogramViewClass=HistogramView,
                 HistgramViewContextualInfoItemClass=None,
                 FlipbookClass=Flipbook):
        """A None value for GeneralViewContextualInfoItemClass or HistgramViewContextualInfoItemClass represents 
        ContextualInfoItemNV if the GL_NV_path_rendering extension is available and ContextualInfoItem otherwise."""
        super().__init__(parent, window_flags)
        # TODO: look deeper into opengl buffer swapping order and such to see if we can become compatible with OS X auto-hiding scrollbars
        # rather than needing to disable them
        if sys.platform == 'darwin':
            style = Qt.QApplication.style()
            if style.styleHint(Qt.QStyle.SH_ScrollBar_Transient) != 0:
                Qt.QApplication.setStyle(NonTransientScrollbarsStyle(style))
        GL_QSURFACE_FORMAT(msaa_sample_count)
        if window_title is not None:
            self.setWindowTitle(window_title)
        self.setAcceptDrops(True)
        if GeneralViewContextualInfoItemClass is None or HistgramViewContextualInfoItemClass is None:
            if NV_PATH_RENDERING_AVAILABLE():
                from .qgraphicsitems.contextual_info_item_nv import ContextualInfoItemNV
            if GeneralViewContextualInfoItemClass is None:
                GeneralViewContextualInfoItemClass = ContextualInfoItemNV if NV_PATH_RENDERING_AVAILABLE() else ContextualInfoItem
            if HistgramViewContextualInfoItemClass is None:
                HistgramViewContextualInfoItemClass = ContextualInfoItemNV if NV_PATH_RENDERING_AVAILABLE() else ContextualInfoItem
        self.FlipbookClass = FlipbookClass
        self._init_scenes_and_views(
            LayerStackItemClass, GeneralSceneClass, GeneralViewClass,
            GeneralViewContextualInfoItemClass,
            HistogramItemClass, HistogramSceneClass, HistogramViewClass,
            HistgramViewContextualInfoItemClass)
        self._layer_stack = None
        self.layer_stack = layer_stack
        self._init_main_flipbook()
        self._init_actions()
        self._init_toolbars()
        self._init_menus()
        import atexit
        atexit.register(_atexit)

    def _init_actions(self):
        self.layer_stack_reset_curr_min_max = Qt.QAction(self)
        self.layer_stack_reset_curr_min_max.setText('Reset Min/Max')
        self.layer_stack_reset_curr_min_max.setShortcut(Qt.Qt.Key_M)
        self.layer_stack_reset_curr_min_max.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.layer_stack_reset_curr_min_max.triggered.connect(self._on_reset_min_max)
        self.layer_stack_toggle_curr_auto_min_max = Qt.QAction(self)
        self.layer_stack_toggle_curr_auto_min_max.setText('Toggle Auto Min/Max')
        self.layer_stack_toggle_curr_auto_min_max.setShortcut(Qt.Qt.Key_A)
        self.layer_stack_toggle_curr_auto_min_max.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.layer_stack_toggle_curr_auto_min_max.triggered.connect(self._on_toggle_auto_min_max)
        self.addAction(self.layer_stack_toggle_curr_auto_min_max) # Necessary for shortcut to work as this action does not appear in a menu or toolbar
        self.layer_stack_reset_curr_gamma = Qt.QAction(self)
        self.layer_stack_reset_curr_gamma.setText('Reset \u03b3')
        self.layer_stack_reset_curr_gamma.setShortcut(Qt.Qt.Key_G)
        self.layer_stack_reset_curr_gamma.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.layer_stack_reset_curr_gamma.triggered.connect(self._on_reset_gamma)
        if sys.platform == 'darwin':
            self.exit_fullscreen_action = Qt.QAction(self)
            # If self.exit_fullscreen_action's text were "Exit Full Screen Mode" as we desire,
            # we would not be able to add it as a menu entry (http://doc.qt.io/qt-5/qmenubar.html#qmenubar-on-os-x).
            # "Leave Full Screen Mode" is a compromise.
            self.exit_fullscreen_action.setText('Leave Full Screen Mode')
            self.exit_fullscreen_action.triggered.connect(self.showNormal)
            self.exit_fullscreen_action.setShortcut(Qt.Qt.Key_Escape)
            self.exit_fullscreen_action.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.main_view.zoom_to_fit_action.setShortcut(Qt.Qt.Key_QuoteLeft)
        self.main_view.zoom_to_fit_action.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.main_view.zoom_one_to_one_action.setShortcut(Qt.Qt.Key_1)
        self.main_view.zoom_one_to_one_action.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.main_scene.layer_stack_item.examine_layer_mode_action.setShortcut(Qt.Qt.Key_Space)
        self.main_scene.layer_stack_item.examine_layer_mode_action.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.main_scene_snapshot_action = Qt.QAction(self)
        self.main_scene_snapshot_action.setText('Main View Snapshot')
        self.main_scene_snapshot_action.setShortcut(Qt.Qt.Key_S)
        self.main_scene_snapshot_action.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.main_scene_snapshot_action.setToolTip('Append snapshot of .main_view to .flipbook.pages')

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

    def _init_scenes_and_views(self, LayerStackItemClass, GeneralSceneClass, GeneralViewClass, GeneralViewContextualInfoItemClass,
                               HistogramItemClass, HistogramSceneClass, HistogramViewClass, HistgramViewContextualInfoItemClass):
        self.main_scene = GeneralSceneClass(self, LayerStackItemClass, self._get_primary_image_stack_current_layer_idx, GeneralViewContextualInfoItemClass)
        self.main_view = GeneralViewClass(self.main_scene, self)
        self.setCentralWidget(self.main_view)
        self.histogram_scene = HistogramSceneClass(self, self.main_scene.layer_stack_item, HistogramItemClass, HistgramViewContextualInfoItemClass)
        self.histogram_dock_widget = Qt.QDockWidget('Histogram', self)
        self.histogram_view, self._histogram_frame = HistogramViewClass.make_histogram_view_and_frame(self.histogram_scene, self.histogram_dock_widget)
        self.histogram_dock_widget.setWidget(self._histogram_frame)
        self.histogram_dock_widget.setAllowedAreas(Qt.Qt.BottomDockWidgetArea | Qt.Qt.TopDockWidgetArea)
        self.histogram_dock_widget.setFeatures(
            Qt.QDockWidget.DockWidgetClosable | Qt.QDockWidget.DockWidgetFloatable |
            Qt.QDockWidget.DockWidgetMovable | Qt.QDockWidget.DockWidgetVerticalTitleBar)
        self.addDockWidget(Qt.Qt.BottomDockWidgetArea, self.histogram_dock_widget)
        self.layer_stack_table_dock_widget = Qt.QDockWidget('Layer Stack', self)
        self.layer_stack_table_model = LayerStackTableModel(
            self.main_scene.layer_stack_item.override_enable_auto_min_max_action,
            self.main_scene.layer_stack_item.examine_layer_mode_action)
        self.layer_stack_table_model_inverter = InvertingProxyModel(self.layer_stack_table_model)
        self.layer_stack_table_model_inverter.setSourceModel(self.layer_stack_table_model)
        self.layer_stack_table_view = LayerStackTableView(self.layer_stack_table_model)
        self.layer_stack_table_view.setModel(self.layer_stack_table_model_inverter)
        self.layer_stack_table_model.setParent(self.layer_stack_table_view)
        self.layer_stack_table_selection_model = self.layer_stack_table_view.selectionModel()
        self.layer_stack_table_selection_model.currentRowChanged.connect(self._on_layer_stack_table_current_idx_changed)
        self.layer_stack_table_dock_widget.setWidget(self.layer_stack_table_view)
        self.layer_stack_table_dock_widget.setAllowedAreas(Qt.Qt.AllDockWidgetAreas)
        self.layer_stack_table_dock_widget.setFeatures(Qt.QDockWidget.DockWidgetClosable | Qt.QDockWidget.DockWidgetFloatable | Qt.QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.Qt.TopDockWidgetArea, self.layer_stack_table_dock_widget)

#   def make_flipbook(self, images=None, name='Flipbook'):
#       """The images argument may be any mixture of ris_widget.image.Image objects and raw data iterables of the sort that
#       may be assigned to RisWidget.image_data or RisWidget.image_data_T.
#       If None is supplied for images, an empty flipbook is created."""
#       if images is not None:
#           if not isinstance(images, SignalingList):
#               images = SignalingList([image if isinstance(image, Image) else Image(image, name=str(image_idx)) for image_idx, image in enumerate(images)])
#       flipbook = (self.layer_stack, self.layer_stack_table_selection_model, images)
#       flipbook.setAttribute(Qt.Qt.WA_DeleteOnClose)
#       dock_widget = Qt.QDockWidget(name, self)
#       dock_widget.setAttribute(Qt.Qt.WA_DeleteOnClose)
#       dock_widget.setWidget(flipbook)
#       flipbook.destroyed.connect(dock_widget.deleteLater) # Get rid of containing dock widget when flipbook is programatically destroyed
#       dock_widget.setAllowedAreas(Qt.Qt.LeftDockWidgetArea | Qt.Qt.RightDockWidgetArea)
#       dock_widget.setFeatures(Qt.QDockWidget.DockWidgetClosable | Qt.QDockWidget.DockWidgetFloatable | Qt.QDockWidget.DockWidgetMovable)
#       self.addDockWidget(Qt.Qt.RightDockWidgetArea, dock_widget)
#       return flipbook

    def _init_main_flipbook(self):
        self._main_flipbook = fb = self.FlipbookClass(self)
        fb.current_page_changed.connect(self._on_flipbook_current_page_changed)
        self.main_flipbook_dock_widget = Qt.QDockWidget('Main Flipbook', self)
        self.main_flipbook_dock_widget.setWidget(fb)
        self.main_flipbook_dock_widget.setAllowedAreas(Qt.Qt.RightDockWidgetArea | Qt.Qt.LeftDockWidgetArea)
        self.main_flipbook_dock_widget.setFeatures(Qt.QDockWidget.DockWidgetClosable | Qt.QDockWidget.DockWidgetFloatable | Qt.QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.Qt.RightDockWidgetArea, self.main_flipbook_dock_widget)

    def _init_toolbars(self):
        self.main_view_toolbar = self.addToolBar('Main View')
        self.main_view_zoom_combo = Qt.QComboBox(self)
        self.main_view_toolbar.addWidget(self.main_view_zoom_combo)
        self.main_view_zoom_combo.setEditable(True)
        self.main_view_zoom_combo.setInsertPolicy(Qt.QComboBox.NoInsert)
        self.main_view_zoom_combo.setDuplicatesEnabled(True)
        self.main_view_zoom_combo.setSizeAdjustPolicy(Qt.QComboBox.AdjustToContents)
        for zoom in GeneralView._ZOOM_PRESETS:
            self.main_view_zoom_combo.addItem(self._format_zoom(zoom * 100) + '%')
        self.main_view_zoom_combo.setCurrentIndex(GeneralView._ZOOM_ONE_TO_ONE_PRESET_IDX)
        self.main_view_zoom_combo.activated[int].connect(self._main_view_zoom_combo_changed)
        self.main_view_zoom_combo.lineEdit().returnPressed.connect(self._main_view_zoom_combo_custom_value_entered)
        self.main_view.zoom_changed.connect(self._main_view_zoom_changed)
        self.main_view_toolbar.addAction(self.main_view.zoom_to_fit_action)
        self.main_view_toolbar.addAction(self.layer_stack_reset_curr_min_max)
        self.main_view_toolbar.addAction(self.layer_stack_reset_curr_gamma)
        self.main_view_toolbar.addAction(self.main_scene.layer_stack_item.override_enable_auto_min_max_action)
        self.main_view_toolbar.addAction(self.main_scene.layer_stack_item.examine_layer_mode_action)
        self.dock_widget_visibility_toolbar = self.addToolBar('Dock Widget Visibility')
        self.dock_widget_visibility_toolbar.addAction(self.layer_stack_table_dock_widget.toggleViewAction())
        self.dock_widget_visibility_toolbar.addAction(self.main_flipbook_dock_widget.toggleViewAction())
        self.dock_widget_visibility_toolbar.addAction(self.histogram_dock_widget.toggleViewAction())

    def _init_menus(self):
        mb = self.menuBar()
        m = mb.addMenu('View')
        if sys.platform == 'darwin':
            m.addAction(self.exit_fullscreen_action)
            m.addSeparator()
        m.addAction(self.main_view.zoom_to_fit_action)
        m.addAction(self.main_view.zoom_one_to_one_action)
        m.addSeparator()
        m.addAction(self.layer_stack_reset_curr_min_max)
        m.addAction(self.layer_stack_reset_curr_gamma)
        m.addAction(self.main_scene.layer_stack_item.examine_layer_mode_action)
        m.addSeparator()
        m.addAction(self.main_scene.layer_stack_item.layer_name_in_contextual_info_action)
        m.addAction(self.main_scene.layer_stack_item.image_name_in_contextual_info_action)

    def dragEnterEvent(self, event):
        event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        mime_data = event.mimeData()
        if mime_data.hasImage():
            image = Image.from_qimage(qimage=qimage, name=mime_data.urls()[0].toDisplayString() if mime_data.hasUrls() else None)
            if image is not None:
                layer = Layer(image=image)
                self.main_flipbook.pages[:] = [layer]
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
            if self.main_flipbook.pages_model.handle_dropped_files(fpaths, len(self.main_flipbook.pages), 0, Qt.QModelIndex()):
                event.accept()

    @property
    def layer_stack(self):
        '''If you wish to replace the current .layer_stack, it may be done by assigning to this
        property.  For example:
        import freeimage
        from ris_widget.layer import Layer
        rw.layer_stack = [Layer(freeimage.read(str(p))) for p in pathlib.Path('./').glob('*.png')]

        Assigning to rw.main_scene.layer_stack_item.layer_stack directly will not cause
        assignment to rw.layer_stack_table_model.signling_list (which contains Layers and
        is therefore a layer stack), leaving the contents of the main view and layer table
        out of sync.  The same is true for assigning to 
        rw.layer_stack_table_model.signling_list directly, mutatis mutandis.  rw.layer_stack's
        setter takes care of setting both.

        Although assigning directly to rw.main_scene.layer_stack_item.layer_stack
        or rw.layer_stack_table_model.signling_list is not recommended, modifying the SignalingList
        instance returned by either of these property getters is safe.  EG,
        rw.main_scene.layer_stack_item.layer_stack.insert(Layer(numpy.zeros((800,800), dtype=numpy.uint8)))
        will cause the layer stack table to update, provided that rw.layer_stack_table_model.signling_list
        and rw.main_scene.layer_stack_item.layer_stack refer to the same SignalingList, as they
        do by default.'''
        return self._layer_stack

    @layer_stack.setter
    def layer_stack(self, v):
        if self._layer_stack is not None:
            self._layer_stack.name_changed.disconnect(self._on_layer_stack_name_changed)
            self._layer_stack.inserted.disconnect(self._on_inserted_into_layer_stack)
            self._layer_stack.replaced.disconnect(self._on_replaced_in_layer_stack)
        if v is None:
            v = om.SignalingList()
        elif isinstance(v, (Image, numpy.ndarray)):
            v = om.SignalingList([Layer(v)])
        elif isinstance(v, Layer):
            v = om.SignalingList([v])
        elif not isinstance(v, om.SignalingList) and any(not hasattr(v, signal) for signal in ('inserted', 'removed', 'replaced', 'name_changed'))\
             or any(not isinstance(ve, Layer) for ve in v):
            # If v is not a SignalingList and also is missing at least one list modification signal that we need, or if at least one element of v
            # is not a Layer, convert v to a SignalingList of Layers
            v = om.SignalingList([ve if isinstance(ve, Layer) else Layer(ve) for ve in v])
        self._layer_stack = v
        v.name_changed.connect(self._on_layer_stack_name_changed)
        # Must be QueuedConnection in order to avoid race condition where self._on_inserted_into_layer_stack is
        # called before self.layer_stack_table_model._on_inserted, causing self._on_inserted_into_layer_stack to
        # attempt to make row 0 in self.layer_stack_table_view current before self.layer_stack_table_model
        # is even aware that a row has been inserted.
        v.inserted.connect(self._on_inserted_into_layer_stack, Qt.Qt.QueuedConnection)
        v.replaced.connect(self._on_replaced_in_layer_stack)
        self.main_scene.layer_stack_item.layer_stack = v
        self.layer_stack_table_model.signaling_list = v
        if v:
            self._on_inserted_into_layer_stack()
            self.histogram_scene.histogram_item.layer = self.current_layer

    def _get_primary_image_stack_current_layer_idx(self):
        # Selection model is with reference to table view's model, which is the inverting proxy model
        pmidx = self.layer_stack_table_selection_model.currentIndex()
        if pmidx.isValid():
            midx = self.layer_stack_table_model_inverter.mapToSource(pmidx)
            if midx.isValid():
                return midx.row()

    current_layer_idx = property(_get_primary_image_stack_current_layer_idx)

    @property
    def current_layer(self):
        """rw.current_layer: A convenience property equivalent to rw.layer_stack[rw.current_layer_idx], with a minor
        difference: in addition to instances of Layer, Image instances and even raw image data may be assigned to rw.layer.
        Image instances and raw image data assigned to rw.layer are wrapped in a Layer, or in an Image wrapped in a layer,
        as required."""
        idx = self.current_layer_idx
        if idx is not None:
            return self.layer_stack[idx]

    @current_layer.setter
    def current_layer(self, v):
        idx = self.current_layer_idx
        if idx is None:
            raise IndexError('No row in .layer_stack_table_view is current/focused.')
        else:
            if not isinstance(v, Layer):
                v = Layer(v)
            self.layer_stack[idx] = v

    @property
    def layer(self):
        """rw.layer: A convenience property equivalent to rw.layer_stack[0], with minor differences:
        * If len(rw.layer_stack) == 0, querying rw.layer causes a new Layer to be inserted at rw.layer_stack[0] and
        returned, and assigning to rw.layer causes the assigned thing to be inserted at rw.layer_stack[0].
        * In addition to instances of Layer, Image instances and even raw image data may be assigned to rw.layer.
        Image instances and raw image data assigned to rw.layer are wrapped in a Layer, or in an Image wrapped
        in a layer, as required."""
        layer_stack = self.layer_stack
        if not layer_stack:
            layer = Layer()
            layer_stack.insert(0, layer)
            return layer
        return layer_stack[0]

    @layer.setter
    def layer(self, v):
        if not isinstance(v, Layer):
            v = Layer(v)
        layer_stack = self.layer_stack
        if layer_stack:
            layer_stack[0] = v
        else:
            layer_stack.insert(0, layer)

    @property
    def image(self):
        """rw.image: A Convenience property exactly equivalent to rw.layer.image, and equivalent to 
        rw.layer_stack[0].image with a minor difference: if len(rw.layer_stack) == 0, a query of rw.image
        returns None rather than raising an exception, and an assignment to it in this scenario is
        equivalent to rw.layer_stack.insert(0, Layer(v))."""
        return self.layer.image

    @image.setter
    def image(self, v):
        self.layer.image = v

    @property
    def main_flipbook(self):
        return self._main_flipbook

    def _on_flipbook_current_page_changed(self, flipbook, idx):
        page = None if idx < 0 else flipbook.pages[idx]
        if isinstance(page, progress_thread_pool.Task):
            return
        self.layer_stack = page

    def _on_layer_stack_name_changed(self, layer_stack):
        assert layer_stack is self.layer_stack
        name = layer_stack.name
        dw_title = 'Layer Stack'
        if len(name) > 0:
            dw_title += ' "{}"'.format(name)
        self.layer_stack_table_dock_widget.setWindowTitle(dw_title)

    def _on_layer_stack_table_current_idx_changed(self, midx, prev_midx):
        row = self.current_layer_idx
        layer = None if row is None else self.layer_stack[row]
        self.layer_stack_table_model.on_view_current_row_changed(row)
        self.histogram_scene.histogram_item.layer = layer
        lsi = self.main_scene.layer_stack_item
        if lsi.examine_layer_mode_enabled:
            # The appearence of a layer_stack_item may depend on which layer table row is current when
            # "examine layer mode" is enabled.
            lsi.update()

    def _on_inserted_into_layer_stack(self, idx=None, layers=None):
        if not self.layer_stack_table_selection_model.currentIndex().isValid():
            self.layer_stack_table_selection_model.setCurrentIndex(
                self.layer_stack_table_model_inverter.index(0, 0),
                Qt.QItemSelectionModel.SelectCurrent | Qt.QItemSelectionModel.Rows)

    def _on_replaced_in_layer_stack(self, idxs, old_layers, new_layers):
        self._on_inserted_into_layer_stack()
        current_midx = self.layer_stack_table_selection_model.currentIndex()
        if current_midx.isValid():
            try:
                change_idx = idxs.index(current_midx.row())
            except ValueError:
                return
            old_current, new_current = old_layers[change_idx], new_layers[change_idx]
            self.histogram_scene.histogram_item.layer = new_current

    def _main_view_zoom_changed(self, zoom_preset_idx, custom_zoom):
        assert zoom_preset_idx == -1 and custom_zoom != 0 or zoom_preset_idx != -1 and custom_zoom == 0, \
               'zoom_preset_idx XOR custom_zoom must be set.'
        if zoom_preset_idx == -1:
            self.main_view_zoom_combo.lineEdit().setText(self._format_zoom(custom_zoom * 100) + '%')
        else:
            self.main_view_zoom_combo.setCurrentIndex(zoom_preset_idx)

    def _main_view_zoom_combo_changed(self, idx):
        self.main_view.zoom_preset_idx = idx

    def _main_view_zoom_combo_custom_value_entered(self):
        txt = self.main_view_zoom_combo.lineEdit().text()
        percent_pos = txt.find('%')
        scale_txt = txt if percent_pos == -1 else txt[:percent_pos]
        try:
            self.main_view.custom_zoom = float(scale_txt) * 0.01
        except ValueError:
            e = 'Please enter a number between {} and {}.'.format(
                self._format_zoom(GeneralView._ZOOM_MIN_MAX[0] * 100),
                self._format_zoom(GeneralView._ZOOM_MIN_MAX[1] * 100))
            Qt.QMessageBox.information(self, 'self.windowTitle() Input Error', e)
            self.main_view_zoom_combo.setFocus()
            self.main_view_zoom_combo.lineEdit().selectAll()

    def _on_reset_min_max(self):
        layer = self.current_layer
        if layer is not None:
            del layer.min
            del layer.max

    def _on_reset_gamma(self):
        layer = self.current_layer
        if layer is not None:
            del layer.gamma

    def _on_toggle_auto_min_max(self):
        layer = self.current_layer
        if layer is not None:
            layer.auto_min_max_enabled = not layer.auto_min_max_enabled

if __name__ == '__main__':
    import sys
    app = Qt.QApplication(sys.argv)
    rw = RisWidget()
    rw.show()
    app.exec_()
