# The MIT License (MIT)
#
# Copyright (c) 2014-2016 WUSTL ZPLAB
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

import sip
sip.setdestroyonexit(True)

import atexit
from PyQt5 import Qt
import sys
from . import async_texture
from .layer import Layer
from .layer_stack import LayerList, LayerStack
from .qwidgets.flipbook import Flipbook
from .qwidgets.fps_display import FPSDisplay
from .qwidgets.layer_table import InvertingProxyModel, LayerTableModel, LayerTableView
from .qwidgets.layer_stack_painter import LayerStackPainter
from .qgraphicsscenes.general_scene import GeneralScene
from .qgraphicsviews.general_view import GeneralView
from .qgraphicsscenes.histogram_scene import HistogramScene
from .qgraphicsviews.histogram_view import HistogramView
from .shared_resources import GL_QSURFACE_FORMAT, FREEIMAGE, query_gl_exts

if sys.platform == 'darwin':
    class NonTransientScrollbarsStyle(Qt.QProxyStyle):
        def styleHint(self, sh, option=None, widget=None, returnData=None):
            if sh == Qt.QStyle.SH_ScrollBar_Transient:
                return 0
            return self.baseStyle().styleHint(sh, option, widget, returnData)

def _atexit_cleanup():
    # With IPython's Qt event loop integration installed, the Qt.QApplication.aboutToQuit signal is not emitted
    # when the Python interpreter exits.  However, we must do certain things before last-pass garbage collection
    # at interpreter exit time in order to avoid segfaulting, and these things are done in response to the
    # Qt.QApplication.aboutToQuit signal.  Fortunately, we can cause Qt.QApplication.aboutToQuit emission
    # ourselves.  Doing so at exit time prompts our cleanup routines, avoiding the segfault upon exit from
    # an IPython session owning one or more RisWidgets.
    app = Qt.QApplication.instance()
    if app is None:
        return
    from . shared_resources import _GL_CACHE, _on_destruction_of_context_with_cached_gl
    for context in _GL_CACHE.keys():
        context.destroyed[Qt.QObject].disconnect(_on_destruction_of_context_with_cached_gl)
    app.aboutToQuit.emit()
    # app.processEvents()

class RisWidgetQtObject(Qt.QMainWindow):
    main_view_change_signal = Qt.pyqtSignal(Qt.QTransform, Qt.QRectF)

    def __init__(
            self,
            app_prefs_name,
            app_prefs_version,
            window_title='RisWidget',
            parent=None,
            window_flags=Qt.Qt.WindowFlags(0),
            msaa_sample_count=2,
            swap_interval=1,
            layers=tuple()):
        super().__init__(parent, window_flags)
        self.app_prefs_name = app_prefs_name
        self.app_prefs_version = app_prefs_version
        self._shown = False
        # TODO: look deeper into opengl buffer swapping order and such to see if we can become compatible with OS X auto-hiding scrollbars
        # rather than needing to disable them
        if sys.platform == 'darwin':
            style = Qt.QApplication.style()
            Qt.QApplication.setStyle(NonTransientScrollbarsStyle(style))
        if window_title is not None:
            self.setWindowTitle(window_title)
        self.setAcceptDrops(True)
        query_gl_exts()
        Qt.QSurfaceFormat.setDefaultFormat(GL_QSURFACE_FORMAT(msaa_sample_count, swap_interval))
        async_texture._TextureCache.init()
        self.layer_stack = LayerStack()
        self._init_scenes_and_views()
        self._init_flipbook()
        self._init_layer_stack_painter()
        self._init_actions()
        self._init_toolbars()
        self._init_menus()
        if layers:
            self.layer_stack.layers = layers
        # RisWidgetQtObject's C++ personality is the QObject-N-parent of lots of Qt stuff that does not appreciate
        # being destroyed by Python's last-pass garbage collection or even simply when no QApplication is running.
        # Therefore, we connect the running QApplication's about to quit signal to our own C++ personality's
        # deleteLater method (the final thing QApplication does as it quits, after emitting the about to quit signal,
        # is delete everything queued up for deletion by deleteLater calls).  Thus, all of our QObject offspring
        # are culled long before the Great Garbage Battle prophesied to occur at The End of The Graceful Shutdown.
        Qt.QApplication.instance().aboutToQuit.connect(self._on_about_to_quit)
        atexit.register(_atexit_cleanup)

    def _on_about_to_quit(self):
        self.deleteLater()
        atexit.unregister(_atexit_cleanup)

    def _init_actions(self):
        self.flipbook_focus_prev_page_action = Qt.QAction(self)
        self.flipbook_focus_prev_page_action.setText('Previous Page')
        self.flipbook_focus_prev_page_action.setShortcut(Qt.Qt.Key_PageUp)
        self.flipbook_focus_prev_page_action.triggered.connect(self.flipbook.focus_prev_page)
        self.flipbook_focus_prev_page_action.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.flipbook_focus_next_page_action = Qt.QAction(self)
        self.flipbook_focus_next_page_action.setText('Next Page')
        self.flipbook_focus_next_page_action.setShortcut(Qt.Qt.Key_PageDown)
        self.flipbook_focus_next_page_action.triggered.connect(self.flipbook.focus_next_page)
        self.flipbook_focus_next_page_action.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.layer_stack_reset_curr_min_max_action = Qt.QAction(self)
        self.layer_stack_reset_curr_min_max_action.setText('Reset Min/Max')
        self.layer_stack_reset_curr_min_max_action.setShortcut(Qt.Qt.Key_M)
        self.layer_stack_reset_curr_min_max_action.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.layer_stack_reset_curr_min_max_action.triggered.connect(self._on_reset_min_max)
        self.layer_stack_toggle_curr_auto_min_max_action = Qt.QAction(self)
        self.layer_stack_toggle_curr_auto_min_max_action.setText('Toggle Auto Min/Max')
        self.layer_stack_toggle_curr_auto_min_max_action.setShortcut(Qt.Qt.Key_A)
        self.layer_stack_toggle_curr_auto_min_max_action.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.layer_stack_toggle_curr_auto_min_max_action.triggered.connect(self._on_toggle_auto_min_max)
        self.addAction(self.layer_stack_toggle_curr_auto_min_max_action) # Necessary for shortcut to work as this action does not appear in a menu or toolbar
        self.layer_stack_reset_curr_gamma_action = Qt.QAction(self)
        self.layer_stack_reset_curr_gamma_action.setText('Reset \u03b3')
        self.layer_stack_reset_curr_gamma_action.setShortcut(Qt.Qt.Key_G)
        self.layer_stack_reset_curr_gamma_action.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.layer_stack_reset_curr_gamma_action.triggered.connect(self._on_reset_gamma)
        self.layer_property_stack_save_action = Qt.QAction(self)
        self.layer_property_stack_save_action.setText('Save layer property stack as...')
        self.layer_property_stack_save_action.triggered.connect(self._on_save_layer_property_stack)
        self.layer_property_stack_load_action = Qt.QAction(self)
        self.layer_property_stack_load_action.setText('Load layer property stack from file...')
        self.layer_property_stack_load_action.triggered.connect(self._on_load_layer_property_stack)
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
        self.layer_stack.examine_layer_mode_action.setShortcut(Qt.Qt.Key_E)
        self.layer_stack.examine_layer_mode_action.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.layer_stack.auto_min_max_master_on_enabled_action.setShortcut(Qt.Qt.Key_F)
        self.layer_stack.auto_min_max_master_on_enabled_action.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.main_view_snapshot_action = Qt.QAction(self)
        self.main_view_snapshot_action.setText('Main View Snapshot')
        self.main_view_snapshot_action.setShortcut(Qt.Qt.Key_S)
        self.main_view_snapshot_action.setShortcutContext(Qt.Qt.ApplicationShortcut)
        self.main_view_snapshot_action.setToolTip('Take snapshot and show save image dialog.')
        self.main_view_snapshot_action.triggered.connect(self._on_main_view_snapshot_action)

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

    def _init_scenes_and_views(self):
        self.main_scene = GeneralScene(self, self.layer_stack)
        self.main_view = GeneralView(self.main_scene, self)
        self.setCentralWidget(self.main_view)
        self.histogram_scene = HistogramScene(self, self.layer_stack)
        self.histogram_dock_widget = Qt.QDockWidget('Histogram', self)
        self.histogram_view, self._histogram_frame = HistogramView.make_histogram_view_and_frame(self.histogram_scene, self.histogram_dock_widget)
        self.histogram_dock_widget.setWidget(self._histogram_frame)
        self.histogram_dock_widget.setAllowedAreas(Qt.Qt.BottomDockWidgetArea | Qt.Qt.TopDockWidgetArea)
        self.histogram_dock_widget.setFeatures(
            Qt.QDockWidget.DockWidgetClosable | Qt.QDockWidget.DockWidgetFloatable |
            Qt.QDockWidget.DockWidgetMovable | Qt.QDockWidget.DockWidgetVerticalTitleBar)
        self.addDockWidget(Qt.Qt.BottomDockWidgetArea, self.histogram_dock_widget)
        self.layer_table_dock_widget = Qt.QDockWidget('Layer Stack', self)
        self.layer_table_model = LayerTableModel(self.layer_stack)
        # NB: Qt.QAbstractItemView, an ancestor of InvertingProxyModel, attempts to start a QTimer as it is destroyed.  Therefore,
        # it must be destroyed before the event dispatcher thread local object is destroyed - IE, not by Python's last-pass garbage
        # collector, which collects in no particular order, often collecting the dispatcher before any stray item models.  To prompt
        # pre-last-pass destruction, it is sufficient to make all Qt.QAbstractItemView progeny QObject-parented to a QObject that
        # is definitely destroyed before the last pass.  We are careful to ensure that RisWidget instances meet this criterion.
        self.layer_table_model_inverter = InvertingProxyModel(self)
        self.layer_table_model_inverter.setSourceModel(self.layer_table_model)
        self.layer_table_view = LayerTableView(self.layer_table_model)
        self.layer_table_view.setModel(self.layer_table_model_inverter)
        self.layer_table_model.setParent(self.layer_table_view)
        self.layer_table_selection_model = self.layer_table_view.selectionModel()
        self.layer_stack.selection_model = self.layer_table_selection_model
        self.layer_table_dock_widget.setWidget(self.layer_table_view)
        self.layer_table_dock_widget.setAllowedAreas(Qt.Qt.AllDockWidgetAreas)
        self.layer_table_dock_widget.setFeatures(
            Qt.QDockWidget.DockWidgetClosable | Qt.QDockWidget.DockWidgetFloatable | Qt.QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.Qt.TopDockWidgetArea, self.layer_table_dock_widget)
        self.fps_display_dock_widget = Qt.QDockWidget('FPS', self)
        self.fps_display = FPSDisplay()
        self.main_scene.layer_stack_item.painted.connect(self.fps_display.notify)
        self.fps_display_dock_widget.setWidget(self.fps_display)
        self.fps_display_dock_widget.setAllowedAreas(Qt.Qt.AllDockWidgetAreas)
        self.fps_display_dock_widget.setFeatures(
            Qt.QDockWidget.DockWidgetClosable | Qt.QDockWidget.DockWidgetFloatable | Qt.QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.Qt.RightDockWidgetArea, self.fps_display_dock_widget)
        self.fps_display_dock_widget.hide()

    def _init_flipbook(self):
        self.flipbook = fb = Flipbook(self.layer_stack, self)
        self.flipbook_dock_widget = Qt.QDockWidget('Main Flipbook', self)
        self.flipbook_dock_widget.setWidget(fb)
        self.flipbook_dock_widget.setAllowedAreas(Qt.Qt.RightDockWidgetArea | Qt.Qt.LeftDockWidgetArea)
        self.flipbook_dock_widget.setFeatures(Qt.QDockWidget.DockWidgetClosable | Qt.QDockWidget.DockWidgetFloatable | Qt.QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.Qt.RightDockWidgetArea, self.flipbook_dock_widget)
        fb.pages_model.rowsInserted.connect(self._update_flipbook_visibility)
        fb.pages_model.rowsRemoved.connect(self._update_flipbook_visibility)
        fb.pages_model.modelReset.connect(self._update_flipbook_visibility)
        self.flipbook_dock_widget.hide()
        # Make the flipbook deal with drop events
        self.dragEnterEvent = self.flipbook.pages_view.dragEnterEvent
        self.dragMoveEvent = self.flipbook.pages_view.dragMoveEvent
        self.dropEvent = self.flipbook.pages_view.dropEvent

    def _init_layer_stack_painter(self):
        self.layer_stack_painter_dock_widget = Qt.QDockWidget('Layer Stack Painter', self)
        self.layer_stack_painter_dock_widget.setAllowedAreas(Qt.Qt.RightDockWidgetArea | Qt.Qt.LeftDockWidgetArea)
        self.layer_stack_painter_dock_widget.setFeatures(
            Qt.QDockWidget.DockWidgetClosable | Qt.QDockWidget.DockWidgetFloatable | Qt.QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.Qt.RightDockWidgetArea, self.layer_stack_painter_dock_widget)
        self.layer_stack_painter_dock_widget.hide()
        self.layer_stack_painter = None
        self.layer_stack_painter_dock_widget.toggleViewAction().toggled.connect(self._on_layer_stack_painter_dock_widget_visibility_toggled)

    def _on_layer_stack_painter_dock_widget_visibility_toggled(self, is_visible):
        if is_visible:
            if self.layer_stack_painter is None:
                self.layer_stack_painter = LayerStackPainter(self.main_scene.layer_stack_item)
                self.layer_stack_painter_dock_widget.setWidget(self.layer_stack_painter)
        else:
            if self.layer_stack_painter is not None:
                self.main_scene.removeItem(self.layer_stack_painter.painter_item)
                self.layer_stack_painter = None

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
        self.main_view_toolbar.addAction(self.layer_stack_reset_curr_min_max_action)
        self.main_view_toolbar.addAction(self.layer_stack_reset_curr_gamma_action)
        self.main_view_toolbar.addAction(self.layer_stack.auto_min_max_master_on_enabled_action)
        self.main_view_toolbar.addAction(self.layer_stack.examine_layer_mode_action)
        self.main_view_toolbar.addAction(self.main_view_snapshot_action)
        self.main_view_toolbar.addAction(self.flipbook.consolidate_selected_action)
        self.main_view_toolbar.addAction(self.flipbook.delete_selected_action)
        self.dock_widget_visibility_toolbar = self.addToolBar('Dock Widget Visibility')
        self.dock_widget_visibility_toolbar.addAction(self.layer_table_dock_widget.toggleViewAction())
        self.dock_widget_visibility_toolbar.addAction(self.layer_stack_painter_dock_widget.toggleViewAction())
        self.dock_widget_visibility_toolbar.addAction(self.histogram_dock_widget.toggleViewAction())
        self.dock_widget_visibility_toolbar.addAction(self.flipbook_dock_widget.toggleViewAction())
        self.dock_widget_visibility_toolbar.addAction(self.fps_display_dock_widget.toggleViewAction())

    def _init_menus(self):
        mb = self.menuBar()
        m = mb.addMenu('File')
        m.addAction(self.layer_property_stack_save_action)
        m.addAction(self.layer_property_stack_load_action)
        m = mb.addMenu('View')
        if sys.platform == 'darwin':
            m.addAction(self.exit_fullscreen_action)
            m.addSeparator()
        m.addAction(self.main_view.zoom_to_fit_action)
        m.addAction(self.main_view.zoom_one_to_one_action)
        m.addSeparator()
        m.addAction(self.flipbook_focus_prev_page_action)
        m.addAction(self.flipbook_focus_next_page_action)
        m.addAction(self.flipbook.toggle_playing_action)
        m.addSeparator()
        m.addAction(self.layer_stack_reset_curr_min_max_action)
        m.addAction(self.layer_stack_reset_curr_gamma_action)
        m.addAction(self.layer_stack.auto_min_max_master_on_enabled_action)
        m.addAction(self.layer_stack.examine_layer_mode_action)
        m.addSeparator()
        m.addAction(self.layer_stack.layer_name_in_contextual_info_action)
        m.addAction(self.layer_stack.image_name_in_contextual_info_action)
        m.addAction(self.layer_stack.histogram_alternate_column_shading_action)

    def showEvent(self, event):
        if not self._shown:
            self._shown = True
            settings = Qt.QSettings("zplab", self.app_prefs_name)
            geometry = settings.value('main_window_geometry')
            #state = settings.value('main_window_state')
            if None not in (geometry,):# state):
                self.restoreGeometry(geometry)
                #self.restoreState(state, self.APP_PREFS_VERSION)
        super().showEvent(event)

    def closeEvent(self, event):
        settings = Qt.QSettings('zplab', self.app_prefs_name)
        settings.setValue('main_window_geometry', self.saveGeometry())
        #settings.setValue('main_window_state', self.saveState(self.APP_PREFS_VERSION))
        super().closeEvent(event)

    @property
    def layers(self):
        """If you wish to replace the current .layers, it may be done by assigning to this property.  For example:
        import freeimage
        from ris_widget.layer import Layer
        rw.layers = [Layer(freeimage.read(str(p))) for p in pathlib.Path('./').glob('*.png')]."""
        return self.layer_stack.layers

    @layers.setter
    def layers(self, v):
        self.layer_stack.layers = v

    @property
    def focused_layer(self):
        """rw.focused_layer: A convenience property equivalent to rw.layer_stack.focused_layer."""
        return self.layer_stack.focused_layer

    @focused_layer.setter
    def focused_layer(self, v):
        self.layer_stack.focused_layer = v

    @property
    def layer(self):
        """rw.layer: A convenience property equivalent to rw.layers[0] and rw.layer_stack.layers[0], with minor differences:
        * If rw.layers is None: Querying rw.layer causes rw.layers to be set to a LayerList containing a single empty Layer which is returned,
        while assigning to rw.layer causes rw.layers to be set to a LayerList containing the thing assigned (wrapped in a Layer as needed).
        * If len(rw.layers) == 0: Querying rw.layer causes a new Layer to be inserted at rw.layers[0] and returned, while assigning to
        rw.layer causes the assigned thing to be inserted at rw.layers[0] (wrapped in a Layer as needed)."""
        layers = self.layers
        if layers:
            layer = layers[0]
        else:
            layer = Layer()
            if layers is None:
                self.layers = [layer]
            else:
                layers.append(layer)
        return layer

    @layer.setter
    def layer(self, v):
        layers = self.layers
        if layers:
            layers[0] = v
        else:
            if layers is None:
                self.layers = v
            else:
                layers.append(v)

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
    def mask(self):
        return self.layer_stack.imposed_image_mask

    @mask.setter
    def mask(self, v):
        self.layer_stack.imposed_image_mask = v

    def _on_flipbook_pages_inserted(self, parent, first_idx, last_idx):
        self._update_flipbook_visibility()

    def _on_flipbook_pages_removed(self, parent, first_idx, last_idx):
        self._update_flipbook_visibility()

    def _on_flipbook_pages_list_replaced(self):
        self._update_flipbook_visibility()

    def _update_flipbook_visibility(self):
        fb_is_visible = self.flipbook_dock_widget.isVisible()
        if (len(self.flipbook.pages) > 0) != fb_is_visible:
            self.flipbook_dock_widget.hide() if fb_is_visible else self.flipbook_dock_widget.show()

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
            Qt.QMessageBox.information(self, self.windowTitle() + ' Input Error', e)
            self.main_view_zoom_combo.setFocus()
            self.main_view_zoom_combo.lineEdit().selectAll()

    def _on_reset_min_max(self):
        layer = self.focused_layer
        if layer is not None:
            del layer.min
            del layer.max

    def _on_reset_gamma(self):
        layer = self.focused_layer
        if layer is not None:
            del layer.gamma

    def _on_toggle_auto_min_max(self):
        layer = self.focused_layer
        if layer is not None:
            layer.auto_min_max_enabled = not layer.auto_min_max_enabled

    def _on_main_view_snapshot_action(self):
        freeimage = FREEIMAGE(show_messagebox_on_error=True, error_messagebox_owner=self, is_read=False)
        if freeimage:
            try:
                snapshot = self.main_view.snapshot()
            except RuntimeError as e:
                Qt.QMessageBox.information(self, self.windowTitle() + ' Snapshot Error', e)
            else:
                if sys.platform == 'darwin':
                    # With the current version of PyQt (5.5.2) and below, if IPython's Qt event loop intergration is
                    # active, the OS X native modal file save dialog ends up being dismissed just as it appears.  This
                    # can be avoided by using Qt's own reasonably good save dialog.  This will probably eventually be
                    # fixed by the IPython developers, at which point this if clause may be deleted with the contents
                    # of the associated else clause unindented and left as the only remaining code path.
                    fn, _ = Qt.QFileDialog.getSaveFileName(
                        self,
                        'Save Snapshot',
                        filter='Images (*.png *.jpg *.tiff *.tif)',
                        options=Qt.QFileDialog.DontUseNativeDialog)
                else:
                    fn, _ = Qt.QFileDialog.getSaveFileName(
                        self,
                        'Save Snapshot',
                        filter='Images (*.png *.jpg *.tiff *.tif)')
                if fn:
                    try:
                        freeimage.write(snapshot, fn)
                    except Exception as e:
                        Qt.QMessageBox.information(self, self.windowTitle() + ' Freeimage Error', type(e).__name__ + ': ' + str(e))

    def _on_save_layer_property_stack(self):
        if sys.platform == 'darwin':
            # With the current version of PyQt (5.5.2) and below, if IPython's Qt event loop intergration is
            # active, the OS X native modal file save dialog ends up being dismissed just as it appears.  This
            # can be avoided by using Qt's own reasonably good save dialog.  This will probably eventually be
            # fixed by the IPython developers, at which point this if clause may be deleted with the contents
            # of the associated else clause unindented and left as the only remaining code path.
            fn, _ = Qt.QFileDialog.getSaveFileName(
                self,
                'Save Layer Property Stack',
                filter='JSON (*.json *.jsn)',
                options=Qt.QFileDialog.DontUseNativeDialog)
        else:
            fn, _ = Qt.QFileDialog.getSaveFileName(
                self,
                'Save Layer Property Stack',
                filter='JSON (*.json *.jsn)')
        if fn:
            try:
                f = open(fn, 'w')
            except PermissionError as e:
                Qt.QMessageBox.information(
                    self, self.windowTitle() + ' File Error', type(e).__name__ + ': ' + str(e))
                return
            try:
                f.write(self.layers.to_json())
            finally:
                f.close()

    def _on_load_layer_property_stack(self):
        if sys.platform == 'darwin':
            # With the current version of PyQt (5.5.2) and below, if IPython's Qt event loop intergration is
            # active, the OS X native modal file save dialog ends up being dismissed just as it appears.  This
            # can be avoided by using Qt's own reasonably good save dialog.  This will probably eventually be
            # fixed by the IPython developers, at which point this if clause may be deleted with the contents
            # of the associated else clause unindented and left as the only remaining code path.
            fn, _ = Qt.QFileDialog.getOpenFileName(
                self,
                'Load Layer Property Stack',
                filter='JSON (*.json *.jsn)',
                options=Qt.QFileDialog.DontUseNativeDialog)
        else:
            fn, _ = Qt.QFileDialog.getOpenFileName(
                self,
                'Load Layer Property Stack',
                filter='JSON (*.json *.jsn)')
        if fn:
            try:
                f = open(fn, 'r')
            except PermissionError as e:
                Qt.QMessageBox.information(
                    self, self.windowTitle() + ' File Error', type(e).__name__ + ': ' + str(e))
                return
            try:
                l = LayerList.from_json(f.read(), show_error_messagebox=True)
                if l is not None:
                    self.layers = l
            finally:
                f.close()

class ProxyProperty(property):
    def __init__(self, name, owner_name, owner_type):
        self.owner_name = owner_name
        self.proxied_property = getattr(owner_type, name)
        self.__doc__ = getattr(owner_type, '__doc__')

    def __get__(self, obj, _=None):
        if obj is None:
            return self.proxied_property
        return self.proxied_property.fget(getattr(obj, self.owner_name))

    def __set__(self, obj, v):
        self.proxied_property.fset(getattr(obj, self.owner_name), v)

    def __delete__(self, obj):
        self.proxied_property.fdel(getattr(obj, self.owner_name))

AUTO_CREATED_QAPPLICATION = None

class RisWidget:
    """RisWidget: A window for viewing images and image stacks from files and live data sources, containing
    an interactive histogram, list view of loaded images and image stacks, and an extensible graphics view.
    The RisWidget class itself presents a handful of properties and attributes, signals, and methods that tend
    to be the most used for day-to-day tasks.  These are proxies back to the their implementations in
    RisWidgetQtObject and its constituent components.  The wrapped RisWidgetQtObject is available as
    .qt_object (eg, rw.qt_object, where rw is a RisWidget instance).

    Signals:
    * main_view_mouse_movement_signal(view_coordinate, scene_coordinate)
    * main_view_right_click_signal(view_coordinate, scene_coordinate)"""
    APP_PREFS_NAME = "RisWidget"
    APP_PREFS_VERSION = 1
    COPY_REFS = [
        'flipbook',
        'main_scene',
        'main_view',
        ('main_scene.viewport_rect_item', 'main_viewport_rect_item'),
        ('main_scene.layer_stack_item', 'layer_stack_item'),
        'show',
        'hide',
        'close'
    ]
    QT_OBJECT_CLASS = RisWidgetQtObject

    def __init__(
            self,
            window_title='RisWidget',
            parent=None,
            window_flags=Qt.Qt.WindowFlags(0),
            msaa_sample_count=2,
            swap_interval=1,
            show=True,
            layers = tuple(),
            **kw):
        global AUTO_CREATED_QAPPLICATION
        if Qt.QApplication.instance() is None:
            AUTO_CREATED_QAPPLICATION = Qt.QApplication(sys.argv)
        self.qt_object = self.QT_OBJECT_CLASS(
            app_prefs_name=self.APP_PREFS_NAME,
            app_prefs_version=self.APP_PREFS_VERSION,
            window_title=window_title,
            parent=parent,
            window_flags=window_flags,
            msaa_sample_count=msaa_sample_count,
            swap_interval=swap_interval,
            layers=layers,
            **kw)
        self.main_view_change_signal = self.qt_object.main_view_change_signal
        for refdesc in self.COPY_REFS:
            if isinstance(refdesc, str):
                path, name = refdesc, refdesc
            else:
                path, name = refdesc
                obj = self.qt_object
            obj = self.qt_object
            for element in path.split('.'):
                obj = getattr(obj, element)
            setattr(self, name, obj)
        self.add_image_files_to_flipbook = self.flipbook.add_image_files
        self.snapshot = self.qt_object.main_view.snapshot
        if show:
            self.show()

    def update(self):
        """Calling this method on the main thread updates all Qt widgets immediately, without requiring
        you to return from the current function or exit from the current loop.

        For example, the following code will create and show a RisWidget that appears to be non-responsive
        for ten seconds, after which a white square is displayed:

        import numpy
        from ris_widget.ris_widget import RisWidget; rw = RisWidget()
        import time
        rw.image = numpy.zeros((100,100), dtype=numpy.uint8)
        for intensity in numpy.linspace(0,255,100).astype(numpy.uint8):
            time.sleep(0.1)
            rw.image.data[:] = intensity
            rw.image.refresh()

        Adding an rw.update() call to the loop fixes this:

        import numpy
        from ris_widget.ris_widget import RisWidget; rw = RisWidget()
        import time
        rw.image = numpy.zeros((100,100), dtype=numpy.uint8)
        for intensity in numpy.linspace(0,255,100).astype(numpy.uint8):
            rw.update()
            time.sleep(0.1)
            rw.image.data[:] = intensity
            rw.image.refresh()
        """
        Qt.QApplication.processEvents()

    image = ProxyProperty('image', 'qt_object', RisWidgetQtObject)
    layer = ProxyProperty('layer', 'qt_object', RisWidgetQtObject)
    focused_layer = ProxyProperty('focused_layer', 'qt_object', RisWidgetQtObject)
    layers = ProxyProperty('layers', 'qt_object', RisWidgetQtObject)
    mask = ProxyProperty('mask', 'qt_object', RisWidgetQtObject)
    # It is not easy to spot the pages property of a flipbook amongst the many possibilities visibile in dir(Flipbook).  So,
    # although flipbook_pages saves no characters compared to flipbook.pages, flipbook_pages is nice to have.
    flipbook_pages = ProxyProperty('pages', 'flipbook', Flipbook)

if __name__ == '__main__':
    import sys
    app = Qt.QApplication(sys.argv)
    rw = RisWidget()
    rw.show()
    app.exec_()
