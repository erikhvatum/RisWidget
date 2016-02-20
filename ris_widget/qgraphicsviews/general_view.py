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

from .base_view import BaseView
import numpy
from PyQt5 import Qt

class GeneralView(BaseView):
    """Signals:
    * mouse_event_signal(mouse_event_type, mouse_event, scene_pos):  Emitted upon mouse activity in the viewport.  A handler
      function, when directly connected - with v.mouse_event_signal.connect(handler_function) and not
      v.mouse_event_signal.connect(handler_function, Qt.Qt.QueuedConnection) - may indicate that it has consumed the
      event by calling mouse_event.accept().  Other handlers connected to mouse_event_signal are still called, but the
      view takes no further action in response to the event.
    * key_event_signal(key_event_type, key_event): Like mouse_event_signal.
    * zoom_changed(zoom_level, custom_zoom_ratio)"""

    def _make_zoom_presets(self=None):
        def f(x, step_zoomout=0.125, step_zoomin=0.2):
            if x == 0:
                return 1
            if x < 0:
                return 2**(x*step_zoomout)
            return 1 + x*step_zoomin
        xs = numpy.linspace(15, -16, 32, endpoint=True, dtype=numpy.float64)
        ys = list(map(f, xs))
        return numpy.array(ys, dtype=numpy.float64)
    _ZOOM_PRESETS = _make_zoom_presets()
    _ZOOM_MIN_MAX = (.001, 3000.0)
    _ZOOM_ONE_TO_ONE_PRESET_IDX = 15
    _ZOOM_INCREMENT_BEYOND_PRESETS_FACTORS = (.8, 1.25)

    mouse_event_signal = Qt.pyqtSignal(str, Qt.QMouseEvent, Qt.QPointF)
    key_event_signal = Qt.pyqtSignal(str, Qt.QKeyEvent)
    zoom_changed = Qt.pyqtSignal(int, float)

    def __init__(self, base_scene, parent):
        super().__init__(base_scene, parent)
        self.setMinimumSize(Qt.QSize(100,100))
        self._zoom_preset_idx = self._ZOOM_ONE_TO_ONE_PRESET_IDX
        self._custom_zoom = 0
        self.zoom_to_fit_action = Qt.QAction('Zoom to Fit', self)
        self.zoom_to_fit_action.setCheckable(True)
        self.zoom_to_fit_action.setChecked(False)
        self._ignore_zoom_to_fit_action_toggle = False
        self.zoom_to_fit_action.toggled.connect(self.on_zoom_to_fit_action_toggled)
        self.zoom_one_to_one_action = Qt.QAction('1:1 Zoom', self)
        self.zoom_one_to_one_action.triggered.connect(lambda: GeneralView.zoom_preset_idx.fset(self, GeneralView._ZOOM_ONE_TO_ONE_PRESET_IDX))
        # Calling self.setDragMode(Qt.QGraphicsView.ScrollHandDrag) would enable QGraphicsView's built-in
        # click-drag panning, saving us from having to implement it.  However, QGraphicsView is very
        # insistent about setting the mouse cursor to the hand icon in ScrollHandDragMode.  It does this
        # in a number of places that would have to be invidually overridden, making it much simpler to
        # implement click-drag panning ourselves.
        self.setDragMode(Qt.QGraphicsView.NoDrag)
        self._panning = False
        self.setAcceptDrops(True)
        # Mouse tracking generally seems to be enabled for QGraphicsViews, but the documentation does
        # not state that this is always the case, and in fact, does state that mouse tracking defaults
        # to disabled for QWidgets - and QGraphicsView has QWidget as a base class.  With mouse tracking
        # disabled, we would receive mouse movement events only while a mouse button is held down.  This
        # is not desirable: the user may depend on mouse_movement_signal for in order to implement a hover
        # behavior, and this signal is emitted upon reception of a mouse movement event.  So, to be safe,
        # we explicitly enable mouse tracking.
        self.setMouseTracking(True)

    def _on_layer_stack_item_bounding_rect_changed(self):
        if self.zoom_to_fit:
            self._apply_zoom()

    def _on_resize(self, size):
        if self.zoom_to_fit:
            self._apply_zoom()

    def wheelEvent(self, event):
        wheel_delta = event.angleDelta().y()
        if wheel_delta != 0:
            zoom_in = wheel_delta > 0
            switched_to_custom = False
            if self._zoom_preset_idx != -1:
                if zoom_in:
                    if self._zoom_preset_idx == 0:
                        self._zoom_preset_idx = -1
                        self._custom_zoom = GeneralView._ZOOM_PRESETS[0]
                        switched_to_custom = True
                    else:
                        self._zoom_preset_idx -= 1
                else:
                    if self._zoom_preset_idx == GeneralView._ZOOM_PRESETS.shape[0] - 1:
                        self._zoom_preset_idx = -1
                        self._custom_zoom = GeneralView._ZOOM_PRESETS[-1]
                        switched_to_custom = True
                    else:
                        self._zoom_preset_idx += 1
            if self._zoom_preset_idx == -1:
                self._custom_zoom *= GeneralView._ZOOM_INCREMENT_BEYOND_PRESETS_FACTORS[zoom_in]
                if not switched_to_custom and self._custom_zoom <= GeneralView._ZOOM_PRESETS[0] and self._custom_zoom >= GeneralView._ZOOM_PRESETS[-1]:
                    # Jump to nearest preset if we are re-entering preset range
                    self._zoom_preset_idx = numpy.argmin(numpy.abs(GeneralView._ZOOM_PRESETS - self._custom_zoom))
                    self._custom_zoom = 0
            if self._zoom_preset_idx == -1:
                if zoom_in:
                    if self._custom_zoom > GeneralView._ZOOM_MIN_MAX[1]:
                        self._custom_zoom = GeneralView._ZOOM_MIN_MAX[1]
                else:
                    if self._custom_zoom < GeneralView._ZOOM_MIN_MAX[0]:
                        self._custom_zoom = GeneralView._ZOOM_MIN_MAX[0]
                desired_zoom = self._custom_zoom
            else:
                desired_zoom = self._ZOOM_PRESETS[self._zoom_preset_idx]
            # With transformationAnchor set to AnchorUnderMouse, QGraphicsView.scale modifies the view's transformation matrix such
            # that the same image pixel remains under the mouse cursor (except where the demands imposed by centering of an
            # undersized scene take priority).  But, that does mean we must modify the transformation via the view's scale function
            # and not by direct manipulation of the view's transformation matrix.  Thus, it is necessary to find the current
            # scaling in order to compute the factor by which scaling must be mulitplied in order to arrive at our desired scaling.
            # This found by taking the view's current vertical scaling under the assumption that a square in the scene will appear
            # as a square in the view (and not a rectangle or trapezoid), which holds if we are displaying images with square
            # pixels - the only kind we support.
            current_zoom = self.transform().m22()
            scale_zoom = desired_zoom / current_zoom
            self.setTransformationAnchor(Qt.QGraphicsView.AnchorUnderMouse)
            self.scale(scale_zoom, scale_zoom)
            self.setTransformationAnchor(Qt.QGraphicsView.AnchorViewCenter)
            if self.zoom_to_fit:
                self._ignore_zoom_to_fit_action_toggle = True
                self.zoom_to_fit_action.setChecked(False)
                self._ignore_zoom_to_fit_action_toggle = False
            self.scene_region_changed.emit(self)
            self.zoom_changed.emit(self._zoom_preset_idx, self._custom_zoom)

    def mousePressEvent(self, event):
        # For our convenience, Qt sets event accepted to true before calling us, so that we don't have to in the common case
        # where a handler handles the event and the event should be considered handled.  (Some unhandled events propagate
        # to parent widgets, so this can be important)
        event.setAccepted(False)
        # However, Qt's handlers are generally good citizens and will, often redundantly, set accepted to true
        # if they recognize and respond to an event.  QGraphicsView.mousePressEvent(..) is such a handler.
        super().mousePressEvent(event)
        if event.isAccepted():
            return
        # Next, we inform everyone connected to our mouse event slot.  Anyone directly connected can set event.accepted(),
        # preventing the following conditional block from executing
        self.mouse_event_signal.emit('press', event, self.mapToScene(event.pos()))
        if event.isAccepted():
            return
        # If the mouse click landed on an interactive scene item, super().mousePressEvent(event) would have set
        # event to accepted.  So, neither the view nor the scene wanted this mouse click, and we check if it is perhaps
        # a click-drag pan initiation - or even a right click, in which case we emit a signal for use by the user
        # (for example, in order to create a new item at a right-clicked location for a point picker).
        if event.button() == Qt.Qt.LeftButton:
            # It is, and we're handling this event
            self._panning = True
            self._panning_prev_mouse_pos = event.pos()
            event.setAccepted(True)

    def mouseReleaseEvent(self, event):
        event.setAccepted(False)
        super().mouseReleaseEvent(event)
        if event.isAccepted():
            return
        self.mouse_event_signal.emit('release', event, self.mapToScene(event.pos()))
        if not event.isAccepted() and event.button() == Qt.Qt.LeftButton and self._panning:
            self._panning = False
            del self._panning_prev_mouse_pos
            event.setAccepted(True)

    def mouseMoveEvent(self, event):
        event.setAccepted(False)
        super().mouseMoveEvent(event)
        if event.isAccepted():
            return
        self.mouse_event_signal.emit('move', event, self.mapToScene(event.pos()))
        if not event.isAccepted():
            if self._panning:
                # This block more or less borrowed from QGraphicsView::mouseMoveEvent(QMouseEvent *event), found in
                # qtbase/src/widgets/graphicsview/qgraphicsview.cpp
                hbar, vbar = self.horizontalScrollBar(), self.verticalScrollBar()
                pos = event.pos()
                delta = pos - self._panning_prev_mouse_pos
                hbar.setValue(hbar.value() + (delta.x() if self.isRightToLeft() else -delta.x()))
                vbar.setValue(vbar.value() - delta.y())
                self._panning_prev_mouse_pos = pos

    def keyPressEvent(self, event):
        event.setAccepted(False)
        super().keyPressEvent(event)
        if event.isAccepted():
            return
        self.key_event_signal.emit('press', event)

    def keyReleaseEvent(self, event):
        event.setAccepted(False)
        super().keyReleaseEvent(event)
        if event.isAccepted():
            return
        self.key_event_signal.emit('release', event)

    def dragEnterEvent(self, event):
        event.setAccepted(False)
        self.parent().dragEnterEvent(event)
        if not event.isAccepted():
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        event.setAccepted(False)
        self.parent().dragMoveEvent(event)
        if not event.isAccepted():
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        event.setAccepted(False)
        self.parent().dropEvent(event)
        if not event.isAccepted():
            super().dropEvent(event)

    def on_zoom_to_fit_action_toggled(self):
        if not self._ignore_zoom_to_fit_action_toggle:
            self._apply_zoom()

    @property
    def zoom_to_fit(self):
        return self.zoom_to_fit_action.isChecked()

    @zoom_to_fit.setter
    def zoom_to_fit(self, zoom_to_fit):
        self.zoom_to_fit_action.setChecked(zoom_to_fit)

    @property
    def custom_zoom(self):
        return self._custom_zoom

    @custom_zoom.setter
    def custom_zoom(self, custom_zoom):
        if custom_zoom < GeneralView._ZOOM_MIN_MAX[0] or custom_zoom > GeneralView._ZOOM_MIN_MAX[1]:
            raise ValueError('Value must be in the range [{}, {}].'.format(*GeneralView._ZOOM_MIN_MAX))
        self._custom_zoom = custom_zoom
        self._zoom_preset_idx = -1
        if self.zoom_to_fit:
            self._ignore_zoom_to_fit_action_toggle = True
            self.zoom_to_fit_action.setChecked(False)
            self._ignore_zoom_to_fit_action_toggle = False
        self._apply_zoom()

    @property
    def zoom_preset_idx(self):
        return self._zoom_preset_idx

    @zoom_preset_idx.setter
    def zoom_preset_idx(self, idx):
        if idx < 0 or idx >= GeneralView._ZOOM_PRESETS.shape[0]:
            raise ValueError('idx must be in the range [0, {}).'.format(GeneralView._ZOOM_PRESETS.shape[0]))
        self._zoom_preset_idx = idx
        self._custom_zoom = 0
        if self.zoom_to_fit:
            self._ignore_zoom_to_fit_action_toggle = True
            self.zoom_to_fit_action.setChecked(False)
            self._ignore_zoom_to_fit_action_toggle = False
        self._apply_zoom()

    def _apply_zoom(self):
        if self.zoom_to_fit:
            self.fitInView(self.scene().layer_stack_item, Qt.Qt.KeepAspectRatio)
            current_zoom = self.transform().m22()
            if current_zoom != self._custom_zoom:
                self._custom_zoom = current_zoom
                self._zoom_preset_idx = -1
                self.zoom_changed.emit(self._zoom_preset_idx, self._custom_zoom)
        else:
            zoom_factor = self._custom_zoom if self._zoom_preset_idx == -1 else GeneralView._ZOOM_PRESETS[self._zoom_preset_idx]
            old_transform = Qt.QTransform(self.transform())
            self.resetTransform()
            self.translate(old_transform.dx(), old_transform.dy())
            self.scale(zoom_factor, zoom_factor)
            self.zoom_changed.emit(self._zoom_preset_idx, self._custom_zoom)
        self.scene_region_changed.emit(self)
