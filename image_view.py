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

from . import canvas
from contextlib import ExitStack
import math
import numpy
from PyQt5 import Qt

class ImageView(canvas.CanvasView):
    _ZOOM_PRESETS = numpy.array((10, 8, 7, 6, 5, 4, 3, 2, 1.5, 1, .75, .6666666, .5, .333333, .25, .1), dtype=numpy.float64)
    _ZOOM_MIN_MAX = (.001, 10000.0)
    _ZOOM_DEFAULT_PRESET_IDX = 9
    _ZOOM_INCREMENT_BEYOND_PRESETS_FACTOR = .25

    zoom_changed = Qt.pyqtSignal(int, float)
    zoom_to_fit_changed = Qt.pyqtSignal(bool)

    def __init__(self, shader_scene, parent):
        super().__init__(shader_scene, parent)
        self.histogram_scene = None
        self.setMinimumSize(Qt.QSize(100,100))
        self._zoom_preset_idx = self._ZOOM_DEFAULT_PRESET_IDX
        self._custom_zoom = 0
        self._zoom_to_fit = False
        self.setDragMode(Qt.QGraphicsView.ScrollHandDrag)

    def _on_image_changing(self, image):
        if self._zoom_to_fit:
            self.fitInView(self.scene().image_item, Qt.Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        if self._zoom_to_fit:
            self.fitInView(self.scene().image_item, Qt.Qt.KeepAspectRatio)
        else:
            super().resizeEvent(event)

    def wheelEvent(self, event):
        wheel_delta = event.angleDelta().y()
        if wheel_delta != 0 and not self._zoom_to_fit:
            zoom_in = wheel_delta > 0
            switched_to_custom = False
            if self._zoom_preset_idx != -1:
                if zoom_in:
                    if self._zoom_preset_idx == 0:
                        self._zoom_preset_idx = -1
                        self._custom_zoom = ImageView._ZOOM_PRESETS[0]
                        switched_to_custom = True
                    else:
                        self._zoom_preset_idx -= 1
                else:
                    if self._zoom_preset_idx == ImageView._ZOOM_PRESETS.shape[0] - 1:
                        self._zoom_preset_idx = -1
                        self._custom_zoom = ImageView._ZOOM_PRESETS[-1]
                        switched_to_custom = True
                    else:
                        self._zoom_preset_idx += 1
            if self._zoom_preset_idx == -1:
                self._custom_zoom *= (1 + ImageView._ZOOM_INCREMENT_BEYOND_PRESETS_FACTOR) if zoom_in else (1 - ImageView._ZOOM_INCREMENT_BEYOND_PRESETS_FACTOR)
                if not switched_to_custom and self._custom_zoom <= ImageView._ZOOM_PRESETS[0] and self._custom_zoom >= ImageView._ZOOM_PRESETS[-1]:
                    # Jump to nearest preset if we are re-entering preset range
                    self._zoom_preset_idx = numpy.argmin(numpy.abs(ImageView._ZOOM_PRESETS - self._custom_zoom))
                    self._custom_zoom = 0
            if self._zoom_preset_idx == -1:
                if zoom_in:
                    if self._custom_zoom > ImageView._ZOOM_MIN_MAX[1]:
                        self._custom_zoom = ImageView._ZOOM_MIN_MAX[1]
                else:
                    if self._custom_zoom < ImageView._ZOOM_MIN_MAX[0]:
                        self._custom_zoom = ImageView._ZOOM_MIN_MAX[0]
                desired_zoom = self._custom_zoom
            else:
                desired_zoom = self._ZOOM_PRESETS[self._zoom_preset_idx]
            # With transformationAnchor set to AnchorUnderMouse, QGraphicsView.scale modifies the view's transformation
            # matrix such that the same image pixel remains under the mouse cursor (except where the demands imposed by
            # centering of an undersized scene take priority).  But, that does mean we must modify the transformation
            # via the view's scale function and not by direct manipulation of the view's transformation matrix.  Thus,
            # it is necessary to find the current scaling in order to compute the factor by which scaling must be
            # mulitplied in order to arrive at our desired scaling.  This found by taking the view's current vertical
            # scaling under the assumption that a square in the scene will appear as a square in the view (and not a
            # rectangle or trapezoid), which holds if we are displaying images with square pixels - the only kind we
            # support.
            current_zoom = self.transform().m22()
            scale_zoom = desired_zoom / current_zoom
            self.setTransformationAnchor(Qt.QGraphicsView.AnchorUnderMouse)
            self.scale(scale_zoom, scale_zoom)
            self.setTransformationAnchor(Qt.QGraphicsView.AnchorViewCenter)
            self.zoom_changed.emit(self._zoom_preset_idx, self._custom_zoom)

    @property
    def zoom_to_fit(self):
        return self._zoom_to_fit

    @zoom_to_fit.setter
    def zoom_to_fit(self, zoom_to_fit):
        self._zoom_to_fit = zoom_to_fit
        self._zoom()

    @property
    def custom_zoom(self):
        return self._custom_zoom

    @custom_zoom.setter
    def custom_zoom(self, custom_zoom):
        if custom_zoom < ImageView._ZOOM_MIN_MAX[0] or custom_zoom > ImageView._ZOOM_MIN_MAX[1]:
            raise ValueError('Value must be in the range [{}, {}].'.format(*ImageView._ZOOM_MIN_MAX))
        self._custom_zoom = custom_zoom
        self._zoom_preset_idx = -1
        self._zoom()

    @property
    def zoom_preset_idx(self):
        return self._zoom_preset_idx

    @zoom_preset_idx.setter
    def zoom_preset_idx(self, idx):
        if idx < 0 or idx >= ImageView._ZOOM_PRESETS.shape[0]:
            raise ValueError('idx must be in the range [0, {}).'.format(ImageView._ZOOM_PRESETS.shape[0]))
        self._zoom_preset_idx = idx
        self._custom_zoom = 0
        self._zoom()

    def _zoom(self, zoom_to_fit_changed=False):
        if self._zoom_to_fit:
            self.fitInView(self.scene().image_item, Qt.Qt.KeepAspectRatio)
        else:
            zoom_factor = self._custom_zoom if self._zoom_preset_idx == -1 else ImageView._ZOOM_PRESETS[self._zoom_preset_idx]
            old_transform = Qt.QTransform(self.transform())
            self.resetTransform()
            self.translate(old_transform.dx(), old_transform.dy())
            self.scale(zoom_factor, zoom_factor)
        if zoom_to_fit_changed:
            self.zoom_to_fit_changed.emit(self._zoom_to_fit)
        else:
            self.zoom_changed.emit(self._zoom_preset_idx, self._custom_zoom)
