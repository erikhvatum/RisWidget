# The MIT License (MIT)
#
# Copyright (c) 2016 WUSTL ZPLAB
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
from ..shared_resources import UNIQUE_QGRAPHICSITEM_TYPE

class LayerStackPainterBrush:
    def __init__(self, content, mask, center=(0,0)):
        assert content.shape[:2] == mask.shape[:2]
        self.content = content
        self.mask = mask
        self.center = center

    def apply(self, target_subimage, brush_subrect):
        br = brush_subrect
        target_subimage[self.mask[br.left():br.right(),br.top():br.bottom()]] = self.content[br.left():br.right(),br.top():br.bottom()]

class LayerStackPainterItem(Qt.QGraphicsObject):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()
    target_layer_idx_changed = Qt.pyqtSignal(Qt.QObject)
    # Something relevant to LayerStackPainter changed: either we are now looking at a different Image
    # instance due to assignment to layer.image, or image data type and/or channel count and/or range
    # changed.  target_image_aspect_changed is not emitted when just image data changes.
    target_image_aspect_changed = Qt.pyqtSignal(Qt.QObject)

    def __init__(self, layer_stack_item, target_layer_idx=None):
        super().__init__(layer_stack_item)
        self.setFlag(Qt.QGraphicsItem.ItemHasNoContents)
        self._boundingRect = Qt.QRectF()
        self.layer_stack_item = layer_stack_item
        layer_stack_item.bounding_rect_changed.connect(self._on_layer_stack_item_bounding_rect_changed)
        self.layer_stack = layer_stack_item.layer_stack
        self.layers = None
        self._target_layer_idx = None
        self.target_layer = None
        self.target_image = None
        self.target_image_dtype = None
        self.target_image_shape = None
        self.target_image_range = None
        self.target_image_type = None
        self._on_layers_replaced(self.layer_stack, None, layer_stack_item.layer_stack.layers)
        self.target_layer_idx = target_layer_idx
        self.brush = None
        self.alternate_brush = None

    def boundingRect(self):
        return self._boundingRect

    def sceneEventFilter(self, watched, event):
        if (self.target_image is not None and
            watched is self.parentItem() and
            event.type() in (Qt.QEvent.GraphicsSceneMousePress, Qt.QEvent.GraphicsSceneMouseMove) and
            event.buttons() in (Qt.Qt.RightButton, Qt.Qt.MidButton)
        ):
            brush = self.brush if event.buttons() == Qt.Qt.RightButton else self.alternate_brush
            if brush is None:
                return False
            p = self.mapFromScene(event.scenePos())
            im = self.target_image
            c = brush.content
            r = Qt.QRect(p.x(), p.y(), c.shape[0], c.shape[1])
            r.translate(-brush.center[0], -brush.center[1])
            if not r.intersects(Qt.QRect(Qt.QPoint(), im.size)):
                return False
            br = Qt.QRect(0, 0, c.shape[0], c.shape[1])
            if r.left() < 0:
                br.setLeft(-r.x())
                r.setLeft(0)
            if r.top() < 0:
                br.setTop(-r.y())
                r.setTop(0)
            if r.right() >= im.size.width():
                br.setRight(br.right() - (r.right() - im.size.width() + 1))
                r.setRight(im.size.width() - 1)
            if r.bottom() >= im.size.height():
                br.setBottom(br.bottom() - (r.bottom() - im.size.height() + 1))
                r.setBottom(im.size.height() - 1)
            brush.apply(im.data[r.left():r.right(),r.top():r.bottom()], br)
            self.target_image.refresh(data_changed=True)
            return True
        return False

    @property
    def target_layer_idx(self):
        """It is permissible to assign a negative value to .target_layer_idx; a negative value selects
        nth-from-end, following the common Python idiom.  Assigning an out of range value is equivalent
        to assigning None, except that the painter will activate if/when the specified index does
        come into range.  Assigning None deactivates the painter."""
        return self._target_layer_idx

    @target_layer_idx.setter
    def target_layer_idx(self, v):
        if v is not None:
            v = int(v)
        if v != self._target_layer_idx:
            self._target_layer_idx = v
            self._on_layer_changed()
            self.target_layer_idx_changed.emit(self)

    def _on_layer_stack_item_bounding_rect_changed(self):
        self.prepareGeometryChange()
        self._boundingRect = self.layer_stack_item.boundingRect()

    def _on_layers_replaced(self, layer_stack, old_layers, layers):
        assert layer_stack is self.layer_stack and (self.layers is None or self.layers is old_layers)
        if old_layers is not None:
            old_layers.inserted.disconnect(self._on_layer_changed)
            old_layers.removed.disconnect(self._on_layer_changed)
            old_layers.replaced.disconnect(self._on_layer_changed)
        self.layers = layers
        if layers is not None:
            layers.inserted.connect(self._on_layer_changed)
            layers.removed.connect(self._on_layer_changed)
            layers.replaced.connect(self._on_layer_changed)
        self._on_layer_changed()

    def _resolve_target_layer(self):
        if self.layers is None:
            return
        try:
            return self.layers[self._target_layer_idx]
        except IndexError:
            pass

    def _on_layer_changed(self):
        target_layer = self._resolve_target_layer()
        if target_layer is self.target_layer:
            return
        if self.target_layer is not None:
            self.target_layer.image_changed.disconnect(self._on_image_changed)
        self.target_layer = target_layer
        if target_layer is not None:
            target_layer.image_changed.connect(self._on_image_changed)
        self._on_image_changed()

    def _on_image_changed(self):
        if self.target_image is not None and (self.target_layer is None or self.target_layer.image is None):
            self.target_image = None
            self.target_image_dtype = None
            self.target_image_shape = None
            self.target_image_range = None
            self.target_image_type = None
            self.setVisible(False)
            self.target_image_aspect_changed.emit(self)
            return
        ti = self.target_image
        if self.target_layer.image is not ti or (
            self.target_image_dtype != ti.dtype or
            self.target_image_shape != ti.shape or
            self.target_image_range != ti.range or
            self.target_image_type != ti.type
        ):
            ti = self.target_image = self.target_layer.image
            self.target_image_dtype = ti.dtype
            self.target_image_shape = ti.shape
            self.target_image_range = ti.range
            self.target_image_type = ti.type
            self.setVisible(True)
            self.target_image_aspect_changed.emit(self)