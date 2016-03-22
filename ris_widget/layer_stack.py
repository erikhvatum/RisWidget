# The MIT License (MIT)
#
# Copyright (c) 2015 WUSTL ZPLAB
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

import json
from PyQt5 import Qt
import numpy
import textwrap
from . import om
from .image import Image
from .layer import Layer

class LayerList(om.UniformSignalingList):
    @classmethod
    def from_json(cls, json_str, show_error_messagebox=False):
        try:
            prop_stackd = json.loads(json_str)['layer property stack']
            layers = cls()
            for props in prop_stackd:
                layers.append(Layer.from_savable_properties_dict(props))
            return layers
        except (FileNotFoundError, ValueError, TypeError) as e:
            if show_error_messagebox:
                Qt.QMessageBox.information(None, 'JSON Error', '{} : {}'.format(type(e).__name__, e))

    def to_json(self):
        return json.dumps(
            {
                'layer property stack' :
                [
                    layer.get_savable_properties_dict() for layer in self
                ]
            },
            ensure_ascii=False, indent=1
        )

    def take_input_element(self, obj):
        if isinstance(obj, (numpy.ndarray, Image)):
            obj = Layer(obj)
        elif not isinstance(obj, Layer):
            raise TypeError("All inputs must be numpy.ndarray, Image, or Layer")
        return obj

class LayerStack(Qt.QObject):
    """LayerStack: The owner of a LayerList (L.layers, in ascending order, with bottom layer - ie, backmost - as element 0) and selection model that is
    referenced by various other objects such as LayerStackItem, LayerTable, RisWidget, and Flipbooks.  LayerStack emits the Qt signal layers_replaced
    when .layers is replaced by assignment (L.layers = [...]), forwards changing & changed signals from its LayerList, and ensures that a Layer is
    focused and selected according to the selection model whenever its LayerList is not empty.

    It is safe to assign None to either or both of LayerStack instance's layers and selection_model properties.

    In ascending order, with bottom layer (backmost) as element 0.

    Signals:
    * layers_replaced(layer_stack, old_layers, layers): layer_stack.layers changed from old_layers to layers, its current value.
    * selection_model_replaced(layer_stack, old_sm, sm): layer_stack.selection_model changed from old_sm to sm, its current value.  LayerStack provides
    the layer_focus_changed signal, which is a proxy for the layer_stack.selection_model.currentRowChanged signal - the most commonly used selection
    model signal.  If this is the only selection model signal you need, you can avoid having to connect to the new selection model's currentRowChanged
    signal in response to selection_model_replaced by using LayerStack's layer_focus_changed signal instead.
    * layer_focus_changed(layer_stack, old_focused_layer, focused_layer): layer_stack.focused_layer changed from old_focused layer to focused_layer,
    its current value."""
    layers_replaced = Qt.pyqtSignal(Qt.QObject, object, object)
    selection_model_replaced = Qt.pyqtSignal(Qt.QObject, object, object)
    layer_focus_changed = Qt.pyqtSignal(Qt.QObject, object, object)

    def __init__(self, layers=None, selection_model=None, parent=None):
        super().__init__(parent)
        self._layers = None
        self.layers = layers
        self._selection_model = None
        self.selection_model = selection_model
        self._layer_instance_counts = {}
        self.layer_name_in_contextual_info_action = Qt.QAction(self)
        self.layer_name_in_contextual_info_action.setText('Include Layer.name in Contextual Info')
        self.layer_name_in_contextual_info_action.setCheckable(True)
        self.layer_name_in_contextual_info_action.setChecked(False)
        self.image_name_in_contextual_info_action = Qt.QAction(self)
        self.image_name_in_contextual_info_action.setText('Include Image.name in Contextual Info')
        self.image_name_in_contextual_info_action.setCheckable(True)
        self.image_name_in_contextual_info_action.setChecked(False)
        self.master_enable_auto_min_max_action = Qt.QAction(self)
        self.master_enable_auto_min_max_action.setText('Auto Min/Max Master On')
        self.master_enable_auto_min_max_action.setCheckable(True)
        self.master_enable_auto_min_max_action.setChecked(False)
        self.master_enable_auto_min_max_action.toggled.connect(self._on_master_enable_auto_min_max_toggled)
        self.examine_layer_mode_action = Qt.QAction(self)
        self.examine_layer_mode_action.setText('Examine Current Layer')
        self.examine_layer_mode_action.setCheckable(True)
        self.examine_layer_mode_action.setChecked(False)
        self.examine_layer_mode_action.setToolTip(textwrap.dedent("""\
                In "Examine Layer Mode", a layer's .visible property does not control whether that
                layer is visible in the main view.  Instead, the layer represented by the row currently
                selected in the layer table is treated as if the value of its .visible property were
                True and all others as if theirs were false."""))
        self.histogram_alternate_column_shading_action = Qt.QAction(self)
        self.histogram_alternate_column_shading_action.setText('Alternate Histogram Bin Shading')
        self.histogram_alternate_column_shading_action.setCheckable(True)
        self.histogram_alternate_column_shading_action.setChecked(False)

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, v):
        v_o = self._layers
        if v is v_o:
            return
        if not (v is None or isinstance(v, LayerList)):
            v = LayerList(v)
        if v_o is not None:
            v_o.inserted.disconnect(self._on_inserted_into_layers)
            v_o.removed.disconnect(self._on_removed_from_layers)
            v_o.replaced.disconnect(self._on_replaced_in_layers)
            v_o.inserted.disconnect(self._delayed_on_inserted_into_layers)
            v_o.removed.disconnect(self._delayed_on_removed_from_layers)
        self._layers = v
        if v is not None:
            v.inserted.connect(self._on_inserted_into_layers)
            v.removed.connect(self._on_removed_from_layers)
            v.replaced.connect(self._on_replaced_in_layers)
            # Must be QueuedConnection in order to avoid race condition where self._on_inserted_into_layers may be called before any associated model's
            # "inserted" handler, which would cause ensure_layer_focused, if layers was empty before insertion, to attempt to focus row 0 before associated
            # models are even aware that a row has been inserted.
            v.inserted.connect(self._delayed_on_inserted_into_layers, Qt.Qt.QueuedConnection)
            v.removed.connect(self._delayed_on_removed_from_layers, Qt.Qt.QueuedConnection)
        self.layers_replaced.emit(self, v_o, v)
        if v:
            self.ensure_layer_focused()

    def get_layers(self):
        if self._layers is None:
            self.layers = LayerList()
        return self._layers

    @property
    def selection_model(self):
        return self._selection_model

    @selection_model.setter
    def selection_model(self, v):
        assert v is None or isinstance(v, Qt.QItemSelectionModel)
        v_o = self._selection_model
        if v is v_o:
            return
        if v_o is not None:
            v_o.currentRowChanged.disconnect(self._on_current_row_changed)
        if v is not None:
            v.currentRowChanged.connect(self._on_current_row_changed)
        self._selection_model = v
        self.selection_model_replaced.emit(self, v_o, v)

    @property
    def focused_layer_idx(self):
        sm = self._selection_model
        if sm is None:
            return
        m = sm.model()
        if m is None:
            return
        midx = sm.currentIndex()
        if isinstance(m, Qt.QAbstractProxyModel):
            # Selection model is with reference to table view's model, which is a proxy model (probably an InvertingProxyModel)
            if not midx.isValid():
                return
            midx = m.mapToSource(midx)
        if midx.isValid():
            return midx.row()

    @property
    def focused_layer(self):
        """Note: L.focused_layer = Layer() is equivalent to L.layers[L.focused_layer_idx] = Layer()."""
        if self._layers is not None:
            idx = self.focused_layer_idx
            if idx is not None:
                return self._layers[idx]

    @focused_layer.setter
    def focused_layer(self, v):
        idx = self.focused_layer_idx
        if idx is None:
            raise IndexError('No layer is currently focused.')
        self._layers[idx] = v

    def ensure_layer_focused(self):
        """If we have both a layer list & selection model and no Layer is selected & .layers is not empty:
           If there is a "current" layer, IE highlighted but not selected, select it.
           If there is no "current" layer, make .layer_stack[0] current and select it."""
        ls = self._layers
        if not ls:
            return
        sm = self._selection_model
        if sm is None:
            return
        m = sm.model()
        if m is None:
            return
        if not sm.currentIndex().isValid():
            sm.setCurrentIndex(m.index(0, 0), Qt.QItemSelectionModel.SelectCurrent | Qt.QItemSelectionModel.Rows)
        if len(sm.selectedRows()) == 0:
            sm.select(sm.currentIndex(), Qt.QItemSelectionModel.SelectCurrent | Qt.QItemSelectionModel.Rows)

    @property
    def layer_name_in_contextual_info_enabled(self):
        return self.layer_name_in_contextual_info_action.isChecked()

    @layer_name_in_contextual_info_enabled.setter
    def layer_name_in_contextual_info_enabled(self, v):
        self.layer_name_in_contextual_info_action.setChecked(v)

    @property
    def image_name_in_contextual_info_enabled(self):
        return self.image_name_in_contextual_info_action.isChecked()

    @image_name_in_contextual_info_enabled.setter
    def image_name_in_contextual_info_enabled(self, v):
        self.image_name_in_contextual_info_action.setChecked(v)

    @property
    def examine_layer_mode_enabled(self):
        return self.examine_layer_mode_action.isChecked()

    @examine_layer_mode_enabled.setter
    def examine_layer_mode_enabled(self, v):
        self.examine_layer_mode_action.setChecked(v)

    @property
    def histogram_alternate_column_shading_enabled(self):
        return self.histogram_alternate_column_shading_action.isChecked()

    @histogram_alternate_column_shading_enabled.setter
    def histogram_alternate_column_shading_enabled(self, v):
        self.histogram_alternate_column_shading_action.setChecked(v)

    @property
    def master_enable_auto_min_max_is_active(self):
        return self.master_enable_auto_min_max_action.isChecked()

    @master_enable_auto_min_max_is_active.setter
    def master_enable_auto_min_max_is_active(self, v):
        self.examine_layer_mode_action.setChecked(v)

    def _attach_layers(self, layers):
        for layer in layers:
            instance_count = self._layer_instance_counts.get(layer, 0) + 1
            assert instance_count > 0
            self._layer_instance_counts[layer] = instance_count
            if instance_count == 1:
                layer.min_changed.connect(self._on_layer_min_or_max_changed)
                layer.max_changed.connect(self._on_layer_min_or_max_changed)

    def _detach_layers(self, layers):
        for layer in layers:
            instance_count = self._layer_instance_counts[layer] - 1
            assert instance_count >= 0
            if instance_count == 0:
                layer.min_changed.disconnect(self._on_layer_min_or_max_changed)
                layer.max_changed.disconnect(self._on_layer_min_or_max_changed)
                del self._layer_instance_counts[layer]
            else:
                self._layer_instance_counts[layer] = instance_count

    def _on_inserted_into_layers(self, idx, layers):
        self._attach_layers(layers)

    def _on_removed_from_layers(self, idxs, layers):
        self._detach_layers(layers)

    def _on_replaced_in_layers(self, idxs, replaced_layers, layers):
        # Note: the selection model may be associated with a proxy model, in which case this method's idxs argument is in terms of the proxy.  Therefore,
        # we can't use self.focused_layer_idx (if the selection model is attached to a proxy, self.focused_layer_idx is in terms of the proxied model,
        # not the proxy).
        #       self.ensure_layer_focused()
        self._detach_layers(replaced_layers)
        self._attach_layers(layers)
        sm = self._selection_model
        if sm is None:
            return
        focused_midx = sm.currentIndex()
        if focused_midx is None:
            return
        focused_row = focused_midx.row()
        try:
            change_idx = idxs.index(focused_row)
        except ValueError:
            return
        old_focused, focused = replaced_layers[change_idx], layers[change_idx]
        if old_focused is not focused:
            self.layer_focus_changed.emit(self, old_focused, focused)

    def _delayed_on_inserted_into_layers(self, idx, layers):
        self.ensure_layer_focused()

    def _delayed_on_removed_from_layers(self, idxs, layers):
        self.ensure_layer_focused()

    def _on_current_row_changed(self, midx, old_midx):
        # TODO: verify that this happens in response to signaling list removing signal and not removed signal
        sm = self._selection_model
        m = sm.model()
        ls = self._layers
        if isinstance(m, Qt.QAbstractProxyModel):
            if old_midx.isValid():
                old_midx = m.mapToSource(old_midx)
            if midx.isValid():
                midx = m.mapToSource(midx)
        ol = ls[old_midx.row()] if old_midx.isValid() else None
        l = ls[midx.row()] if midx.isValid() else None
        if l is not ol:
            self.layer_focus_changed.emit(self, ol, l)

    def _on_master_enable_auto_min_max_toggled(self, checked):
        if checked:
            for layer in self._layers:
                pass

    def _on_layer_min_or_max_changed(self, layer):
        pass