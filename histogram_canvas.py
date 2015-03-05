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
import sys

class ScalarPropWidgets:
    def __init__(self, label, slider, edit, edit_validator):
        self.label = label
        self.slider = slider
        self.edit = edit
        self.edit_validator = edit_validator

class ScalarProp:
    SLIDER_RAW_RANGE = (0, 1.0e9)
    SLIDER_RAW_RANGE_WIDTH = SLIDER_RAW_RANGE[1] - SLIDER_RAW_RANGE[0]
    next_grid_row = 0

    def __init__(self, scalar_props, name, name_in_label=None, channel_name=None):
        self.name = name
        self.name_in_label = name_in_label
        self.channel_name = channel_name
        scalar_props.append(self)
        self.widgets = {}
        self.grid_row = ScalarProp.next_grid_row
        ScalarProp.next_grid_row += 1
        self.values = {}

    def __get__(self, histogram_view, objtype=None):
        if histogram_view is None:
            return self
        return self.values[histogram_view]

    def __set__(self, histogram_view, value):
        if histogram_view is None:
            raise AttributeError("Can't set instance attribute of class.")
        if value is None:
            raise ValueError('None is not a valid {} value.'.format(self.name))
        range_ = self._get_range(histogram_view)
        if value < range_[0] or value > range_[1]:
            raise ValueError('Value supplied for {} must be in the range [{}, {}].'.format(self.name, range_[0], range_[1]))
        widgets = self.widgets[histogram_view]
        self.values[histogram_view] = value
        widgets.slider.setValue(self._value_to_slider_raw(value, histogram_view))

    def instantiate(self, histogram_view, layout):
        label_str = '' if self.channel_name is None else self.channel_name.title() + ' '
        if self.name_in_label is None:
            label_str += self.name if label_str else self.name.title()
        else:
            label_str += self.name_in_label
        label_str += ':'
        label = Qt.QLabel(label_str)
        slider = Qt.QSlider(Qt.Qt.Horizontal)
        label.setBuddy(slider)
        edit = Qt.QLineEdit()
        edit_validator = Qt.QDoubleValidator(histogram_view)
        edit.setValidator(edit_validator)
        layout.addWidget(label, self.grid_row, 0, Qt.Qt.AlignRight)
        layout.addWidget(slider, self.grid_row, 1)
        layout.addWidget(edit, self.grid_row, 2)
        self.widgets[histogram_view] = ScalarPropWidgets(label, slider, edit, edit_validator)
        if self.channel_name is not None:
            histogram_view._channel_control_widgets += [label, slider, edit]
        self.values[histogram_view] = None
        range_ = self._get_range(histogram_view)
        edit_validator = Qt.QDoubleValidator(range_[0], range_[1], 6, histogram_view)
        slider.setRange(*self.SLIDER_RAW_RANGE)
        slider.valueChanged.connect(lambda raw: self._on_slider_value_changed(histogram_view, raw))
        edit.editingFinished.connect(lambda: self._on_edit_changed(histogram_view))

    def propagate_slider_value(self, histogram_view):
        widgets = self.widgets[histogram_view]
        value = self._slider_raw_to_value(widgets.slider.value(), histogram_view)
        widgets.edit.setText('{:.6}'.format(value))
        self.values[histogram_view] = value

    def _slider_raw_to_value(self, raw, histogram_view):
        raise NotImplementedError()

    def _value_to_slider_raw(self, value, histogram_view):
        raise NotImplementedError()

    def _get_range(self, histogram_view):
        raise NotImplementedError()

    def _on_slider_value_changed(self, histogram_view, raw):
        value = self._slider_raw_to_value(raw, histogram_view)
        self.values[histogram_view] = value
        self._on_value_changed(histogram_view, value)

    def _on_edit_changed(self, histogram_view):
        widgets = self.widgets[histogram_view]
        try:
            value = float(widgets.edit.text())
        except ValueError:
            return
        widgets.slider.setValue(self._value_to_slider_raw(value, histogram_view))

    def _on_value_changed(self, histogram_view, value):
        pass

class GammaProp(ScalarProp):
    EXP2_RANGE = (-4, 2)
    EXP2_RANGE_WIDTH = EXP2_RANGE[1] - EXP2_RANGE[0]
    RANGE = tuple(map(lambda x:2**x, EXP2_RANGE))

    def _slider_raw_to_value(self, raw, histogram_view):
        value = float(raw)
        # Transform raw integer into linear floating point range (with gamma being 2 to the power of the linear value)
        value -= GammaProp.SLIDER_RAW_RANGE[0]
        value /= GammaProp.SLIDER_RAW_RANGE_WIDTH
        value *= GammaProp.EXP2_RANGE_WIDTH
        value += GammaProp.EXP2_RANGE[0]
        # Transform to logarithmic scale
        return 2**value

    def _value_to_slider_raw(self, value, histogram_view):
        # Transform value into linear floating point range
        raw = math.log2(value)
        # Transform float into raw integer range
        raw -= GammaProp.EXP2_RANGE[0]
        raw /= GammaProp.EXP2_RANGE_WIDTH
        raw *= GammaProp.SLIDER_RAW_RANGE_WIDTH
        raw += GammaProp.SLIDER_RAW_RANGE[0]
        return int(raw)

    def _get_range(self, histogram_view):
        return GammaProp.RANGE

    def _on_value_changed(self, histogram_view, value):
        widgets = self.widgets[histogram_view]
        widgets.edit.setText('{:.6}'.format(value))
        if histogram_view._image is not None:
            if self.name == 'gamma_gamma':
                # Refresh the histogram when gamma scale (ie gamma gamma) changes
                histogram_view.update()
            elif self.channel_name is None:
                if histogram_view._image.is_grayscale:
                    histogram_view.gamma_or_min_max_changed.emit()
                else:
                    histogram_view.gamma_red = value
                    histogram_view.gamma_green = value
                    histogram_view.gamma_blue = value
            elif not histogram_view._image.is_grayscale:
                histogram_view.gamma_or_min_max_changed.emit()

class MinMaxProp(ScalarProp):
    def __init__(self, scalar_props, min_max_props, name, name_in_label=None, channel_name=None):
        super().__init__(scalar_props, name, name_in_label, channel_name)
        attr_name = name
        if channel_name is not None:
            attr_name += '_' + channel_name
        min_max_props[attr_name] = self

    def instantiate(self, histogram_view, layout):
        super().instantiate(histogram_view, layout)
        if self.name == 'min':
            self.widgets[histogram_view].slider.setInvertedAppearance(True)

    def _slider_raw_to_value(self, raw, histogram_view):
        range_ = self._get_range(histogram_view)
        value = float(raw)
        value -= MinMaxProp.SLIDER_RAW_RANGE[0]
        value /= MinMaxProp.SLIDER_RAW_RANGE_WIDTH
        if self.name == 'min':
            value = 1 - value
        value *= range_[1] - range_[0]
        value += range_[0]
        return value

    def _value_to_slider_raw(self, value, histogram_view):
        range_ = self._get_range(histogram_view)
        raw = value - range_[0]
        raw /= range_[1] - range_[0]
        if self.name == 'min':
            raw = 1 - raw
        raw *= MinMaxProp.SLIDER_RAW_RANGE_WIDTH
        raw += MinMaxProp.SLIDER_RAW_RANGE[0]
        return int(raw)

    def _get_range(self, histogram_view):
        if histogram_view._image is None:
            return (0, 1)
        else:
            return histogram_view._image.range

    def _on_value_changed(self, histogram_view, value):
        if not histogram_view.allow_inversion:
            is_min = self.name == 'min'
            anti_attr = 'max' if is_min else 'min'
            if self.channel_name is not None:
                anti_attr += '_' + self.channel_name
            anti_val = getattr(histogram_view, anti_attr)
            if is_min and value > anti_val or not is_min and value < anti_val:
                setattr(histogram_view, anti_attr, value)
        widgets = self.widgets[histogram_view]
        widgets.edit.setText('{:.6}'.format(value))
        if histogram_view._image is not None:
            if self.channel_name is None:
                if histogram_view._image.is_grayscale:
                    histogram_view.gamma_or_min_max_changed.emit()
                else:
                    if self.name == 'min':
                        histogram_view.min_red = value
                        histogram_view.min_green = value
                        histogram_view.min_blue = value
                    else:
                        histogram_view.max_red = value
                        histogram_view.max_green = value
                        histogram_view.max_blue = value
            elif not histogram_view._image.is_grayscale:
                histogram_view.gamma_or_min_max_changed.emit()

class HistogramView(canvas.CanvasView):
    """Unlike ImageScene & ImageView, HistogramScene and HistogramView use the Qt local widget coordinate
    system; HistogramView's transformation matrix is always identity, whereas ImageView's transformation
    matrix is modified to project the image at user-controlled zoom and offset.

    This means that two HistogramViews should not share the same HistogramScene unless both HistogramViews
    are always the same size."""
    gamma_or_min_max_changed = Qt.pyqtSignal()

    _scalar_props = []
    _min_max_props = {}

    gamma_gamma = GammaProp(_scalar_props, 'gamma_gamma', '\u03b3\u03b3')
    gamma = GammaProp(_scalar_props, 'gamma', '\u03b3')
    gamma_red = GammaProp(_scalar_props, 'gamma', '\u03b3', 'red')
    gamma_green = GammaProp(_scalar_props, 'gamma', '\u03b3', 'green')
    gamma_blue = GammaProp(_scalar_props, 'gamma', '\u03b3', 'blue')
    max = MinMaxProp(_scalar_props, _min_max_props, 'max')
    min = MinMaxProp(_scalar_props, _min_max_props, 'min')
    max_red = MinMaxProp(_scalar_props, _min_max_props, 'max', channel_name='red')
    min_red = MinMaxProp(_scalar_props, _min_max_props, 'min', channel_name='red')
    max_green = MinMaxProp(_scalar_props, _min_max_props, 'max', channel_name='green')
    min_green = MinMaxProp(_scalar_props, _min_max_props, 'min', channel_name='green')
    max_blue = MinMaxProp(_scalar_props, _min_max_props, 'max', channel_name='blue')
    min_blue = MinMaxProp(_scalar_props, _min_max_props, 'min', channel_name='blue')

    @classmethod
    def make_histogram_and_container_widgets(cls, canvas_scene, parent):
        container = Qt.QWidget(parent)
        container.setLayout(Qt.QHBoxLayout())
        splitter = Qt.QSplitter()
        container.layout().addWidget(splitter)
        histogram_frame = Qt.QFrame(splitter)
        histogram_frame.setMinimumSize(Qt.QSize(120, 60))
        histogram_frame.setFrameShape(Qt.QFrame.StyledPanel)
        histogram_frame.setFrameShadow(Qt.QFrame.Sunken)
        histogram_frame.setLayout(Qt.QHBoxLayout())
        histogram_frame.layout().setSpacing(0)
        histogram_frame.layout().setContentsMargins(Qt.QMargins(0,0,0,0))
        histogram = cls(canvas_scene, parent)
        histogram_frame.layout().addWidget(histogram)
        splitter.addWidget(histogram._control_widgets_pane)
        splitter.addWidget(histogram_frame)
        histogram.channel_control_widgets_visible = False
        return (histogram, container)

    def __init__(self, canvas_scene, parent):
        super().__init__(canvas_scene, parent)
        self._image = None
        self._tex = None
        self._make_control_widgets_pane()

    def _make_control_widgets_pane(self):
        self._control_widgets_pane = Qt.QWidget(self)
        layout = Qt.QGridLayout()
        self._control_widgets_pane.setLayout(layout)
        self._channel_control_widgets = []
        self._channel_control_widgets_visible = True
        self._allow_inversion = True # Set to True during initialization for convenience...
        for scalar_prop in HistogramView._scalar_props:
            scalar_prop.instantiate(self, layout)
        self.gamma_gamma = 1
        self.gamma = 1
        self.gamma_red = 1
        self.gamma_green = 1
        self.gamma_blue = 1
        self.min = 0
        self.max = 1
        self.min_red = 0
        self.max_red = 1
        self.min_green = 0
        self.max_green = 1
        self.min_blue = 0
        self.max_blue = 1
        hlayout = Qt.QHBoxLayout()
        layout.addLayout(hlayout, ScalarProp.next_grid_row, 0, 1, -1)
        self._rescale_checkbox = Qt.QCheckBox('Rescale image')
        self._rescale_checkbox.setTristate(False)
        self._rescale_checkbox.setChecked(True)
        self._rescale_enabled = True
        self._rescale_checkbox.toggled.connect(self._on_rescale_checkbox_toggled)
        hlayout.addWidget(self._rescale_checkbox)
        self._allow_inversion_checkbox = Qt.QCheckBox('Allow inversion')
        self._allow_inversion_checkbox.setTristate(False)
        self._allow_inversion_checkbox.setChecked(False)
        self._allow_inversion_checkbox.toggled.connect(self._on_allow_inversion_checkbox_toggled)
        self._allow_inversion = False # ... and, enough stuff has been initialized that this can now be set to False without trouble
        hlayout.addWidget(self._allow_inversion_checkbox)
        hlayout.addItem(Qt.QSpacerItem(0, 0, Qt.QSizePolicy.MinimumExpanding, Qt.QSizePolicy.MinimumExpanding))
        self._mouseover_info_label = Qt.QLabel()
        hlayout.addWidget(self._mouseover_info_label)

    def resizeEvent(self, event):
        scene = self.scene()
        size = self.viewport().size()
        scene.histogram_item.set_bounding_rect(Qt.QRectF(0, 0, size.width(), size.height()))

    def _on_image_changed(self, image):
        if image is None or image.is_grayscale:
            self.channel_control_widgets_visible = False
        else:
            self.channel_control_widgets_visible = True
        range_changed = (self._image is None or image is None) or self._image.range != image.range
        self._image = image
        if range_changed:
            if image is None or image.is_grayscale:
                self._min_max_props['max'].propagate_slider_value(self)
                self._min_max_props['min'].propagate_slider_value(self)
            else:
                for min_max_prop in self._min_max_props.values():
                    min_max_prop.propagate_slider_value(self)
        self._correct_inversion()

        self.scene().histogram_item.on_image_changed(image)

    def _on_rescale_checkbox_toggled(self, checked):
        self.rescale_enabled = checked

    def _on_allow_inversion_checkbox_toggled(self, checked):
        self.allow_inversion = checked

    def _on_request_mouseover_info_status_text_change(self, txt):
        if sys.platform == 'darwin' and txt is None: # Workaround for gui cheese bug
            txt = '                          '
        self._mouseover_info_label.setText(txt)

    def _correct_inversion(self):
        if not self._allow_inversion:
            if self.max < self.min:
                self.max = self.min
            if self._image is not None and not self._image.is_grayscale:
                if self.max_red < self.min_red:
                    self.max_red = self.min_red
                if self.max_green < self.min_green:
                    self.max_green = self.min_green
                if self.max_blue < self.min_blue:
                    self.max_blue = self.min_blue

    @property
    def channel_control_widgets_visible(self):
        return self._channel_control_widgets_visible

    @channel_control_widgets_visible.setter
    def channel_control_widgets_visible(self, visible):
        if visible != self._channel_control_widgets_visible:
            self._channel_control_widgets_visible = visible
            for widget in self._channel_control_widgets:
                widget.setVisible(visible)

    @property
    def rescale_enabled(self):
        return self._rescale_enabled

    @rescale_enabled.setter
    def rescale_enabled(self, rescale_enabled):
        if self._rescale_enabled != rescale_enabled:
            self._rescale_enabled = rescale_enabled
            if self._rescale_checkbox.isChecked() != rescale_enabled:
                self._rescale_checkbox.setChecked(rescale_enabled)
            if self._image is not None:
                self.gamma_or_min_max_changed.emit()

    @property
    def allow_inversion(self):
        return self._allow_inversion

    @allow_inversion.setter
    def allow_inversion(self, allow_inversion):
        if self._allow_inversion != allow_inversion:
            self._allow_inversion = allow_inversion
            if self._allow_inversion_checkbox.isChecked() != allow_inversion:
                self._allow_inversion_checkbox.setChecked(allow_inversion)
            self._correct_inversion()

    @property
    def histogram(self):
        if self._image is not None:
            return self._image.histogram.copy()

class HistogramItem(canvas.CanvasGLItem):
    def __init__(self, graphics_item_parent=None):
        super().__init__(graphics_item_parent)
        self._image = None
        self._image_id = 0
        self._bounding_rect = Qt.QRectF()

    def boundingRect(self):
        return Qt.QRectF() if self._image is None else self._bounding_rect

    def set_bounding_rect(self, rect):
        if self._image is not None:
            self.prepareGeometryChange()
        self._bounding_rect = rect

    def paint(self, qpainter, option, widget):
        if widget is None:
            print('WARNING: histogram_view.HistogramItem.paint called with widget=None.  Ensure that view caching is disabled.')
        elif self._image is None:
            if widget.view in self._view_resources:
                self._del_tex()
        else:
            image = self._image
            view = widget.view
            gl = view.glfs
            with ExitStack() as stack:
                qpainter.beginNativePainting()
                stack.callback(qpainter.endNativePainting)
                if view in self._view_resources:
                    vrs = self._view_resources[view]
                else:
                    self._view_resources[view] = vrs = {}
                    self.build_shader_prog('g',
                                           'histogram_widget_vertex_shader.glsl',
                                           'histogram_widget_fragment_shader_g.glsl',
                                           view)
                    vrs['progs']['ga'] = vrs['progs']['g']
                    self.build_shader_prog('rgb',
                                           'histogram_widget_vertex_shader.glsl',
                                           'histogram_widget_fragment_shader_rgb.glsl',
                                           view)
                    vrs['progs']['rgba'] = vrs['progs']['rgb']
                desired_tex_width = image.histogram.shape[-1]
                if 'tex' in vrs:
                    tex, tex_image_id, tex_width = vrs['tex']
                    if desired_tex_width != tex_width:
                        self._del_tex()
                if 'tex' not in vrs:
                    tex = gl.glGenTextures(1)
                    gl.glBindTexture(gl.GL_TEXTURE_1D, tex)
                    stack.callback(lambda: gl.glBindTexture(gl.GL_TEXTURE_1D, 0))
                    gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
                    gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
                    # tex stores histogram bin counts - values that are intended to be addressed by element without
                    # interpolation.  Thus, nearest neighbor for texture filtering.
                    gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
                    gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
                    tex_image_id = -1
                else:
                    gl.glBindTexture(gl.GL_TEXTURE_1D, tex)
                    stack.callback(lambda: gl.glBindTexture(gl.GL_TEXTURE_1D, 0))
                if image.is_grayscale:
                    if image.type == 'g':
                        histogram = image.histogram
                        max_bin_val = histogram[image.max_histogram_bin]
                    else:
                        histogram = image.histogram[0]
                        max_bin_val = histogram[image.max_histogram_bin[0]]
                    if tex_image_id != self._image_id:
                        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
                        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
                        gl.glTexImage1D(gl.GL_TEXTURE_1D, 0,
                                        gl.GL_LUMINANCE32UI_EXT, desired_tex_width, 0,
                                        gl.GL_LUMINANCE_INTEGER_EXT, gl.GL_UNSIGNED_INT,
                                        histogram.data)
                        vrs['tex'] = tex, self._image_id, desired_tex_width
                    view.quad_buffer.bind()
                    stack.callback(view.quad_buffer.release)
                    view.quad_vao.bind()
                    stack.callback(view.quad_vao.release)
                    prog = vrs['progs'][image.type]
                    prog.bind()
                    stack.callback(prog.release)
                    vert_coord_loc = prog.attributeLocation('vert_coord')
                    prog.enableAttributeArray(vert_coord_loc)
                    prog.setAttributeBuffer(vert_coord_loc, gl.GL_FLOAT, 0, 2, 0)
                    prog.setUniformValue('tex', 0)
                    prog.setUniformValue('inv_view_size', 1/widget.size().width(), 1/widget.size().height())
                    inv_max_transformed_bin_val = max_bin_val**-view.gamma_gamma
                    prog.setUniformValue('inv_max_transformed_bin_val', inv_max_transformed_bin_val)
                    prog.setUniformValue('gamma_gamma', view.gamma_gamma)
                    prog.setUniformValue('rescale_enabled', view.rescale_enabled)
                    if view.rescale_enabled:
                        prog.setUniformValue('gamma', view.gamma)
                        min_max = numpy.array((view.min, view.max), dtype=float)
                        self._normalize_min_max(min_max)
                        prog.setUniformValue('intensity_rescale_min', min_max[0])
                        prog.setUniformValue('intensity_rescale_range', min_max[1] - min_max[0])
                    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
                    gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)
                else:
                    pass
                    # personal time todo: per-channel RGB histogram support

    def on_image_changed(self, image):
        if (self._image is None) != (image is not None) or \
           self._image is not None and image is not None and self._image.histogram.shape[-1] != image.histogram.shape[-1]:
            self.prepareGeometryChange()
        self._image = image
        self._image_id += 1
        self.update()

    def release_resources_for_view(self, canvas_view):
        if canvas_view in self._view_resources:
            if 'tex' in self._view_resources[canvas_view]:
                self._del_tex()
        super().release_resources_for_view(canvas_view)

    def _del_tex(self):
        vrs = self._view_resources[widget.view]
        self.widget.view.glfs.glDeleteTextures(1, (vrs['tex'][0],))
        del vrs['tex']

class GammaPlotItem(Qt.QGraphicsItem):
    pass

class HistogramScene(canvas.CanvasScene):
    def __init__(self, parent):
        super().__init__(parent)
        self.histogram_item = HistogramItem()
        self.addItem(self.histogram_item)
