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

from . import canvas
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

    def __get__(self, histogram_widget, objtype=None):
        if histogram_widget is None:
            return self
        return self.values[histogram_widget]

    def __set__(self, histogram_widget, value):
        if histogram_widget is None:
            raise AttributeError("Can't set instance attribute of class.")
        if value is None:
            raise ValueError('None is not a valid {} value.'.format(self.name))
        range_ = self._get_range(histogram_widget)
        if value < range_[0] or value > range_[1]:
            raise ValueError('Value supplied for {} must be in the range [{}, {}].'.format(self.name, range_[0], range_[1]))
        widgets = self.widgets[histogram_widget]
        self.values[histogram_widget] = value
        widgets.slider.setValue(self._value_to_slider_raw(value, histogram_widget))

    def instantiate(self, histogram_widget, layout):
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
        edit_validator = Qt.QDoubleValidator(histogram_widget)
        edit.setValidator(edit_validator)
        layout.addWidget(label, self.grid_row, 0, Qt.Qt.AlignRight)
        layout.addWidget(slider, self.grid_row, 1)
        layout.addWidget(edit, self.grid_row, 2)
        self.widgets[histogram_widget] = ScalarPropWidgets(label, slider, edit, edit_validator)
        if self.channel_name is not None:
            histogram_widget._channel_control_widgets += [label, slider, edit]
        self.values[histogram_widget] = None
        range_ = self._get_range(histogram_widget)
        edit_validator = Qt.QDoubleValidator(range_[0], range_[1], 6, histogram_widget)
        slider.setRange(*self.SLIDER_RAW_RANGE)
        slider.valueChanged.connect(lambda raw: self._on_slider_value_changed(histogram_widget, raw))
        edit.editingFinished.connect(lambda: self._on_edit_changed(histogram_widget))

    def propagate_slider_value(self, histogram_widget):
        widgets = self.widgets[histogram_widget]
        value = self._slider_raw_to_value(widgets.slider.value(), histogram_widget)
        widgets.edit.setText('{:.6}'.format(value))
        self.values[histogram_widget] = value

    def _slider_raw_to_value(self, raw, histogram_widget):
        raise NotImplementedError()

    def _value_to_slider_raw(self, value, histogram_widget):
        raise NotImplementedError()

    def _get_range(self, histogram_widget):
        raise NotImplementedError()

    def _on_slider_value_changed(self, histogram_widget, raw):
        value = self._slider_raw_to_value(raw, histogram_widget)
        self.values[histogram_widget] = value
        self._on_value_changed(histogram_widget, value)

    def _on_edit_changed(self, histogram_widget):
        widgets = self.widgets[histogram_widget]
        try:
            value = float(widgets.edit.text())
        except ValueError:
            return
        widgets.slider.setValue(self._value_to_slider_raw(value, histogram_widget))

    def _on_value_changed(self, histogram_widget, value):
        pass

class GammaProp(ScalarProp):
    EXP2_RANGE = (-4, 2)
    EXP2_RANGE_WIDTH = EXP2_RANGE[1] - EXP2_RANGE[0]
    RANGE = tuple(map(lambda x:2**x, EXP2_RANGE))

    def _slider_raw_to_value(self, raw, histogram_widget):
        value = float(raw)
        # Transform raw integer into linear floating point range (with gamma being 2 to the power of the linear value)
        value -= GammaProp.SLIDER_RAW_RANGE[0]
        value /= GammaProp.SLIDER_RAW_RANGE_WIDTH
        value *= GammaProp.EXP2_RANGE_WIDTH
        value += GammaProp.EXP2_RANGE[0]
        # Transform to logarithmic scale
        return 2**value

    def _value_to_slider_raw(self, value, histogram_widget):
        # Transform value into linear floating point range
        raw = math.log2(value)
        # Transform float into raw integer range
        raw -= GammaProp.EXP2_RANGE[0]
        raw /= GammaProp.EXP2_RANGE_WIDTH
        raw *= GammaProp.SLIDER_RAW_RANGE_WIDTH
        raw += GammaProp.SLIDER_RAW_RANGE[0]
        return int(raw)

    def _get_range(self, histogram_widget):
        return GammaProp.RANGE

    def _on_value_changed(self, histogram_widget, value):
        widgets = self.widgets[histogram_widget]
        widgets.edit.setText('{:.6}'.format(value))
        if histogram_widget._image is not None:
            if self.name == 'gamma_gamma':
                # Refresh the histogram when gamma scale (ie gamma gamma) changes
                histogram_widget.update()
            elif self.channel_name is None:
                if histogram_widget._image.is_grayscale:
                    histogram_widget.gamma_or_min_max_changed.emit()
                else:
                    histogram_widget.gamma_red = value
                    histogram_widget.gamma_green = value
                    histogram_widget.gamma_blue = value
            elif not histogram_widget._image.is_grayscale:
                histogram_widget.gamma_or_min_max_changed.emit()

class MinMaxProp(ScalarProp):
    def __init__(self, scalar_props, min_max_props, name, name_in_label=None, channel_name=None):
        super().__init__(scalar_props, name, name_in_label, channel_name)
        attr_name = name
        if channel_name is not None:
            attr_name += '_' + channel_name
        min_max_props[attr_name] = self

    def instantiate(self, histogram_widget, layout):
        super().instantiate(histogram_widget, layout)
        if self.name == 'min':
            self.widgets[histogram_widget].slider.setInvertedAppearance(True)

    def _slider_raw_to_value(self, raw, histogram_widget):
        range_ = self._get_range(histogram_widget)
        value = float(raw)
        value -= MinMaxProp.SLIDER_RAW_RANGE[0]
        value /= MinMaxProp.SLIDER_RAW_RANGE_WIDTH
        if self.name == 'min':
            value = 1 - value
        value *= range_[1] - range_[0]
        value += range_[0]
        return value

    def _value_to_slider_raw(self, value, histogram_widget):
        range_ = self._get_range(histogram_widget)
        raw = value - range_[0]
        raw /= range_[1] - range_[0]
        if self.name == 'min':
            raw = 1 - raw
        raw *= MinMaxProp.SLIDER_RAW_RANGE_WIDTH
        raw += MinMaxProp.SLIDER_RAW_RANGE[0]
        return int(raw)

    def _get_range(self, histogram_widget):
        if histogram_widget._image is None:
            return (0, 1)
        else:
            return histogram_widget._image.range

    def _on_value_changed(self, histogram_widget, value):
        if not histogram_widget.allow_inversion:
            is_min = self.name == 'min'
            anti_attr = 'max' if is_min else 'min'
            if self.channel_name is not None:
                anti_attr += '_' + self.channel_name
            anti_val = getattr(histogram_widget, anti_attr)
            if is_min and value > anti_val or not is_min and value < anti_val:
                setattr(histogram_widget, anti_attr, value)
        widgets = self.widgets[histogram_widget]
        widgets.edit.setText('{:.6}'.format(value))
        if histogram_widget._image is not None:
            if self.channel_name is None:
                if histogram_widget._image.is_grayscale:
                    histogram_widget.gamma_or_min_max_changed.emit()
                else:
                    if self.name == 'min':
                        histogram_widget.min_red = value
                        histogram_widget.min_green = value
                        histogram_widget.min_blue = value
                    else:
                        histogram_widget.max_red = value
                        histogram_widget.max_green = value
                        histogram_widget.max_blue = value
            elif not histogram_widget._image.is_grayscale:
                histogram_widget.gamma_or_min_max_changed.emit()

class HistogramWidget(canvas.CanvasWidget):
    _MAX_BIN_COUNT = 1024

    _scalar_props = []
    _min_max_props = {}

    gamma_or_min_max_changed = Qt.pyqtSignal()

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
    def make_histogram_and_container_widgets(cls, parent, qsurface_format):
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
        histogram = cls(histogram_frame, qsurface_format)
        histogram_frame.layout().addWidget(histogram)
        splitter.addWidget(histogram._control_widgets_pane)
        splitter.addWidget(histogram_frame)
        histogram.channel_control_widgets_visible = False
        return (histogram, container)

    def __init__(self, parent, qsurface_format):
        super().__init__(parent, qsurface_format)
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
        for scalar_prop in HistogramWidget._scalar_props:
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

#   def initializeGL(self):
#       self._init_glfs()
#       self._glfs.glClearColor(0,0,0,1)
#       self._glfs.glClearDepth(1)
#       self._glsl_prog_g = self._build_shader_prog('g',
#                                                   'histogram_widget_vertex_shader.glsl',
#                                                   'histogram_widget_fragment_shader_g.glsl')
#       self._glsl_prog_rgb = self._build_shader_prog('rgb',
#                                                     'histogram_widget_vertex_shader.glsl',
#                                                     'histogram_widget_fragment_shader_rgb.glsl')
#       self._image_type_to_glsl_prog = {'g'   : self._glsl_prog_g,
#                                        'ga'  : self._glsl_prog_g,
#                                        'rgb' : self._glsl_prog_rgb,
#                                        'rgba': self._glsl_prog_rgb}
#       self._make_quad_vao()
#
#   def paintGL(self):
#       self._glfs.glClear(self._glfs.GL_COLOR_BUFFER_BIT | self._glfs.GL_DEPTH_BUFFER_BIT)
#       if self._image is not None:
#           if self._image.is_grayscale:
#               prog = self._image_type_to_glsl_prog[self._image.type]
#               prog.bind()
#               self._quad_buffer.bind()
#               self._glfs.glBindTexture(self._glfs.GL_TEXTURE_1D, self._tex)
#               vert_coord_loc = prog.attributeLocation('vert_coord')
#               quad_vao_binder = Qt.QOpenGLVertexArrayObject.Binder(self._quad_vao)
#               prog.enableAttributeArray(vert_coord_loc)
#               prog.setAttributeBuffer(vert_coord_loc, self._glfs.GL_FLOAT, 0, 2, 0)
#               prog.setUniformValue('tex', 0)
#               prog.setUniformValue('inv_view_size', 1/self.size().width(), 1/self.size().height())
#               if self._image.type == 'g':
#                   max_bin_val = self._image.histogram[self._image.max_histogram_bin]
#               else:
#                   max_bin_val = self._image.histogram[0, self._image.max_histogram_bin[0]]
#               inv_max_transformed_bin_val = max_bin_val**-self.gamma_gamma
#               prog.setUniformValue('inv_max_transformed_bin_val', inv_max_transformed_bin_val)
#               prog.setUniformValue('gamma_gamma', self.gamma_gamma)
#               self._glfs.glEnableClientState(self._glfs.GL_VERTEX_ARRAY)
#               self._glfs.glDrawArrays(self._glfs.GL_TRIANGLE_FAN, 0, 4)
#               self._quad_buffer.release()
#               prog.release()

    def _on_image_changed(self, image):
        if image is None or image.is_grayscale:
            self.channel_control_widgets_visible = False
        else:
            self.channel_control_widgets_visible = True
        range_changed = (self._image is None or image is None) or self._image.range != image.range
        if range_changed:
            if image is None or image.is_grayscale:
                self._min_max_props['max'].propagate_slider_value(self)
                self._min_max_props['min'].propagate_slider_value(self)
            else:
                for min_max_prop in self._min_max_props.values():
                    min_max_prop.propagate_slider_value(self)
        self._correct_inversion()

#       try:
#           self.makeCurrent()
#           if self._image is not None and (image is None or self._image.histogram.shape != image.histogram.shape):
#               if self._tex is not None:
#                   self._glfs.glDeleteTextures(1, (self._tex,))
#                   self._tex = None
#               self._image = None
#           if image is not None:
#               if self._tex is None:
#                   self._tex = self._glfs.glGenTextures(1)
#                   self._glfs.glBindTexture(self._glfs.GL_TEXTURE_1D, self._tex)
#                   self._glfs.glTexParameteri(self._glfs.GL_TEXTURE_1D, self._glfs.GL_TEXTURE_WRAP_S, self._glfs.GL_CLAMP_TO_EDGE)
#                   self._glfs.glTexParameteri(self._glfs.GL_TEXTURE_1D, self._glfs.GL_TEXTURE_WRAP_T, self._glfs.GL_CLAMP_TO_EDGE)
#                   # self._tex stores histogram bin counts - values that are intended to be addressed by element without
#                   # interpolation.  Thus, nearest neighbor for texture filtering.
#                   self._glfs.glTexParameteri(self._glfs.GL_TEXTURE_1D, self._glfs.GL_TEXTURE_MIN_FILTER, self._glfs.GL_NEAREST)
#                   self._glfs.glTexParameteri(self._glfs.GL_TEXTURE_1D, self._glfs.GL_TEXTURE_MAG_FILTER, self._glfs.GL_NEAREST)
#               else:
#                   self._glfs.glBindTexture(self._glfs.GL_TEXTURE_1D, self._tex)
#               self._glfs.glPixelStorei(self._glfs.GL_PACK_ALIGNMENT, 1)
#               self._glfs.glPixelStorei(self._glfs.GL_UNPACK_ALIGNMENT, 1)
#               if image.is_grayscale:
#                   if image.type == 'g':
#                       intensity_histogram = image.histogram
#                       max_bin_val = intensity_histogram[self._image.max_histogram_bin]
#                   else:
#                       intensity_histogram = image.histogram[0]
#                       max_bin_val = intensity_histogram[self._image.max_histogram_bin[0]]
#                   self._glfs.glTexImage1D(self._glfs.GL_TEXTURE_1D, 0, self._glfs.GL_LUMINANCE32UI_EXT,
#                                           len(intensity_histogram), 0, self._glfs.GL_LUMINANCE_INTEGER_EXT,
#                                           self._glfs.GL_UNSIGNED_INT, intensity_histogram.data)
#               else:
#                   pass
#                   # personal time todo: per-channel RGB histogram support
#               self._image = image
#           self.update()
#       finally:
#           self.doneCurrent()
        self._image = image

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
