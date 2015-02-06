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

from .canvas_widget import CanvasWidget
import math
import numpy
from PyQt5 import Qt

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

    def __set__(self, histogram_widget, gamma):
        if histogram_widget is None:
            raise AttributeError("Can't set instance attribute of class.")
        if gamma is None:
            raise ValueError('None is not a valid {} value.'.format(self.name))
        range_ = self._get_range(histogram_widget)
        if gamma < range_[0] or gamma > range_[1]:
            raise ValueError('Value supplied for {} must be in the range [{}, {}].'.format(self.name, range_[0], range_[1]))
        widgets = self.widgets[histogram_widget]
        widgets.slider.setValue(self._value_to_slider_raw(gamma, histogram_widget))

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

    def _slider_raw_to_value(self, raw, histogram_widget):
        raise NotImplementedError('Pure virtual method called.')

    def _value_to_slider_raw(self, value, histogram_widget):
        raise NotImplementedError('Pure virtual method called.')

    def _get_range(self, histogram_widget):
        raise NotImplementedError('Pure virtual method called.')

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
    def _slider_raw_to_value(self, raw, histogram_widget):
        range_ = self._get_range(histogram_widget)
        value = float(raw)
        value -= MinMaxProp.SLIDER_RAW_RANGE[0]
        value /= MinMaxProp.SLIDER_RAW_RANGE_WIDTH
        value *= range_[1] - range_[0]
        value += range_[0]
        return value

    def _value_to_slider_raw(self, value, histogram_widget):
        range_ = self._get_range(histogram_widget)
        raw = value - range_[0]
        raw /= range_[1] - range_[0]
        raw *= MinMaxProp.SLIDER_RAW_RANGE_WIDTH
        raw += MinMaxProp.SLIDER_RAW_RANGE[0]
        return int(raw)

    def _get_range(self, histogram_widget):
        if histogram_widget._image is None:
            return (0, 1)
        else:
            return histogram_widget._image.range

    def _on_value_changed(self, histogram_widget, value):
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

class HistogramWidget(CanvasWidget):
    _NUMPY_DTYPE_TO_LIMITS_AND_QUANT = {
        numpy.uint8  : (0, 256, 'd'),
        numpy.uint16 : (0, 65535, 'd'),
        numpy.float32: (0, 1, 'c')}
    _scalar_props = []

    gamma_or_min_max_changed = Qt.pyqtSignal()

    gamma_gamma = GammaProp(_scalar_props, 'gamma_gamma', '\u03b3\u03b3')
    gamma = GammaProp(_scalar_props, 'gamma', '\u03b3')
    gamma_red = GammaProp(_scalar_props, 'gamma', '\u03b3', 'red')
    gamma_green = GammaProp(_scalar_props, 'gamma', '\u03b3', 'green')
    gamma_blue = GammaProp(_scalar_props, 'gamma', '\u03b3', 'blue')
    max = MinMaxProp(_scalar_props, 'max')
    min = MinMaxProp(_scalar_props, 'min')
    max_red = MinMaxProp(_scalar_props, 'max', channel_name='red')
    min_red = MinMaxProp(_scalar_props, 'min', channel_name='red')
    max_green = MinMaxProp(_scalar_props, 'max', channel_name='green')
    min_green = MinMaxProp(_scalar_props, 'min', channel_name='green')
    max_blue = MinMaxProp(_scalar_props, 'max', channel_name='blue')
    min_blue = MinMaxProp(_scalar_props, 'min', channel_name='blue')

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
        self._make_control_widgets_pane()

    def _make_control_widgets_pane(self):
        self._control_widgets_pane = Qt.QWidget(self)
        layout = Qt.QGridLayout()
        self._control_widgets_pane.setLayout(layout)
        self._channel_control_widgets = []
        self._channel_control_widgets_visible = True
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
#       self._gamma_transform_checkbox = Qt.QCheckBox('Enable gamma transform')
#       layout.addWidget(self._gamma_transform_checkbox, row_ref[0], 0, 1, -1)

    def initializeGL(self):
        self._init_glfs()
        self._glfs.glClearColor(0,0,0,1)
        self._glfs.glClearDepth(1)
#       self._glsl_prog_g = self._build_shader_prog('g',
#                                                   'histogram_widget_vertex_shader.glsl',
#                                                   'histogram_widget_fragment_shader_g.glsl')
#       self._glsl_prog_rgb = self._build_shader_prog('rgb',
#                                                     'histogram_widget_vertex_shader.glsl',
#                                                     'histogram_widget_fragment_shader_rgb.glsl')
#       self._image_type_to_glsl_prog = {'g'   : self._glsl_prog_g,
#                                        'ga'  : self._glsl_prog_ga,
#                                        'rgb' : self._glsl_prog_rgb,
#                                        'rgba': self._glsl_prog_rgba}
        self._make_quad_buffer()

    def paintGL(self):
        pass

    def resizeGL(self, x, y):
        pass

    def _on_image_changed(self, image):
        if image is None or image.is_grayscale:
            self.channel_control_widgets_visible = False
        else:
            self.channel_control_widgets_visible = True
        self._image = image
        self.update()

    @property
    def channel_control_widgets_visible(self):
        return self._channel_control_widgets_visible

    @channel_control_widgets_visible.setter
    def channel_control_widgets_visible(self, visible):
        if visible != self._channel_control_widgets_visible:
            self._channel_control_widgets_visible = visible
            for widget in self._channel_control_widgets:
                widget.setVisible(visible)
