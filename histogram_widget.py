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
import numpy
from PyQt5 import Qt

class ScalarProp:
    self._add_

class HistogramWidget(CanvasWidget):
    _NUMPY_DTYPE_TO_LIMITS_AND_QUANT = {
        numpy.uint8  : (0, 256, 'd'),
        numpy.uint16 : (0, 65535, 'd'),
        numpy.float32: (0, 1, 'c')}

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
        self._scalar_props = {}
        self._make_control_widgets_pane()

    def _make_control_widgets_pane(self):
        self._control_widgets_pane = Qt.QWidget(self)
        layout = Qt.QGridLayout()
        self._control_widgets_pane.setLayout(layout)
        row_ref = [0]
        self._add_scalar_prop('gamma_gamma', layout, row_ref, name_in_label='\u03b3\u03b3')
        self._gamma_transform_checkbox = Qt.QCheckBox('Enable gamma transform')
        layout.addWidget(self._gamma_transform_checkbox, row_ref[0], 0, 1, -1)
        row_ref[0] += 1
        self._channel_control_widgets = []
        self._add_scalar_prop('gamma', layout, row_ref, name_in_label='\u03b3')
        self._add_scalar_prop('gamma', layout, row_ref, channel_name='red', name_in_label='\u03b3')
        self._add_scalar_prop('gamma', layout, row_ref, channel_name='green', name_in_label='\u03b3')
        self._add_scalar_prop('gamma', layout, row_ref, channel_name='blue', name_in_label='\u03b3')
        self._add_scalar_prop('min', layout, row_ref)
        self._add_scalar_prop('max', layout, row_ref)
        self._add_scalar_prop('min', layout, row_ref, channel_name='red')
        self._add_scalar_prop('max', layout, row_ref, channel_name='red')
        self._add_scalar_prop('min', layout, row_ref, channel_name='green')
        self._add_scalar_prop('max', layout, row_ref, channel_name='green')
        self._add_scalar_prop('min', layout, row_ref, channel_name='blue')
        self._add_scalar_prop('max', layout, row_ref, channel_name='blue')
        self._channel_control_widgets_visible = True

    def _add_scalar_prop(self, name, layout, row_ref, channel_name=None, name_in_label=None):
        attr_stem_str = '_' + name + '_'
        label_str = ''

        if channel_name is None:
            prop_str = name
        else:
            prop_str = channel_name + '_' + name
            attr_stem_str += channel_name + '_'
            label_str = channel_name.title() + ' '

        if prop_str in self._scalar_props:
            raise RuntimeError('Duplicate scalar property name...')
        self._scalar_props[prop_str] = 1

        if name_in_label is None:
            label_str += name if label_str else name.title()
        else:
            label_str += name_in_label
        label_str += ':'

        label = Qt.QLabel(label_str)
        slider = Qt.QSlider(Qt.Qt.Horizontal)
        slider.setRange(0, 1048576)
        edit = Qt.QLineEdit()
        layout.addWidget(label, row_ref[0], 0, Qt.Qt.AlignRight)
        layout.addWidget(slider, row_ref[0], 1)
        layout.addWidget(edit, row_ref[0], 2)
        row_ref[0] += 1
        setattr(self, attr_stem_str+'label', label)
        setattr(self, attr_stem_str+'slider', slider)
        setattr(self, attr_stem_str+'edit', edit)
        if channel_name is not None:
            self._channel_control_widgets += [label, slider, edit]

    def __getattr__(self, name):
        try:
            return self._scalar_props[name]
        except KeyError:
            pass
        raise AttributeError(name)

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
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        if image is None or image.is_grayscale:
            self.channel_control_widgets_visible = False
        else:
            self.channel_control_widgets_visible = True
        self.update()
