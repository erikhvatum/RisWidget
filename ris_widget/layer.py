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

from PyQt5 import Qt
import textwrap
from .image import Image
from . import om

class Layer(Qt.QObject):
    """BasicImage's properties are all either computed from that ndarray, provide views into that ndarray's data (in the case of .data
    and .data_T), or, in the special cases of .is_twelve_bit for uint16 images and .range for floating-point images, represent unenforced
    constraints limiting the domain of valid values that are expected to be assumed by elements of the ndarray.

    Image adds properties such as min/max/gamma scaling that control presentation of the image data contained by BasicImage, which
    is a base class of Image.

    In summary,
    BasicImage: raw image data and essential information for interpreting that data in any context
    Image: BasicImage + presentation data and metadata for RisWidget such as rescaling min/max/gamma values and an informative name

    The changed signal is emitted when any property impacting image presentation is modified or image data is explicitly changed or refreshed.
    In the case where any image appearence change should cause a function to be executed, do changed.connect(your_function) rather than
    min_changed.connect(your_function); max_changed.connect(your_function); etc.

    Although Image uses Property descriptors, subclasses adding properties are not obligated
    to use Property to represent the additional properties.  The regular @property decorator syntax or property(..) builtin
    remain available - Property provides an abstraction that is potentially convenient and worth understanding and using when
    defining a large number of properties."""

    GAMMA_RANGE = (0.0625, 16.0)
    IMAGE_TYPE_TO_GETCOLOR_EXPRESSION = {
        'G'   : 'vec4(s.rrr, 1.0f)',
        'Ga'  : 'vec4(s.rrr, s.g)',
        'rgb' : 'vec4(s.rgb, 1.0f)',
        'rgba': 's'}
    DEFAULT_TRANSFORM_SECTION = 'out_.rgb = pow(clamp((in_.rgb - rescale_min) / (rescale_range), 0.0f, 1.0f), gamma); out_.rgba *= tint;'
    # Blend functions adapted from http://dev.w3.org/SVG/modules/compositing/master/
    BLEND_FUNCTIONS = {
        'src' :      ('dca = sca;',
                      'da = s.a;'),
        'src-over' : ('dca = sca + dca * (1.0f - s.a);',
                      'da = s.a + da - s.a * da;'),
        'dst-over' : ('dca = dca + sca * (1.0f - da);',
                      'da = s.a + da - s.a * da;'),
        'plus'     : ('dca += sca;',
                      'da += s.a;'),
        'multiply' : ('dca = sca * dca + sca * (1.0f - da) + dca * (1.0f - s.a);',
                      'da = s.a + da - s.a * da;'),
        'screen'   : ('dca = sca + dca - sca * dca;',
                      'da = s.a + da - s.a * da;'),
        'overlay'  : ('isa = 1.0f - s.a; osa = 1.0f + s.a;',
                      'ida = 1.0f - da; oda = 1.0f + da;',
                      'sada = s.a * da;',
                      'for(i = 0; i < 3; ++i){',
                      '    dca[i] = (dca[i] + dca[i] <= da) ?',
                      '             (sca[i] + sca[i]) * dca[i] + sca[i] * ida + dca[i] * isa :',
                      '             sca[i] * oda + dca[i] * osa - (dca[i] + dca[i]) * sca[i] - sada;}',
                      'da = s.a + da - sada;'),
        'difference':('dca = (sca * da + dca * s.a - (sca + sca) * dca) + sca * (1.0f - da) + dca * (1.0f - s.a);',
                      'da = s.a + da - s.a * da;')}
    for k, v in BLEND_FUNCTIONS.items():
        BLEND_FUNCTIONS[k] = '    // blending function name: {}\n    '.format(k) + '\n    '.join(v)
    del k, v
    # A change to any mutable property, including .image, potentially impacts layer presentation.  For convenience, .changed is emitted whenever
    # any mutable-property-changed signal is emitted, including as a result of assigning to .image.name, calling .image.set_data(..), or calling
    # .image.refresh().  NB: .image_changed is the more specific signal emitted in addition to .changed for modifications to .image.
    # 
    # For example, this single call supports extensibility by subclassing:
    # image_instance.changed.connect(something.refresh)
    # And that single call replaces the following set of calls, which is not even necessarily complete if Image is subclassed:
    # image_instance.name_changed.connect(something.refresh)
    # image_instance.data_changed.connect(something.refresh)
    # image_instance.min_changed.connect(something.refresh)
    # image_instance.max_changed.connect(something.refresh)
    # image_instance.gamma_changed.connect(something.refresh)
    # image_instance.trilinear_filtering_enabled_changed.connect(something.refresh)
    # image_instance.getcolor_expression_changed.connect(something.refresh)
    # image_instance.transformation_expression_changed.connect(something.refresh)
    # image_instance.tint_changed.connect(something.refresh)
    # image_instance.visible_changed.connect(something.refresh)
    # image_instance.image_changed.connect(something.refresh)
    #
    # In the __init__ function of any Image subclass that adds presentation-affecting properties
    # and associated change notification signals, do not forget to connect the subclass's change signals to changed.
    changed = Qt.pyqtSignal(object)
    name_changed = Qt.pyqtSignal(object)
    image_changed = Qt.pyqtSignal(object)
    opacity_changed = Qt.pyqtSignal(object)

    def __init__(self, image=None, name=None, parent=None):
        super().__init__(parent)
        self._retain_auto_min_max_enabled_on_min_max_change = False
        self._image = None
        for property in self.properties:
            property.instantiate(self)
        self.objectNameChanged.connect(self._on_objectNameChanged)
        self.name_changed.connect(self.changed)
        self.image_changed.connect(self.changed)
        if name:
            self.setObjectName(name)
        self.image = image

    def __repr__(self):
        name = self.name
        image = self.image
        return '{}; {}{}, image={}>'.format(
            super().__repr__()[:-1],
            'with name "{}"'.format(name) if name else 'unnamed',
            ', visible=False' if not self.visible else '',
            'None' if image is None else image.__repr__())

    def get_savable_properties_dict(self):
        ret = {prop.name : prop.__get__(self) for prop in self.properties if not prop.is_default(self)}
        ret['name'] = self.name
        return ret

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, v):
        if v is not self._image:
            if self._image is not None:
                self._image.data_changed.disconnect(self._on_image_data_changed)
            if v is not None:
                if not isinstance(v, Image):
                    v = Image(v)
                try:
                    v.data_changed.connect(self._on_image_data_changed)
                except Exception as e:
                    self._image = None
                    raise e
            self._image = v
            self._on_image_data_changed(v)

    def _on_image_data_changed(self, image):
        assert image is self.image
        self._find_auto_min_max()
        if image is not None:
            if self.auto_min_max_enabled:
                self.do_auto_min_max()
            else:
                r = image.range
                if self.min < r[0]:
                    self.min = r[0]
                if self.max > r[1]:
                    self.max = r[1]
        for property in self.properties:
            property.update_default(self)
        self.image_changed.emit(self)

#   def copy_property_values_from(self, source):
#       for property in self.properties:
#           property.copy_instance_value(source, self)
#       sname = source.name
#       if sname:
#           self.name = sname + ' dupe'
#       else:
#           self.name = 'dupe'

    def generate_contextual_info_for_pos(self, x, y, idx=None, include_layer_name=True, include_image_name=True):
        image = self.image
        if image is None:
            image_text = 'None'
        else:
            image_text = image.generate_contextual_info_for_pos(x, y, include_image_name)
            if image_text is None:
                return
        ts = []
        if idx is not None:
            ts.append('{: 3}'.format(idx))
        if include_layer_name:
            layer_name = self.name
            if layer_name:
                ts.append('"' + layer_name + '"')
        t = ' '.join(ts)
        if t:
            t += ': '
        t += image_text
        return t

    def _find_auto_min_max(self):
        image = self.image
        if image is None:
            self._auto_min_max_values = 0.0, 1.0
        else:
            extremae = image.extremae
            if image.has_alpha_channel:
                self._auto_min_max_values = extremae[:-1, 0].min(), extremae[:-1, 1].max()
            elif image.num_channels > 1:
                self._auto_min_max_values = extremae[:, 0].min(), extremae[:, 1].max()
            else:
                self._auto_min_max_values = extremae

    def do_auto_min_max(self):
        self._retain_auto_min_max_enabled_on_min_max_change = True
        try:
            self.min, self.max = self._auto_min_max_values
        finally:
            self._retain_auto_min_max_enabled_on_min_max_change = False

    properties = []

    visible = om.Property(
        properties, 'visible',
        doc = textwrap.dedent(
            """\
            Generally, a non-visible image is not visible in the "main view" but does remain visible in specialized views,
            such as the histogram view and image stack table widget.

            In more detail:
            If an Image's visible property is False, that Image does not contribute to mixed output.  For example,
            any single pixel in an LayerStackItem rendering may represent the result of blending a number of Images,
            whereas only one Image at a time may be associated with a HistogramItem; no HistogramItem pixel in the
            rendering of a HistogramItem is a function of more than one Image.  Therefore, a non-visible Image that is part
            of a SignalingList that is associated with an LayerStackItem will not be visible in the output of that
            LayerStackItem's render function, although the histogram of the Image will still be visible in the output
            of the render function of a HistogramItem associated with the Image."""),
        default_value_callback = lambda image: True,
        take_arg_callback = lambda image, v: bool(v))

    def _auto_min_max_enabled_post_set(self, v):
        if v and self.image is not None:
            self.do_auto_min_max()
    auto_min_max_enabled = om.Property(
        properties, 'auto_min_max_enabled',
        default_value_callback = lambda image: False,
        take_arg_callback = lambda image, v: bool(v),
        post_set_callback = _auto_min_max_enabled_post_set)

    def _min_max_default(self, is_max):
        image = self.image
        if image is None:
            return 65535.0 if is_max else 0.0
        else:
            return image.range[is_max]
    def _min_max_pre_set(self, v):
        image = self.image
        if image is not None:
            r = image.range
            if not r[0] <= v <= r[1]:
                raise ValueError('min/max values for this image must be in the closed interval [{}, {}].'.format(*r))
    def _min_max_post_set(self, v, is_max):
        if is_max:
            if v < self.min:
                self.min = v
        else:
            if v > self.max:
                self.max = v
        if not self._retain_auto_min_max_enabled_on_min_max_change:
            self.auto_min_max_enabled = False
    min = om.Property(
        properties, 'min',
        default_value_callback = lambda layer, f=_min_max_default: f(layer, False),
        take_arg_callback = lambda layer, v: float(v),
        pre_set_callback = _min_max_pre_set,
        post_set_callback = lambda layer, v, f=_min_max_post_set: f(layer, v, False))
    max = om.Property(
        properties, 'max',
        default_value_callback = lambda layer, f=_min_max_default: f(layer, True),
        take_arg_callback = lambda layer, v: float(v),
        pre_set_callback = _min_max_pre_set,
        post_set_callback = lambda layer, v, f=_min_max_post_set: f(layer, v, True))

    def _gamma_pre_set(self, v):
        r = self.GAMMA_RANGE
        if not r[0] <= v <= r[1]:
            raise ValueError('gamma value must be in the closed interval [{}, {}].'.format(*r))
    gamma = om.Property(
        properties, 'gamma',
        default_value_callback = lambda layer: 1.0,
        take_arg_callback = lambda layer, v: float(v),
        pre_set_callback = _gamma_pre_set)

    trilinear_filtering_enabled = om.Property(
        properties, 'trilinear_filtering_enabled',
        default_value_callback = lambda layer: True,
        take_arg_callback = lambda layer, v: bool(v))

    # TODO: finish updating SHAD_PROP_HELP
    SHAD_PROP_HELP = textwrap.dedent("""\
        The GLSL fragment shader used to render an LayerStackItem is generated by iterating through LayerStackItem.layer_stack,
        replacing the ${values} in the following template with with those of the Layer (or
        LayerStackItem.BLEND_FUNCTIONS[Image.blend_function] in the case of ${blend_function}) at each iteration and
        appending the resulting text to a string.  The accumulated string is the GLSL fragment shader's source code.

            // image_stack[${idx}]
            s = texture2D(tex_${idx}, tex_coord);
            ${getcolor_channel_mapping_expression};
            s = ${getcolor_expression};
            sa = clamp(s.a, 0, 1) * global_alpha_${idx};
            sc = min_max_gamma_transform(s.rgb, rescale_min_${idx}, rescale_range_${idx}, gamma_${idx});
            ${extra_transformation_expression}; // extra_transformation_expression
            sca = sc * sa;
            ${blend_function}
            da = clamp(da, 0, 1);
            dca = clamp(dca, 0, 1);

        So, the value stored in a Layer's .getcolor_expression property replaces ${getcolor_expression}.  Supplying
        None or an empty string would create a GLSL syntax error that must be rectified before a LayerStackItem
        containing the Layer in question can be successfully rendered (unless the Layer's .visible property is False,
        or the Layer's .image property is None). In order to revert .getcolor_expression to something appropriate
        for the Layer's image.type, simply del .getcolor_expression so that it reverts to default.""")

    def _getcolor_expression_default(self):
        image = self.image
        if image is None:
            return ''
        else:
            return self.IMAGE_TYPE_TO_GETCOLOR_EXPRESSION[image.type]
    getcolor_expression = om.Property(
        properties, 'getcolor_expression',
        default_value_callback = _getcolor_expression_default,
        take_arg_callback = lambda layer, v: '' if v is None else str(v),
        doc = SHAD_PROP_HELP)

    def _tint_take_arg(self, v):
        v = tuple(map(float, v))
        if len(v) not in (3,4) or not all(map(lambda v_: 0 <= v_ <= 1, v)):
            raise ValueError('The iteraterable assigned to .tint must represent 3 or 4 real numbers in the interval [0, 1].')
        if len(v) == 3:
            v += (1.0,)
        return v
    def _tint_preset(self, v):
        if self.tint[3] != v:
            self.opacity_changed.emit(self)
    tint = om.Property(
        properties, 'tint',
        default_value_callback = lambda layer: (1.0, 1.0, 1.0, 1.0),
        take_arg_callback = _tint_take_arg,
        pre_set_callback = _tint_preset,
        doc = textwrap.dedent("""\
            .tint: This property is used by the default .transform_section, and with that default, has
            the following meaning: .tint contains 0-1 normalized RGBA component values by which the results
            of applying .getcolor_expression are scaled."""))

    @property
    def opacity(self):
        return self.tint[3]

    @opacity.setter
    def opacity(self, v):
        v = float(v)
        if not 0 <= v <= 1:
            raise ValueError('The value assigned to .tint must be a real number in the interval [0, 1].')
        t = list(self.tint)
        t[3] = v
        self.tint = t #NB: tint takes care of emitting opacity_changed

    transform_section = om.Property(
        properties, 'transform_section',
        default_value_callback = lambda layer: layer.DEFAULT_TRANSFORM_SECTION,
        take_arg_callback = lambda layer, v: '' if v is None else str(v))

    def _blend_function_pre_set(self, v):
        if v not in self.BLEND_FUNCTIONS:
            raise ValueError('The string assigned to blend_function must be one of:\n' + '\n'.join("'" + s + "'" for s in sorted(self.BLEND_FUNCTIONS.keys())))
    blend_function = om.Property(
        properties, 'blend_function',
        default_value_callback = lambda layer: 'screen',
        take_arg_callback = lambda layer, v: str(v),
        pre_set_callback = _blend_function_pre_set,
        doc = SHAD_PROP_HELP + '\n\nSupported blend_functions:\n\n    ' + '\n    '.join("'" + s + "'" for s in sorted(BLEND_FUNCTIONS.keys())))

    for property in properties:
        exec(property.changed_signal_name + ' = Qt.pyqtSignal(object)')
    del property
    del SHAD_PROP_HELP

    # NB: This a property, not a Property.  There is already a change signal, setter, and a getter for objectName, which
    # we proxy/use.
    name_changed = Qt.pyqtSignal(object)
    def _on_objectNameChanged(self):
        self.name_changed.emit(self)
    name = property(
        Qt.QObject.objectName,
        lambda self, name: self.setObjectName('' if name is None else name),
        doc='Property proxy for QObject::objectName Qt property, which is directly accessible via the objectName getter and '
            'setObjectName setter, with change notification signal objectNameChanged.  The proxied change signal, which conforms '
            'to the requirements of ris_widget.om.signaling_list.PropertyTableModel, is name_changed.')