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

import datetime
import numpy
from PyQt5 import Qt
import textwrap
import time
from .basic_image import BasicImage

class _Property(property):
    # Derived from "property" for the sole reason that IPython's question-mark magic is special-cased for
    # properties.  Deriving from property causes _Property to receive the same treatment, providing
    # useful output for something.prop? in IPython (where prop is a _Property instance).
    def __init__(self, properties, name, default_value_callback, transform_callback=None, pre_set_callback=None, post_set_callback=None, doc=None):
        self.name = name
        self.var_name = '_' + name
        self.default_val_var_name = '_default_' + name
        self.changed_signal_name = name + '_changed'
        self.default_value_callback = default_value_callback
        self.transform_callback = transform_callback
        self.pre_set_callback = pre_set_callback
        self.post_set_callback = post_set_callback
        if doc is not None:
            self.__doc__ = doc
        properties.append(self)

    def instantiate(self, image):
        setattr(image, self.default_val_var_name, self.default_value_callback(image))
        getattr(image, self.changed_signal_name).connect(image.changed)

    def update_default(self, image):
        if hasattr(image, self.var_name):
            # An explicitly set value is overriding the default, so even if the default has changed, the apparent value of the property has not
            setattr(image, self.default_val_var_name, self.default_value_callback(image))
        else:
            # The default value is the apparent value, meaning that we must check if the default has changed and signal an apparent value change
            # if it has
            old_default = getattr(image, self.default_val_var_name)
            new_default = self.default_value_callback(image)
            if old_default != new_default:
                setattr(image, self.default_val_var_name, new_default)
                getattr(image, self.changed_signal_name).emit(image)

    def __get__(self, image, _=None):
        if image is None:
            return self
        try:
            return getattr(image, self.var_name)
        except AttributeError:
            return getattr(image, self.default_val_var_name)

    def __set__(self, image, v):
        if self.transform_callback is not None:
            v = self.transform_callback(image, v)
        if not hasattr(image, self.var_name) or v != getattr(image, self.var_name):
            if self.pre_set_callback is not None:
                self.pre_set_callback(image, v)
            setattr(image, self.var_name, v)
            if self.post_set_callback is not None:
                self.post_set_callback(image, v)
            getattr(image, self.changed_signal_name).emit(image)

    def __delete__(self, image):
        """Reset to default value by way of removing the explicitly set override, causing the apparent value to be default."""
        try:
            old_value = getattr(image, self.var_name)
            delattr(image, self.var_name)
            new_value = getattr(image, self.default_val_var_name)
            if old_value != new_value:
                if self.post_set_callback is not None:
                    self.post_set_callback(image, new_value)
                getattr(image, self.changed_signal_name).emit(image)
        except AttributeError:
            # Property was already using default value
            pass
        

class Image(BasicImage, Qt.QObject):
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

    Although Image uses _Property descriptors, subclasses adding properties are not obligated
    to use _Property to represent the additional properties.  The regular @property decorator syntax or property(..) builtin
    remain available - _Property provides an abstraction that is potentially convenient and worth understanding and using when
    defining a large number of properties."""

    GAMMA_RANGE = (0.0625, 16.0)
    IMAGE_TYPE_TO_GETCOLOR_EXPRESSION = {
        'g'   : 'vec4(s.r, s.r, s.r, 1.0f)',
        'ga'  : 'vec4(s.r, s.r, s.r, s.a)',
        'rgb' : 'vec4(s.r, s.g, s.b, 1.0f)',
        'rgba': 's'}
    # Blend functions adapted from http://dev.w3.org/SVG/modules/compositing/master/ 
    BLEND_FUNCTIONS = {
        'src' :      ('dca = sca;',
                      'da = sa;'),
        'src-over' : ('dca = sca + dca * (1.0f - sa);',
                      'da = sa + da - sa * da;'),
        'dst-over' : ('dca = dca + sca * (1.0f - da);',
                      'da = sa + da - sa * da;'),
        'plus'     : ('dca += sca;',
                      'da += sa;'),
        'multiply' : ('dca = sca * dca + sca * (1.0f - da) + dca * (1.0f - sa);',
                      'da = sa + da - sa * da;'),
        'screen'   : ('dca = sca + dca - sca * dca;',
                      'da = sa + da - sa * da;'),
        'overlay'  : ('isa = 1.0f - sa; osa = 1.0f + sa;',
                      'ida = 1.0f - da; oda = 1.0f + da;',
                      'sada = sa * da;',
                      'for(i = 0; i < 3; ++i){',
                      '    dca[i] = (dca[i] + dca[i] <= da) ?',
                      '             (sca[i] + sca[i]) * dca[i] + sca[i] * ida + dca[i] * isa :',
                      '             sca[i] * oda + dca[i] * osa - (dca[i] + dca[i]) * sca[i] - sada;}',
                      'da = sa + da - sada;'),
        'difference':('dca = (sca * da + dca * sa - (sca + sca) * dca) + sca * (1.0f - da) + dca * (1.0f - sa);',
                      'da = sa + da - sa * da;')}
    for k, v in BLEND_FUNCTIONS.items():
        BLEND_FUNCTIONS[k] = '    // blending function name: {}\n    '.format(k) + '\n    '.join(v)
    del k, v
    # A call to .set_data or a change to any mutable property potentially impacts image presentation.  For convenience, changed is emitted whenever
    # .set_data or .refresh is called or any of the more specific mutable-property-changed signals are emitted.
    # 
    # For example, this single call supports extensibility by subclassing:
    # image_instance.changed.connect(something.refresh)
    # And that single call replaces the following set of calls, which is not even complete if Image is subclassed:
    # image_instance.name_changed.connect(something.refresh)
    # image_instance.data_changed.connect(something.refresh)
    # image_instance.min_changed.connect(something.refresh)
    # image_instance.max_changed.connect(something.refresh)
    # image_instance.gamma_changed.connect(something.refresh)
    # image_instance.trilinear_filtering_enabled_changed.connect(something.refresh)
    # image_instance.auto_getcolor_expression_enabled_changed.connect(something.refresh)
    # image_instance.getcolor_expression_changed.connect(something.refresh)
    # image_instance.extra_transformation_expression_changed.connect(something.refresh)
    # image_instance.global_alpha_changed.connect(something.refresh)
    # image_instance.mute_enabled_changed.connect(something.refresh)
    #
    # In the __init__ function of any Image subclass that adds presentation-affecting properties
    # and associated change notification signals, do not forget to connect the subclass's change signals to changed.
    changed = Qt.pyqtSignal(object)
    data_changed = Qt.pyqtSignal(object)

    def __init__(self, data, is_twelve_bit=False, float_range=None, shape_is_width_height=True, name=None, parent=None):
        Qt.QObject.__init__(self, parent)
        BasicImage.set_data(self, data, is_twelve_bit, float_range, shape_is_width_height)
        self._retain_auto_min_max_enabled_on_min_max_change = False
        self._retain_auto_getcolor_expression_enabled_on_getcolor_expression_change = False
        for property in self.properties:
            property.instantiate(self)
        if name:
            self.setObjectName(name)
        if self.auto_min_max_enabled:
            self.do_auto_min_max()
        self._blend_function_impl = self.BLEND_FUNCTIONS[self.blend_function]
        self.objectNameChanged.connect(lambda: self.name_changed.emit(self))
        self.name_changed.connect(self.changed)
        self.data_changed.connect(self.changed)

    def set_data(self, data, is_twelve_bit=False, float_range=None, shape_is_width_height=True, keep_name=True, name=None):
        """If keep_name is True, the existing name is not changed, and the value supplied for the name argument is ignored.
        If keep_name is False, the existing name is replaced with the supplied name or is cleared if supplied name is None
        or an empty string."""
        BasicImage.set_data(self, data, is_twelve_bit, float_range, shape_is_width_height)
        if not keep_name:
            self.name = name
        for property in self.properties:
            property.update_default(self)
        self.data_changed.emit(self)

    def refresh(self):
        BasicImage.refresh(self)
        if self.auto_min_max_enabled:
            self.do_auto_min_max()
        self.data_changed.emit(self)

    def generate_contextual_info_for_pos(self, x, y, idx=None):
        if not self.visible:
            return
        sz = self.size
        if 0 <= x < sz.width() and 0 <= y < sz.height():
            type_ = self.type
            num_channels = self.num_channels
            name = self.name
            mst = '' if idx is None else '{: 3}, '.format(idx)
            if name:
                mst += '"' + name + '", '
            mst+= 'x:{} y:{} '.format(x, y)
            vt = '(' + ' '.join((c + ':{}' for c in self.type)) + ')'
            if num_channels == 1:
                vt = vt.format(self.data[x, y])
            else:
                vt = vt.format(*self.data[x, y])
            return mst+vt

    properties = []

    visible = _Property(
        properties, 'visible',
        doc = textwrap.dedent(
            """\
            Generally, a non-visible image is not visible in the "main view" but does remain visible in specialized views,
            such as the histogram view and image stack table widget.

            In more detail:
            If an Image's visible property is False, that Image does not contribute to mixed output.  For example,
            any single pixel in an ImageStackItem rendering may represent the result of blending a number of Images,
            whereas only one Image at a time may be associated with a HistogramItem; no HistogramItem pixel in the
            rendering of a HistogramItem is a function of more than one Image.  Therefore, a non-visible Image that is part
            of a SignalingList that is associated with an ImageStackItem will not be visible in the output of that
            ImageStackItem's render function, although the histogram of the Image will still be visible in the output
            of the render function of a HistogramItem associated with the Image."""),
        default_value_callback = lambda image: True,
        transform_callback = lambda image, v: bool(v))

    def _auto_min_max_enabled_post_set(self, v):
        if v:
            self.do_auto_min_max()
    auto_min_max_enabled = _Property(
        properties, 'auto_min_max_enabled',
        default_value_callback = lambda image: False,
        transform_callback = lambda image, v: bool(v),
        post_set_callback = _auto_min_max_enabled_post_set)

    def _min_max_pre_set(self, v):
        r = self.range
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
    min = _Property(
        properties, 'min',
        default_value_callback = lambda image: float(image.range[0]),
        transform_callback = lambda image, v: float(v),
        pre_set_callback = _min_max_pre_set,
        post_set_callback = lambda image, v, f=_min_max_post_set: f(image, v, False))
    max = _Property(
        properties, 'max',
        default_value_callback = lambda image: float(image.range[1]),
        transform_callback = lambda image, v: float(v),
        pre_set_callback = _min_max_pre_set,
        post_set_callback = lambda image, v, f=_min_max_post_set: f(image, v, True))

    def _gamma_pre_set(self, v):
        r = self.GAMMA_RANGE
        if not r[0] <= v <= r[1]:
            raise ValueError('gamma value must be in the closed interval [{}, {}].'.format(*r))
    gamma = _Property(
        properties, 'gamma',
        default_value_callback = lambda image: 1.0,
        transform_callback = lambda image, v: float(v),
        pre_set_callback = _gamma_pre_set)

    trilinear_filtering_enabled = _Property(
        properties, 'trilinear_filtering_enabled',
        default_value_callback = lambda image: True,
        transform_callback = lambda image, v: bool(v))

    SHAD_PROP_HELP = textwrap.dedent("""\
        The GLSL fragment shader used to render an ImageStackItem is generated by iterating through ImageStackItem.image_stack,
        replacing the ${values} in the following template with with those of the Image (or
        ImageStackItem.BLEND_FUNCTIONS[Image.blend_function] in the case of ${blend_function}) at each iteration and
        appending the resulting text to a string.  The accumulated string is the GLSL fragment shader's source code.

            s = texture2D(tex_${idx}, tex_coord);
            s = ${getcolor_expression};
            sa = clamp(s.a, 0, 1) * global_alpha_${idx};
            sc = min_max_gamma_transform(s.rgb, rescale_min_${idx}, rescale_range_${idx}, gamma_${idx});
            ${extra_transformation_expression}; // extra_transformation_expression
            sca = sc * sa;
            ${blend_function}
            da = clamp(da, 0, 1);
            dca = clamp(dca, 0, 1);

        So, the value stored in an Image's .getcolor_expression property replaces ${getcolor_expression}.  Supplying
        None or an empty string would create a GLSL syntax error that must be rectified before an ImageStackItem
        containing the Image in question can be successfully rendered (unless the Image's .visible property is False).
        In order to revert .getcolor_expression to something appropriate for the Image's .type, simply assign True
        to that Image's .auto_getcolor_expression_enabled property (likewise, replacing the contents of
        .getcolor_expression causes .auto_getcolor_expression_enabled to assume the value False).

        Unlike .getcolor_expression, .extra_transformation_expression does accept None or an empty string.  Supplying
        either results in a GLSL line consisting of "; // extra_transformation_expression", which compiles to nothing,
        with the effect that None or an empty string result in no extra transformation expression being applied (this
        is the default).""")

    def _auto_getcolor_expression_enabled_post_set(self, v):
        if v:
            self.do_auto_getcolor_expression()
    auto_getcolor_expression_enabled = _Property(
        properties, 'auto_getcolor_expression_enabled',
        default_value_callback = lambda image: True,
        transform_callback = lambda image, v: bool(v),
        post_set_callback = _auto_getcolor_expression_enabled_post_set,
        doc = SHAD_PROP_HELP)

    def _getcolor_expression_post_set(self, v):
        if not self._retain_auto_getcolor_expression_enabled_on_getcolor_expression_change:
            self.auto_getcolor_expression_enabled = False
    getcolor_expression = _Property(
        properties, 'getcolor_expression',
        default_value_callback = lambda image: image.IMAGE_TYPE_TO_GETCOLOR_EXPRESSION[image.type],
        transform_callback = lambda image, v: str(v),
        post_set_callback = lambda image, v, f=_getcolor_expression_post_set: f(image, v),
        doc = SHAD_PROP_HELP)

    extra_transformation_expression = _Property(
        properties, 'extra_transformation_expression',
        default_value_callback = lambda image: None,
        transform_callback = lambda image, v: None if v is None else str(v),
        doc = SHAD_PROP_HELP)

    def _blend_function_pre_set(self, v):
        if v not in self.BLEND_FUNCTIONS:
            raise ValueError('The string assigned to blend_function must be one of:\n' + '\n'.join("'" + s + "'" for s in sorted(self.BLEND_FUNCTIONS.keys())))
    blend_function = _Property(
        properties, 'blend_function',
        default_value_callback = lambda image: 'screen',
        transform_callback = lambda image, v: str(v),
        pre_set_callback = lambda image, v, f=_blend_function_pre_set: f(image, v),
        doc = SHAD_PROP_HELP)
    @property
    def blend_function_impl(self):
        return self.BLEND_FUNCTIONS[self.blend_function]

    def _global_alpha_pre_set(self, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('The value assigned to global_alpha must be in the closed interval [0, 1].')
    global_alpha = _Property(
        properties, 'global_alpha',
        default_value_callback = lambda image: 1.0,
        transform_callback = lambda image, v: float(v),
        pre_set_callback = lambda image, v, f=_global_alpha_pre_set: f(image, v))

    for property in properties:
        exec(property.changed_signal_name + ' = Qt.pyqtSignal(object)')
    del property
    del SHAD_PROP_HELP

    # NB: This a property, not a _Property.  There is already a change signal, setter, and a getter for objectName, which
    # we proxy/use.
    name_changed = Qt.pyqtSignal(object)
    name = property(
        Qt.QObject.objectName,
        lambda self, name: self.setObjectName('' if name is None else name),
        doc='Property proxy for QObject::objectName Qt property, which is directly accessible via the objectName getter and '
            'setObjectName setter.  Upon change, objectNameChanged is emitted.')

    def __repr__(self):
        name = self.name
        return '{}, {}{}>'.format(
            super().__repr__()[:-1],
            'with name "{}"'.format(name) if name else 'unnamed',
            ', visible=False' if not self.visible else '')

    def do_auto_min_max(self):
        self._retain_auto_min_max_enabled_on_min_max_change = True
        try:
            extremae = self.extremae
            if self.has_alpha_channel:
                eae = extremae[:-1, 0].min(), extremae[:-1, 1].max()
            elif self.num_channels > 1:
                eae = extremae[:, 0].min(), extremae[:, 1].max()
            else:
                eae = extremae
            self.min, self.max = eae
        finally:
            self._retain_auto_min_max_enabled_on_min_max_change = False

    def do_auto_getcolor_expression(self):
        self._retain_auto_getcolor_expression_enabled_on_getcolor_expression_change = True
        try:
            self.getcolor_expression = self.IMAGE_TYPE_TO_GETCOLOR_EXPRESSION[self.type]
        finally:
            self._retain_auto_getcolor_expression_enabled_on_getcolor_expression_change = False
