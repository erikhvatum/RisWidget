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
import time
from .immutable_image import ImmutableImage

class _Property:
    def __init__(self, properties, name, default_value_callback, transform_callback=None, pre_set_callback=None, post_set_callback=None):
        self.name = name
        self.var_name = '_' + name
        self.changed_signal_name = name + '_changed'
        self.default_value_callback = default_value_callback
        self.transform_callback = transform_callback
        self.pre_set_callback = pre_set_callback
        self.post_set_callback = post_set_callback
        properties.append(self)

    def instantiate(self, iiwmp):
        setattr(iiwmp, self.var_name, self.default_value_callback(iiwmp))
        iiwmp.property_changed.connect(getattr(iiwmp, self.changed_signal_name))

    def __get__(self, iiwmp, _=None):
        if iiwmp is None:
            return self
        return getattr(iiwmp, self.var_name)

    def __set__(self, iiwmp, v):
        if self.transform_callback is not None:
            v = self.transform_callback(iiwmp, v)
        if v != getattr(iiwmp, self.var_name):
            if self.pre_set_callback is not None:
                self.pre_set_callback(iiwmp, v)
            setattr(iiwmp, self.var_name, v)
            if self.post_set_callback is not None:
                self.post_set_callback(iiwmp, v)
            getattr(iiwmp, self.changed_signal_name).emit()

    def __delete__(self, iiwmp):
        self.__set__(iiwmp, self.default_value_callback(iiwmp))

class ImmutableImageWithMutableProperties(ImmutableImage, Qt.QObject):
    """ImmutableImageWithMutableProperties contains immutable (read-only) image data and metadata inherited from ImmutableImage
    plus mutable (modifyable) properties controling image presentation and naming.

    The property_changed signal is emitted when any of the specific property changed signals are emitted.  In the case
    where any property change should cause a function to be executed, do property_changed.connect(your_function) rather than
    min_changed.connect(your_function); max_changed.connect(your_function); etc.

    Although ImmutableImageWithMutableProperties uses _Property descriptors, subclasses adding properties are not obligated
    to use _Property to represent the additional properties.  The regular @property decorator syntax or property(..) builtin
    remain available - _Property provides an abstraction that is potentially convenient and worth understanding & using when
    defining a large number of properties."""

    GAMMA_RANGE = (0.0625, 16.0)
    IMAGE_TYPE_TO_GETCOLOR_EXPRESSION = {
        'g'   : 'vec4(s.r, s.r, s.r, 1.0f)',
        'ga'  : 'vec4(s.r, s.r, s.r, s.a)',
        'rgb' : 'vec4(s.r, s.g, s.b, 1.0f)',
        'rgba': 's'}
    # A change to any mutable property potentially impacts image presentation.  For convenience, property_changed is emitted whenever
    # any of the more specific mutable-property-changed signals are emitted.
    # 
    # For example, this single call:
    # immutable_image_with_mutable_properties_instance.property_changed.connect(something.refresh)
    # Rather than:
    # immutable_image_with_mutable_properties_instance.min_changed.connect(something.refresh)
    # immutable_image_with_mutable_properties_instance.max_changed.connect(something.refresh)
    # immutable_image_with_mutable_properties_instance.gamma_changed.connect(something.refresh)
    # immutable_image_with_mutable_properties_instance.trilinear_filtering_enabled_changed.connect(something.refresh)
    # immutable_image_with_mutable_properties_instance.auto_getcolor_expression_enabled_changed.connect(something.refresh)
    # immutable_image_with_mutable_properties_instance.getcolor_expression_changed.connect(something.refresh)
    # immutable_image_with_mutable_properties_instance.extra_transformation_expression_changed.connect(something.refresh)
    #
    # In the __init__ function of any ImmutableImageWithMutableProperties subclass that adds presentation-affecting properties
    # and associated change notification signals, do not forget to connect the subclass's change signals to property_changed.
    property_changed = Qt.pyqtSignal()

    def __init__(self, data, is_twelve_bit=False, float_range=None, shape_is_width_height=True, name=None, parent=None):
        Qt.QObject.__init__(self, parent)
        if name is None:
            name = self._generate_anon_name()
        self.setObjectName(name)
        ImmutableImage.__init__(self, data, is_twelve_bit, float_range, shape_is_width_height)
        self._retain_auto_min_max_enabled_on_min_max_change = False
        self._retain_auto_getcolor_expression_enabled_on_getcolor_expression_change = False
        for property in ImmutableImageWithMutableProperties.properties:
            property.instantiate(self)

        mm = self.min_max
        if self.has_alpha_channel:
            self._default_min_max_values = mm[:-1, 0].min(), mm[:-1, 1].max()
        elif self.num_channels > 1:
            self._default_min_max_values = mm[:, 0].min(), mm[:, 1].max()
        else:
            self._default_min_max_values = tuple(mm)

        if self.auto_min_max_enabled:
            self.do_auto_min_max()

    properties = []

    # Considering that image data is immutable, an auto_min_max_enabled property may seem a strange thing to have.  It
    # does have an effect, even in strictly in the context this class: if the min and max properties are not equal
    # to the min and max channel intensity values (excluding alpha) and True is assigned to auto_min_max_enabled,
    # the min and max channel intensity values are assigned to the min and max properties.
    #
    # In the larger context, where ImageStack contains a number of layers, each represented by an ImmutableImageWithMutableProperties
    # (or subclass) instance, additional utility is apparent: if the content of a layer is replaced by direct assignment of
    # a numpy array, implicitly causing a new ImmutableImageWithMutableProperties to be instaniated, the auto_min_max_enabled
    # value may be used by the new instance, preventing auto min/maxness from being forgotten exactly when it is typically
    # desired.
    def _auto_min_max_enabled_post_set(self, v):
        if v:
            self.do_auto_min_max()
    auto_min_max_enabled = _Property(
        properties, 'auto_min_max_enabled',
        default_value_callback = lambda iiwmp: True,
        transform_callback = lambda iiwmp, v: bool(v),
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
        default_value_callback = lambda iiwmp: iiwmp._default_min_max_values[0],
        transform_callback = lambda iiwmp, v: float(v),
        pre_set_callback = _min_max_pre_set,
        post_set_callback = lambda iiwmp, v, f=_min_max_post_set: f(iiwmp, v, False))
    max = _Property(
        properties, 'max',
        default_value_callback = lambda iiwmp: iiwmp._default_min_max_values[1],
        transform_callback = lambda iiwmp, v: float(v),
        pre_set_callback = _min_max_pre_set,
        post_set_callback = lambda iiwmp, v, f=_min_max_post_set: f(iiwmp, v, True))

    def _gamma_pre_set(self, v):
        r = self.GAMMA_RANGE
        if not r[0] <= v <= r[1]:
            raise ValueError('gamma value must be in the closed interval [{}, {}].'.format(*r))
    gamma = _Property(
        properties, 'gamma',
        default_value_callback = lambda iiwmp: 1.0,
        transform_callback = lambda iiwmp, v: float(v),
        pre_set_callback = _gamma_pre_set)

    trilinear_filtering_enabled = _Property(
        properties, 'trilinear_filtering_enabled',
        default_value_callback = lambda iiwmp: True,
        transform_callback = lambda iiwmp, v: bool(v))

    # The rationale for auto_getcolor_expression_enabled is the same as for auto_min_max_enabled:
    # so that it may be preserved and applied when an implicitly created instance of
    # ImmutableImageWithMutableProperties replaces and existing instance.
    def _auto_getcolor_expression_enabled_post_set(self, v):
        if v:
            self.do_auto_getcolor_expression()
    auto_getcolor_expression_enabled = _Property(
        properties, 'auto_getcolor_expression_enabled',
        default_value_callback = lambda iiwmp: True,
        transform_callback = lambda iiwmp, v: bool(v),
        post_set_callback = _auto_getcolor_expression_enabled_post_set)

    def _getcolor_expression_post_set(self, v):
        if not self._retain_auto_getcolor_expression_enabled_on_getcolor_expression_change:
            self.auto_getcolor_expression_enabled = False
    getcolor_expression = _Property(
        properties, 'getcolor_expression',
        default_value_callback = lambda iiwmp: iiwmp.IMAGE_TYPE_TO_GETCOLOR_EXPRESSION[iiwmp.type],
        transform_callback = lambda iiwmp, v: str(v),
        post_set_callback = lambda iiwmp, v, f=_getcolor_expression_post_set: f(iiwmp, v))

    extra_transformation_expression = _Property(
        properties, 'extra_transformation_expression',
        default_value_callback = lambda iiwmp: None,
        transform_callback = lambda iiwmp, v: str(v))

    for property in properties:
        exec(property.changed_signal_name + ' = Qt.pyqtSignal()')
    del property

    name = property(
        Qt.QObject.objectName,
        Qt.QObject.setObjectName,
        doc='Property proxy for QObject::objectName Qt property, which is directly accessible via the objectName getter and '
            'setObjectName setter.  Upon change, objectNameChanged is emitted.')

    def __repr__(self):
        return '{}, with name "{}">'.format(
            super().__repr__()[:-1],
            self.name)

    def do_auto_min_max(self):
        self._retain_auto_min_max_enabled_on_min_max_change = True
        try:
            del self.min, self.max
        finally:
            self._retain_auto_min_max_enabled_on_min_max_change = False

    def do_auto_getcolor_expression(self):
        self._retain_auto_getcolor_expression_enabled_on_getcolor_expression_change = True
        try:
            self.getcolor_expression = self.IMAGE_TYPE_TO_GETCOLOR_EXPRESSION[self.type]
        finally:
            self._retain_auto_getcolor_expression_enabled_on_getcolor_expression_change = False

    _previous_anon_name_timestamp = None
    _previous_anon_name_timestamp_dupe_count = None
    @staticmethod
    def _generate_anon_name():
        timestamp = time.time()
        if timestamp == ImmutableImageWithMutableProperties._previous_anon_name_timestamp:
            ImmutableImageWithMutableProperties._previous_anon_name_timestamp_dupe_count += 1
        else:
            ImmutableImageWithMutableProperties._previous_anon_name_timestamp = timestamp
            ImmutableImageWithMutableProperties._previous_anon_name_timestamp_dupe_count = 0
        name = str(timestamp)
        if ImmutableImageWithMutableProperties._previous_anon_name_timestamp_dupe_count > 0:
            name += '-{:04}'.format(ImmutableImageWithMutableProperties._previous_anon_name_timestamp_dupe_count)
        name += ' ({})'.format(datetime.datetime.fromtimestamp(timestamp).strftime('%c'))
        return name
