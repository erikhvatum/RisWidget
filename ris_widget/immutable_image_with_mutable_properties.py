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
    def __init__(self, properties, name, get_default, transform_callback=None, pre_set_callback=None, post_set_callback=None):
        self.name = name
        self.var_name = '_' + name
        self.changed_signal_name = name + '_changed'
        self.get_default = get_default
        self.transform_callback = transform_callback
        self.pre_set_callback = pre_set_callback
        self.post_set_callback = post_set_callback
        properties.append(self)

    def instantiate(self, iiwmp):
        setattr(iiwmp, self.var_name, self.get_default())
        iiwmp.property_changed.connect(getattr(iiwmp, self.changed_signal_name))

    def __get__(self, iiwmp, _=None):
        if iiwmp is None:
            return self
        return getattr(iiwmp, self.var_name)

    def __set__(self, iiwmp, v):
        if self.transform_callback is not None:
            v = self.transform_callback(v)
        if v != getattr(iiwmp, self.var_name):
            if pre_set_callback is not None:
                self.pre_set_callback(v)
            setattr(iiwmp, self.var_name, v)
            if post_set_callback is not None:
                self.post_set_callback(v)
            getattr(iiwmp, self.changed_signal_name).emit()

    def __delete__(self, iiwmp):
        self.__set__(iiwmp, self.get_default())

class ImmutableImageWithMutableProperties(ImmutableImage, Qt.QObject):
    """ImmutableImageWithMutableProperties contains immutable (read-only) image data and metadata inherited from ImmutableImage
    plus modifyable (mutable) properties controling image presentation and naming.

    Note that the property_changed signal is emitted when any of the specific property changed signals are emitted.  In the case
    where any property change should cause a function to be executed, do property_changed.connect(your_function) rather than
    min_changed.connect(your_function); max_changed.connect(your_function); etc."""

    def __init__(self, data, is_twelve_bit=False, float_range=None, shape_is_width_height=True, name=None, parent=None):
        Qt.QObject.__init__(self, parent)
        if name is None:
            name = self._generate_anon_name()
        self.setObjectName(name)
        ImmutableImage.__init__(self, data, is_twelve_bit, float_range, shape_is_width_height)
        specific_property_signals = (
            self.objectNameChanged,
            self.auto_min_max_enabled_changed,
            self.min_changed,
            self.max_changed,
            self.gamma_changed,
            self.trilinear_filtering_enabled_changed,
            self.auto_getcolor_expression_enabled_changed,
            self.getcolor_expression_changed,
            self.extra_transformation_expression_changed)
        for specific_property_signal in specific_property_signals:
            specific_property_signal.connect(self.property_changed)

    name = property(
        Qt.QObject.objectName,
        Qt.QObject.setObjectName,
        doc='Property proxy for QObject::objectName Qt property, which is directly accessible via the objectName getter and '
            'setObjectName setter.  Upon change, objectNameChanged is emitted.')

    # A change to any mutable property potentially impacts image presentation.  For convenience, property_changed is emitted whenever
    # any of the more specific mutable-property-changed signals are emitted.
    # 
    # For example:
    # immutable_image_with_mutable_properties_instance.property_changed.connect(refresh)
    # Instead of:
    # immutable_image_with_mutable_properties_instance.min_changed.connect(refresh)
    # immutable_image_with_mutable_properties_instance.max_changed.connect(refresh)
    # immutable_image_with_mutable_properties_instance.gamma_changed.connect(refresh)
    # immutable_image_with_mutable_properties_instance.trilinear_filtering_enabled_changed.connect(refresh)
    # immutable_image_with_mutable_properties_instance.auto_getcolor_expression_enabled_changed.connect(refresh)
    # immutable_image_with_mutable_properties_instance.getcolor_expression_changed.connect(refresh)
    # immutable_image_with_mutable_properties_instance.extra_transformation_expression_changed.connect(refresh)
    #
    # In the __init__ function of any ImmutableImageWithMutableProperties that adds presentation-affecting properties
    # and associated change notification signals, do not forget to connect the subclass's change signals to property_changed.
    property_changed = Qt.pyqtSignal()

    def _auto_min_max_post_set_callback(self, v):
        if v:
            self.do_auto_min_max()

    def _min_max_default(self, is_min):
        return 

    properties = []
    auto_min_max_enabled = _Property(properties, 'auto_min_max_enabled', lambda: True, bool, None, self._auto_min_max_post_set_callback)
    min = _Property(properties, 'min', )
    auto_min_max = 
    auto_min_max_enabled = Qt.pyqtSignal()
    min_changed = Qt.pyqtSignal()
    max_changed = Qt.pyqtSignal()
    gamma_changed = Qt.pyqtSignal()
    trilinear_filtering_enabled_changed = Qt.pyqtSignal()
    auto_getcolor_expression_enabled_changed = Qt.pyqtSignal()
    getcolor_expression_changed = Qt.pyqtSignal()
    extra_transformation_expression_changed = Qt.pyqtSignal()

    @property
    def auto_min_max_enabled(self):
        return self._auto_min_max_enabled

    @auto_min_max_enabled.setter
    def auto_min_max_enabled(self, v):
        v = bool(v)
        if v != self._auto_min_max_enabled:

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, v):

    _previous_anon_name_timestamp = None
    _previous_anon_name_timestamp_dupe_count = None
    @staticmethod
    def _generate_anon_name():
        timestamp = time.time()
        if timestamp == Image._previous_anon_name_timestamp:
            Image._previous_anon_name_timestamp_dupe_count += 1
        else:
            Image._previous_anon_name_timestamp = timestamp
            Image._previous_anon_name_timestamp_dupe_count = 0
        name = str(timestamp)
        if Image._previous_anon_name_timestamp_dupe_count > 0:
            name += '-{:04}'.format(Image._previous_anon_name_timestamp_dupe_count)
        name += ' ({})'.format(datetime.datetime.fromtimestamp(timestamp).strftime('%c'))
        return name
