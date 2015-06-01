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

class ImmutableImageWithMutableProperties(ImmutableImage, Qt.QObject):
    """ImmutableImageWithMutableProperties contains immutable (read-only) image data and metadata inherited from ImmutableImage
    plus modifyable (mutable) properties describing """
    def __init__(self, data, is_twelve_bit=False, float_range=None, shape_is_width_height=True, name=None, parent=None):
        Qt.QObject.__init__(self, parent)
        if name is None:
            name = self._generate_anon_name()
        self.setObjectName(name)
        ImmutableImage.__init__(self, data, is_twelve_bit, float_range, shape_is_width_height)

    name = property(
        Qt.QObject.objectName,
        Qt.QObject.setObjectName,
        doc='Property proxy for QObject.objectName Qt property.  Upon change, objectNameChanged is emitted.')

    min_changed = Qt.pyqtSignal()
    max_changed = Qt.pyqtSignal()
    gamma_changed = Qt.pyqtSignal()
    trilinear_filtering_enabled_changed = Qt.pyqtSignal()
    auto_getcolor_expression_enabled_changed = Qt.pyqtSignal()
    getcolor_expression_changed = Qt.pyqtSignal()
    extra_transformation_expression_changed = Qt.pyqtSignal()



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
