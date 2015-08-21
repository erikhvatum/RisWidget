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

import ctypes
from PyQt5 import Qt

class _RowInstanceCounts:
    __slots__ = ('row_instance_count', 'column_instance_counts')
    self.__init__(self, row_instance_count=0, column_instance_counts=None):
        self.row_instance_count = row_instance_count
        self.column_instance_counts = {} if column_instance_counts is None else dict(column_instance_counts)

class ListTableModel(Qt.QAbstractTableModel):
    """Glue for presenting a list of lists as a table whose rows are outer list elements
    and columns are inner list elements."""
    def __init__(self, element_property_name, element_element_property_name, signaling_list=None, parent=None):
        """element_property_name: The name of the property or attribute of self._signaling_list[r] to show
        in row r, column 0.
        element_element_property_name: The name of the property or attribute of self._signaling_list[r].signaling_list[n]
        to show in row r, column n+1."""
        super().__init__(parent)
        self._signaling_list = None
        self.element_property_name = element_property_name
        self.element_element_property_name = element_element_property_name

        self._instance_counts = {}
        self.signaling_list = signaling_list

    def _on_element_property_changed():
        pass
