# The MIT License (MIT)
#
# Copyright (c) 2016 WUSTL ZPLAB
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

from ..qwidgets.flipbook_page_annotator import _BaseField, FlipbookPageAnnotator
from ..qwidgets.point_list_picker_table import PointListPickerTable
from .poly_line_point_picker import PolyLinePointPicker

class _PointListField(_BaseField):
    def __init__(self, field_tuple, parent):
        self.layer_stack_item = field_tuple[3]
        self.general_view = field_tuple[4]
        super().__init__(field_tuple, parent)

    def __del__(self):
        try:
            self.layer_stack_item.scene().removeItem(self.picker)
        except (AttributeError, RuntimeError):
            pass

    def _init_widget(self):
        self.picker = PolyLinePointPicker(self.general_view, self.layer_stack_item)
        self.widget = PointListPickerTable(self.picker)
        self.picker.point_list_contents_changed.connect(self._on_widget_change)

    def _on_widget_change(self):
        super()._on_widget_change()

    def value(self):
        return list(self.picker.points)

    def refresh(self, value):
        self.picker.points = value

class FlipbookPagePolyLineAnnotator(FlipbookPageAnnotator):
    """Ex:

    from ris_widget.ris_widget import RisWidget
    from ris_widget.examples.flipbook_page_poly_line_annotator import PolyLinePointPicker, FlipbookPagePolyLineAnnotator
    import numpy
    rw = RisWidget()
    xr = numpy.linspace(0, 2*numpy.pi, 65536, True)
    xg = xr + 2*numpy.pi/3
    xb = xr + 4*numpy.pi/3
    im = (((numpy.dstack(list(map(numpy.sin, (xr, xg, xb)))) + 1) / 2) * 65535).astype(numpy.uint16)
    rw.flipbook_pages.append(im.swapaxes(0,1).reshape(256,256,3))
    fpa = FlipbookPagePolyLineAnnotator(
        rw.flipbook,
        'annotation',
        (
            ('foo', str, 'default_text'),
            ('bar', int, -11, -20, 35),
            ('baz', float, -1.1, -1000, 1101.111),
            ('choice', tuple, 'za', list('aaaa basd casder eadf ZZza aasdfer lo ad bas za e12 1'.split())),
            ('toggle', bool, False),
            ('line_points', PolyLinePointPicker.POINT_LIST_TYPE, [(10,100),(100,10)], rw.main_scene.layer_stack_item, rw.main_view)
        )
    )
    fpa.show()"""

    TYPE_FIELD_CLASSES = FlipbookPageAnnotator.TYPE_FIELD_CLASSES.copy()
    TYPE_FIELD_CLASSES[PolyLinePointPicker.POINT_LIST_TYPE] = _PointListField

    def closeEvent(self, e):
        try:
            del self.fields
        except AttributeError:
            pass