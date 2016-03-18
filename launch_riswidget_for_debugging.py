#!/usr/bin/env python

import numpy
from pathlib import Path
from PyQt5 import Qt
from ris_widget.ris_widget import RisWidget
from ris_widget.point_list_picker import PointListPicker
from ris_widget.examples.simple_point_picker import SimplePointPicker
from ris_widget.examples.simple_poly_line_point_picker import SimplePolyLinePointPicker
from ris_widget.image import Image
from ris_widget.layer import Layer
import freeimage
import sys
import gc

#if sys.platform == 'darwin':
#    im = freeimage.read('/Volumes/scopearray/pharyngeal_pumping_max_fps/pharyngeal_pumping_max_fps_010_0.274411275.png')
#   im = freeimage.read('/Users/ehvatum/zplrepo/ris_widget/top_left_g.png')
#elif sys.platform == 'linux':
#   im = freeimage.read('/home/ehvatum/2048.png')
#   im = freeimage.read('/mnt/scopearray/pharyngeal_pumping_max_fps/pharyngeal_pumping_max_fps_010_0.274411275.png')
#    im = freeimage.read('/home/ehvatum/potw1509a.jpg').astype(numpy.float32)
#elif sys.platform == 'win32':
#    im = freeimage.read('C:/zplrepo/ris_widget/top_left_g.png')

argv = sys.argv
#Qt.QCoreApplication.setAttribute(Qt.Qt.AA_ShareOpenGLContexts)
app = Qt.QApplication(argv)
rw = RisWidget()
rw.show()

rw.add_image_files_to_flipbook([Path(__file__).parent / 'Opteron_6300_die_shot_16_core_mod.jpg'])
# rw.qt_object.layer_stack.histogram_alternate_column_shading_enabled = True
# rw.layer.histogram_min = 0
# rw.layer.histogram_max = 1
# rw.image = imf
#
# btn = Qt.QPushButton('swap float range setting')
# float_range_state = False
# def on_btn_clicked():
#     global float_range_state
#     rw.image.set(imposed_float_range=[0,255] if float_range_state else [50,100])
#     float_range_state = not float_range_state
# btn.clicked.connect(on_btn_clicked)
# btn.show()

#rw.histogram_view.gl_widget.start_logging()

from ris_widget.qwidgets.flipbook_page_annotator import FlipbookPageAnnotator
from ris_widget.examples.flipbook_page_poly_line_annotator import FlipbookPagePolyLineAnnotator, PolyLinePointPicker
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
fpa.show()

app.exec_()