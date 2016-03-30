#!/usr/bin/env python

import numpy
from pathlib import Path
from PyQt5 import Qt
from ris_widget.ris_widget import RisWidget
import freeimage
import sys

argv = sys.argv
#Qt.QCoreApplication.setAttribute(Qt.Qt.AA_ShareOpenGLContexts)
app = Qt.QApplication(argv)
rw = RisWidget()
# rw.main_view.zoom_preset_idx = 27

rw.image = numpy.zeros((100,100), dtype=numpy.uint8)

from ris_widget.examples.main_thread_mandelbrot import MandelbrotWidget
mandelbrot_widget = MandelbrotWidget(rw.image)

# rw.add_image_files_to_flipbook([
#     ['/Users/ehvatum/zplrepo/ris_widget/Opteron_6300_die_shot_16_core_mod.jpg', '/Users/ehvatum/zplrepo/ris_widget/top_left_g.png'],
#     ['/Volumes/MrSpinny/14/2015-11-18t0948 focus-03_ffc.png']
# ])



# from ris_widget.qwidgets.layer_stack_painter import LayerStackPainter
# lsp = LayerStackPainter(rw.main_scene.layer_stack_item)
# lsp.show()

# plp, plpt = rw.make_poly_line_picker_and_table()

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

app.exec_()