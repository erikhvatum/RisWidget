#!/usr/bin/env python

import sys
import numpy
# from ris_widget.ndimage_statistics.test_and_benchmark import test
# test()
#

# im = numpy.array(list(range(65536)), dtype=numpy.uint8).reshape((256,256),order='F')
im = numpy.random.randint(0,255,(256,511),numpy.uint8).T

mask = numpy.zeros((510,256),order='F',dtype=numpy.bool)
mask[-1,0] = 1

from ris_widget.ndimage_statistics import _measures_fast as _fast_statistics
from ris_widget.ndimage_statistics import _measures_slow as _slow_statistics

fast_stats = _fast_statistics._statistics(im, False, mask)
slow_stats = _slow_statistics._statistics(im, False, mask)

print(im.shape, im.strides, mask.shape, mask.strides)
print((fast_stats.histogram == slow_stats.histogram).all())
print(fast_stats.min_max_intensity)
print(slow_stats.min_max_intensity)

sys.exit(0)

import numpy
import os.path
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

# rw.flipbook.pages.append(numpy.array(list(range(10000)),numpy.uint16).reshape((100,100)))
# rw.flipbook.pages[0][0].name = 'image'
# rw.flipbook.pages[0].append(numpy.zeros((100,10), numpy.bool))
# rw.flipbook.pages[0][1].name = 'mask'

# rw.image = numpy.zeros((100,100), dtype=numpy.uint8)

# im = freeimage.read('/Volumes/MrSpinny/14/2015-11-18t0948 focus-03_ffc.png')
# rw_dpath = Path(os.path.expanduser('~')) / 'zplrepo' / 'ris_widget'
# rw.flipbook_pages.append(im)
# mask = im > 0
# rw.qt_object.layer_stack.imposed_image_mask = mask[:781,:1000]
# rw.flipbook_pages.append(rw.qt_object.layer_stack.imposed_image_mask)
#
# rw.add_image_files_to_flipbook([
#     [rw_dpath / 'Opteron_6300_die_shot_16_core_mod.jpg', rw_dpath / 'top_left_g.png'],
#     # ['/Volumes/MrSpinny/14/2015-11-18t0948 focus-03_ffc.png']
# ])

#rw.histogram_view.gl_widget.start_logging()

app.exec_()