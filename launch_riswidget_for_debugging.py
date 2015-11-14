#!/usr/bin/env python

import numpy
from pathlib import Path
from PyQt5 import Qt
from ris_widget.ris_widget import RisWidget
from ris_widget.image import Image
from ris_widget.layer import Layer
import freeimage
import sys

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
#rw.image = im

#image_fpath_stacks = [(str(bf_fpath), str(gfp_fpath)) for bf_fpath, gfp_fpath in [(Path(gfp_fpath.parent / (gfp_fpath.stem[:15] + ' bf.png')), gfp_fpath) for gfp_fpath in sorted(Path('/mnt/bulkdata/2015.10.08_ZPL8Prelim/00').glob('2015* gfp.png'))] if bf_fpath.exists()]
#rw.flipbook.add_image_file_stacks(image_fpath_stacks[:5])

#rw.histogram_view.gl_widget.start_logging()

app.exec_()
import gc
del rw
gc.collect()