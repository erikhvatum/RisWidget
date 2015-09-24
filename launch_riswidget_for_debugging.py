#!/usr/bin/env python

import numpy
from PyQt5 import Qt
from ris_widget.ris_widget import RisWidget
from ris_widget.image import Image
from ris_widget.layer import Layer
import freeimage
import sys

if sys.platform == 'darwin':
    im = freeimage.read('/Volumes/scopearray/pharyngeal_pumping_max_fps/pharyngeal_pumping_max_fps_010_0.274411275.png')
#   im = freeimage.read('/Users/ehvatum/zplrepo/ris_widget/top_left_g.png')
elif sys.platform == 'linux':
#   im = freeimage.read('/home/ehvatum/2048.png')
#   im = freeimage.read('/mnt/scopearray/pharyngeal_pumping_max_fps/pharyngeal_pumping_max_fps_010_0.274411275.png')
    im = freeimage.read('/home/ehvatum/potw1509a.jpg').astype(numpy.float32)
elif sys.platform == 'win32':
    im = freeimage.read('C:/zplrepo/ris_widget/top_left_g.png')

argv = sys.argv
#Qt.QCoreApplication.setAttribute(Qt.Qt.AA_ShareOpenGLContexts)
app = Qt.QApplication(argv)
rw = RisWidget()
rw.show()
rw.layer_stack = im
#rw.histogram_view.gl_widget.start_logging()

app.exec_()
import gc
del rw
gc.collect()