#!/usr/bin/env python

import numpy
from pathlib import Path
from PyQt5 import Qt
from ris_widget.ris_widget import RisWidget
# from ris_widget.point_list_picker import PointListPicker
# from ris_widget.examples.simple_point_picker import SimplePointPicker
# from ris_widget.examples.simple_poly_line_point_picker import SimplePolyLinePointPicker
from ris_widget.image import Image
# from ris_widget.layer import Layer
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
# rw.show()

# imf = freeimage.read('/home/ehvatum/zplrepo/ris_widget/2016-02-11t1418 bf.tiff')
# image = Image(imf)
# print(image)

# im = freeimage.read('/home/ehvatum/m1_hubble.jpg')
# rw.image = im
# try:
#     rw.image = im.astype(numpy.float64)
# except Exception as e:
#     print(e)

# rw.image = im

# simple_point_picker = SimplePolyLinePointPicker(rw.main_view, rw.main_scene.layer_stack_item)

# point_list_picker, point_list_table_view = rw.make_poly_line_picker_and_table()
#
# points = [
#     [1256.0, 675.0],
#     [1541.0, 738.0],
#     [1786.0, 995.0],
#     [1289.0, 1360.0],
#     [1001.0, 1112.0],
#     [1018.0, 899.0],
#     [815.0, 685.0],
#     [858.0, 959.0],
#     [824.0, 1315.0],
#     [1027.0, 1456.0],
#     [961.0, 1728.0],
#     [1288.0, 1823.0],
#     [1294.0, 1703.0],
#     [1703.0, 1889.0],
#     [1634.0, 1593.0],
#     [1745.0, 1668.0],
#     [1761.0, 1425.0],
#     [1948.0, 1594.0],
#     [1910.0, 1323.0],
#     [2066.0, 1375.0],
#     [2006.0, 1120.0]
# ]
# point_list_picker.points = points

# def on_debug_gc():
#     rw.image = None
#     rw.flipbook_pages = None
#     gc.set_debug(gc.DEBUG_LEAK)
#     gc.collect()
#     print(gc.garbage)
#     gc.set_debug(0)
#     # objgraph.show_backrefs(objgraph.by_type('Task'))
#     # objgraph.show_refs(objgraph.by_type('Task'))
#
# btn = Qt.QPushButton('debug gc')
# btn.clicked.connect(on_debug_gc)
# btn.show()

# def on_timer():
#     rw.image = im
#
# timer = Qt.QTimer()
# timer.setSingleShot(True)
# timer.timeout.connect(on_timer)
# timer.start(1000)

# from ris_widget.qwidgets.flipbook_TESTER import TestWidget;tw = TestWidget(rw.flipbook);tw.show()

# image_fpath_stacks = [(str(bf_fpath), str(gfp_fpath)) for bf_fpath, gfp_fpath in [(Path(gfp_fpath.parent / (gfp_fpath.stem[:15] + ' bf.png')), gfp_fpath) for gfp_fpath in sorted(Path('/mnt/scopearray/Sinha_Drew/2015.11.13_ZPL8Prelim3/03').glob('2015* gfp.png'))] if bf_fpath.exists()]
# rw.flipbook.add_image_file_stacks(image_fpath_stacks[:10])

#rw.histogram_view.gl_widget.start_logging()

app.exec_()