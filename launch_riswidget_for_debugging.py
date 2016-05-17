#!/usr/bin/env python

# import sys
# import numpy
# from ris_widget.ndimage_statistics.test_and_benchmark import test
# test()
#

# im = numpy.array(list(range(65536)), dtype=numpy.uint8).reshape((256,256),order='F')

# from ris_widget.ndimage_statistics.test_and_benchmark import test
# test()

# sys.exit(0)

from concurrent.futures import ThreadPoolExecutor
import numpy
import os.path
from pathlib import Path
from PyQt5 import Qt
from ris_widget.ris_widget import RisWidget
import ris_widget.async_texture
import freeimage
import re
import sys
import socket

# if socket.gethostname() == 'pincuslab-2.wucon.wustl.edu':
#     ris_widget.async_texture._TextureCache.MAX_LRU_CACHE_KIBIBYTES = 1

argv = sys.argv
app = Qt.QApplication(argv)
rw = RisWidget()

image_dpath = Path(
    {
        'pincuslab-2.wucon.wustl.edu' : '/Volumes/MrSpinny/14',
    }.get(socket.gethostname(), '/mnt/iscopearray/experiment02/0002')
)

im_fpaths = sorted(im_fpath for im_fpath in image_dpath.glob('*') if re.match('''[^.].*\.(png|tiff|tif)''', str(im_fpath)))[:200]
pagesize = 2
im_fpath_pages = list(im_fpaths[i*pagesize:(i+1)*pagesize] for i in range(int(len(im_fpaths)/pagesize)))
rw.flipbook.add_image_files(im_fpath_pages)

# pool = ThreadPoolExecutor(4)
# ims = list(pool.map(freeimage.read, sorted(Path('/mnt/iscopearray/experiment02/0002').glob('*'))[:200]))
# ims = list(ims[i*pagesize:(i+1)*pagesize] for i in range(int(len(ims)/pagesize)))
#
# rw.flipbook.pages = ims
#
# btn = Qt.QPushButton('go')
# btn.setWindowTitle('go button')
# btn.show()
# def on_btn_clicked():
#     for i in range(100):
#         for page in ims:
#             if isinstance(page, numpy.ndarray):
#                 rw.image = page
#             else:
#                 rw.layers = page
#             Qt.QApplication.processEvents()
#             if not btn.isVisible():
#                 return
# btn.clicked.connect(on_btn_clicked)

app.exec_()
