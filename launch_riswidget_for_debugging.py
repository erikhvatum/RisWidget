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
from pathlib import Path
from PyQt5 import Qt
Qt.QApplication.setAttribute(Qt.Qt.AA_ShareOpenGLContexts)
from ris_widget.ris_widget import RisWidget
import freeimage
import re
import sys
import socket

pool = ThreadPoolExecutor()#max_workers=4)

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
pagesize = 1
pages_im_fpaths = list(im_fpaths[i*pagesize:(i+1)*pagesize] for i in range(int(len(im_fpaths)/pagesize)))
pages = list(pool.map(lambda im_fpaths: [freeimage.read(im_fpath) for im_fpath in im_fpaths], pages_im_fpaths))
del pool

# import yappi

class RawPlayer(Qt.QObject):
    show_next_page = Qt.pyqtSignal()

    def __init__(self):
        super().__init__(rw.qt_object)
        a = self.play_raw_action = Qt.QAction('\N{BLACK RIGHT-POINTING POINTER}', rw.qt_object)
        a.setCheckable(True)
        a.setChecked(False)
        a.toggled.connect(self.on_play_raw_action_toggled)
        rw.qt_object.main_view_toolbar.addAction(a)
        self.next_page_idx = 0
        self.show_next_page.connect(self.on_show_next_page, Qt.Qt.QueuedConnection)

    def on_play_raw_action_toggled(self, checked):
        if checked:
            # yappi.start(builtins=False, profile_threads=False)
            self.on_show_next_page()
        # else:
            # yappi.stop()
            # stats = yappi.get_func_stats()
            # stats.save('/home/ehvatum/zplrepo/ris_widget/launch_riswidget_for_debugging.profile.callgrind', type='callgrind')

    def on_show_next_page(self):
        if not self.play_raw_action.isChecked():
            return
        rw.layers = pages[self.next_page_idx]
        self.next_page_idx += 1
        if self.next_page_idx >= len(pages):
            self.next_page_idx = 0
        self.show_next_page.emit()

raw_player = RawPlayer()

app.exec_()
