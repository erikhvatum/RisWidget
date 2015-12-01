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

import math
import numpy
from numpy.random import randint as R
from PyQt5 import Qt
import sip

from ..image import Image

class TestWidget(Qt.QWidget):
    IMAGE_SIZE = Qt.QSize(80, 60)

    def __init__(self, flipbook, parent=None):
        super().__init__(parent)
        self.flipbook = flipbook
        self._init_images()
        l = Qt.QVBoxLayout()
        self.add_pages_button = Qt.QPushButton('Add pages')
        self.add_pages_button.clicked.connect(self.add_pages)
        l.addWidget(self.add_pages_button)
        hl = Qt.QHBoxLayout()
        self.number_of_operations_spinner = Qt.QSpinBox()
        self.number_of_operations_spinner.setRange(1, 50000)
        self.number_of_operations_spinner.setValue(10)
        self.number_of_operations_spinner.valueChanged.connect(
            lambda: self.do_random_operations_button.setText('Do {} random operations'.format(self.number_of_operations_spinner.value())))
        hl.addWidget(self.number_of_operations_spinner)
        self.do_random_operations_button = Qt.QPushButton('Do 10 random operations')
        self.do_random_operations_button.clicked.connect(self.do_random_operations)
        hl.addWidget(self.do_random_operations_button)
        l.addLayout(hl)
        self.setLayout(l)
        self._expected_pages = [[image for image in page] for page in self.flipbook.pages]
        self._random_operations = [
            (self.randomly_delete_pages, 'randomly_delete_pages'),
            (self.randomly_delete_images, 'randomly_delete_images'),
            (self.randomly_insert_images, 'randomly_insert_images'),
            (self.randomly_insert_pages, 'randomly_insert_pages'),
            (self.randomly_merge_pages, 'randomly_merge_pages')
        ]
        self.check_state('initialization')

    def _init_images(self):
        A, = 'A'.encode('ascii')
        gs = Qt.QGraphicsScene()
        sti = Qt.QGraphicsSimpleTextItem()
        sti.setFont(Qt.QFont('Courier', pointSize=24, weight=Qt.QFont.Bold))
        gs.addItem(sti)
        self.images = []
        for char in range(A, A + 26):
            for i in range(0, 10):
                text = bytes([char]).decode('ascii') + str(i)
                sti.setText(text)
                scene_rect_f = gs.itemsBoundingRect()
                scene_rect = Qt.QRect(
                    0,
                    0,
                    math.ceil(scene_rect_f.width()),
                    math.ceil(scene_rect_f.height()))
                gs.setSceneRect(scene_rect_f)
                buffer = numpy.empty((scene_rect.height(), scene_rect.width(), 4), dtype=numpy.uint8)
                buffer[:] = 255
                qimage = Qt.QImage(sip.voidptr(buffer.ctypes.data), scene_rect.size().width(), scene_rect.size().height(), Qt.QImage.Format_RGBA8888)
                qpainter = Qt.QPainter()
                qpainter.begin(qimage)
                qpainter.setRenderHint(Qt.QPainter.Antialiasing)
                qpainter.setRenderHint(Qt.QPainter.HighQualityAntialiasing)
                gs.render(qpainter)
                qpainter.end()
                self.images.append(Image(buffer.copy(), shape_is_width_height=False, name=text))

    def add_pages(self):
        self.flipbook.pages.extend(self.images)
        self._expected_pages.extend([image] for image in self.images)
        self.check_state()

    def do_random_operations(self):
        for _ in range(self.number_of_operations_spinner.value()):
            op, op_name = self._random_operations[R(0, len(self._random_operations))]
            op()
            if not self.check_state(op_name):
                break

    def randomly_delete_pages(self):
        for _ in range(R(1, 6)):
            if not self._expected_pages:
                break
            idx = R(0, len(self._expected_pages))
            del self._expected_pages[idx]
            del self.flipbook.pages[idx]

    def randomly_delete_images(self):
        if not self._expected_pages:
            return
        page_idx = R(0, len(self._expected_pages))
        page, expected_page = self.flipbook.pages[page_idx], self._expected_pages[page_idx]
        if not expected_page:
            return
        imcount = R(1, len(expected_page)+1)
        im_idx = R(0, len(expected_page))
        del page[im_idx:(im_idx+imcount)]
        del expected_page[im_idx:(im_idx+imcount)]

    def randomly_insert_images(self):
        if not self._expected_pages:
            return
        page_idx = R(0, len(self._expected_pages))
        page, expected_page = self.flipbook.pages[page_idx], self._expected_pages[page_idx]
        images = [self.images[R(0, len(self.images))] for _ in range(R(1,6))]
        image_idx = R(0, len(expected_page)) if expected_page else 0
        page[image_idx:image_idx] = images
        expected_page[image_idx:image_idx] = images

    def randomly_insert_pages(self):
        for _ in range(R(1, 6)):
            page_idx = R(0, len(self._expected_pages)) if self._expected_pages else 0
            images = [self.images[R(0, len(self.images))] for _ in range(R(1,6))]
            self.flipbook.pages.insert(page_idx, images)
            self._expected_pages.insert(page_idx, images)

    def randomly_merge_pages(self):
        if len(self._expected_pages) < 2:
            return
        idxs = sorted(numpy.random.choice([i for i in range(len(self._expected_pages))], min(R(2,6), len(self._expected_pages)), False))
        sm = self.flipbook.pages_view.selectionModel()
        m = self.flipbook.pages_view.model()
        sm.clearSelection()
        for idx in idxs:
            sm.select(m.index(idx, 0), Qt.QItemSelectionModel.Select)
        self.flipbook.merge_selected()
        target_idx = idxs.pop(0)
        expected_target_page = self._expected_pages[target_idx]
        for idx in idxs:
            expected_target_page.extend(self._expected_pages[idx])
        for idx in reversed(idxs):
            del self._expected_pages[idx]

    def check_state(self, info=None):
        class E(Exception):
            pass
        try:
            if len(self.flipbook.pages) != len(self._expected_pages):
                raise E('len(self.flipbook.pages) != len(self._expected_pages)')
            for page_idx, page in enumerate(self.flipbook.pages):
                expected_page = self._expected_pages[page_idx]
                if len(page) != len(expected_page):
                    raise E('self.flipbook.pages[{0}] != len(self._expected_pages[{0}])'.format(page_idx))
                for image_idx, image in enumerate(page):
                    if image is not expected_page[image_idx]:
                        raise E('self.flipbook.pages[{0}][{1}] (name:"{2}") is not self._expected_pages[{0}][{1}] (name:"{3}")'.format(
                            page_idx, image_idx, image.name, expected_page[image_idx].name))
        except E as e:
            Qt.QMessageBox.warning(self, 'Test Failed{}'.format('' if info is None else (' ' + info)), str(e))
            return False
        return True