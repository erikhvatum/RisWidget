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

import numpy
from PyQt5 import Qt
from ..image import Image
from ..shared_resources import UNIQUE_QGRAPHICSITEM_TYPE

class ImagePainterItem(Qt.QGraphicsObject):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()

    def __init__(self, image=None, parent_item=None):
        """Painting modifies an internal buffer, ._qimage_ndarray.  In order to propagate changes back to .image, call the
        .update_image method."""
        super().__init__(parent_item)
        self._image = None
        self._qimage_ndarray = None
        self._qimage_ndarray_dirty = False
        self._qimage = None
        self.image = image

    def boundingRect(self):
        return Qt.QRectF() if self._image is None else Qt.QRectF(0, 0, *self._image.shape[:2])

    def paint(self, qpainter, option, widget):
        if self._ndarray is None:
            return
        try:
            r = option.rect()
        except AttributeError:
            r = self.boundingRect()
        qpainter.drawImage(r, self._qimage, r)

    def update_image(self):
        image = self._image
        if image is None or not self._qimage_ndarray_dirty:
            self._qimage_ndarray_dirty = False
            return

        if image.dtype in (bool, numpy.uint8):
            offset = 0
            scale = 1
        elif image.dtype == numpy.uint16:
            offset = 0
            scale = 256
        else:
            offset = image.range[0]
            scale = (image.range[1] - image.range[0]) / 255

        if image.num_channels in (1,3):
            image.data[...] = offset + self._qimage_ndarray[...] * scale
        # else:
        #     qimage_non_premult = self._qimage.
        elif image.num_channels == 2:
            image.data[..., 0] = offset + (self._qimage_ndarray[..., 0] / self._qimage_ndarray[..., 3]) * scale
            image.data[..., 1] = offset + self._qimage_ndarray[..., 3] * scale
        else:
            image.data[..., 0] = offset + (self._qimage_ndarray[..., 0] / self._qimage_ndarray[..., 3]) * scale
            image.data[..., 1] = offset + (self._qimage_ndarray[..., 1] / self._qimage_ndarray[..., 3]) * scale
            image.data[..., 2] = offset + (self._qimage_ndarray[..., 2] / self._qimage_ndarray[..., 3]) * scale
            image.data[..., 3] = offset + self._qimage_ndarray[..., 3] * scale
        self._qimage_ndarray_dirty = False

    @property
    def image(self):
        self.update_image()
        return self._image

    @image.setter
    def image(self, image):
        if image is self._image:
            return
        self.prepareGeometryChange()
        self._qimage = None
        self._qimage_ndarray = None
        self._image = None
        if image is None:
            self._qimage_ndarray_dirty = False
            self.update()
            return
        assert isinstance(image, Image)
        if image.dtype in (bool, numpy.uint8):
            offset = 0
            scale = 1
        elif image.dtype == numpy.uint16:
            offset = 0
            scale = 1/256
        else:
            offset = -image.range[0]
            scale = 255 / (image.range[1] - image.range[0])

        if image.num_channels in (1, 3):
            image.data[...] = (offset + self._qimage_ndarray[...]) * scale
        elif image.num_channels == 2:
            image.data[..., 0] = (offset + self._qimage_ndarray[..., 0]) * self._qimage_ndarray[..., 3] * scale
            image.data[..., 1] = (offset + self._qimage_ndarray[..., 3]) * scale
        else:
            image.data[..., 0] = (offset + self._qimage_ndarray[..., 0]) * self._qimage_ndarray[..., 3] * scale
            image.data[..., 1] = (offset + self._qimage_ndarray[..., 1]) * self._qimage_ndarray[..., 3] * scale
            image.data[..., 2] = (offset + self._qimage_ndarray[..., 2]) * self._qimage_ndarray[..., 3] * scale
            image.data[..., 3] = (offset + self._qimage_ndarray[..., 3]) * scale
        self._qimage_ndarray_dirty = True
        self._image = image
        self.update()