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

import collections
import enum
import numpy
from PyQt5 import QtCore, QtGui
import queue
import threading
import weakref
from .shared_resources import GL_QSURFACE_FORMAT

NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE = {
    numpy.bool8: QtGui.QOpenGLTexture.UInt8,
    numpy.uint8: QtGui.QOpenGLTexture.UInt8,
    numpy.uint16: QtGui.QOpenGLTexture.UInt16,
    numpy.float32: QtGui.QOpenGLTexture.Float32
}
IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT = {
    'G': QtGui.QOpenGLTexture.R32F,
    'Ga': QtGui.QOpenGLTexture.RG32F,
    'rgb': QtGui.QOpenGLTexture.RGB32F,
    'rgba': QtGui.QOpenGLTexture.RGBA32F
}
IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT = {
    'G': QtGui.QOpenGLTexture.Red,
    'Ga': QtGui.QOpenGLTexture.RG,
    'rgb': QtGui.QOpenGLTexture.RGB,
    'rgba': QtGui.QOpenGLTexture.RGBA
}

class AsyncTextureState(enum.Enum):
    Incomplete = 0
    Uploading = 1
    Uploaded = 2
    UploadFailed = 3
    Freed = 4

class AsyncTexture:
    def __init__(self, texture_manager):
        self.texture_manager = texture_manager
        self._state = AsyncTextureState.NotUploaded
        self._state_cv = threading.Condition()
        self._texture_id = None

    def __del__(self):


    @property
    def state(self):
        with self._state_cv:
            return self._state

    def texture_id(self):
        with self._state_cv:
            while self._state != AsyncTextureState.Uploaded:
                assert self._state != AsyncTextureState.Incomplete
                if self._state == AsyncTextureState.Freed:
                    self.texture_manager.upload(weakref.ref(self))
                elif self._state == AsyncTextureState.UploadFailed:
                    if hasattr(self, '_upload_exception'):
                        raise self._upload_exception
                    raise RuntimeError('Texture upload failed for unknown reasons.')
                self._state_cv.wait()
        self.texture_manager.bump(weakref.ref(self))
        return self._texture_id

class AsyncTextureUploadThread(QtCore.QThread):
    def __init__(self, work_queue):
        super().__init__()
        self.work_queue = work_queue

    def run(self):
        glsf = GL_QSURFACE_FORMAT()
        offscreen_surface = QtGui.QOffscreenSurface()
        offscreen_surface.setFormat(glsf)
        offscreen_surface.create()
        tex = self.work_queue.get()
        while tex is not None:

            tex = self.work_queue.get()

class TextureManager(QtCore.QObject):
    ASYNC_TEXTURE_UPLOAD_THREAD_COUNT = 4

    def __init__(self):
        super().__init__()
        glsf = GL_QSURFACE_FORMAT()
        self.offscreen_surface = QtGui.QOffscreenSurface()
        self.offscreen_surface.setFormat(glsf)
        self.offscreen_surface.create()
        self.work_queue = queue.LifoQueue()
        self.async_texture_lru = collections.deque()
        self.async_texture_upload_threads = [
            AsyncTextureUploadThread(self.work_queue) for i in range(self.ASYNC_TEXTURE_UPLOAD_THREAD_COUNT)
        ]

    def bump(self, async_texture_wr):
        self.async_texture_lru.remove(async_texture_wr)
        self.async_texture_lru.append(async_texture_wr)

    def retire(self, async_texture_wr):
        self.async_texture_lru.remove(async_texture_wr)