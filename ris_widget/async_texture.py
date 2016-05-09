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
from PyQt5 import QtCore, QtGui, QtWidgets
import queue
import threading
import weakref
from . import shared_resources

class AsyncTextureState(enum.Enum):
    Incomplete = 0
    Uploading = 1
    UploadFailed = 2
    Uploaded = 2

    Freed = 4

class AsyncTexture:
    def __init__(self, data, source_format, source_type, data_lock=None):
        self.data = data
        self.source_format = source_format
        self.source_type = source_type
        self.data_lock = threading.Lock() if data_lock is None else data_lock
        self.finalizer = weakref.finalize(self, texture_cache.remove_texture, weakref.ref(self))
        self._state = AsyncTextureState.NotUploaded
        self._state_cv = threading.Condition()
        self.tex = None

    @property
    def state(self):
        with self._state_cv:
            return self._state

    def texture_id(self):
        with self._state_cv:
            while self._state != AsyncTextureState.Uploaded:
                assert self._state != AsyncTextureState.Incomplete
                if self._state == AsyncTextureState.Freed:
                    self.texture_cache.upload(weakref.ref(self))
                elif self._state == AsyncTextureState.UploadFailed:
                    if hasattr(self, '_upload_exception'):
                        raise self._upload_exception
                    raise RuntimeError('Texture upload failed for unknown reasons.')
                self._state_cv.wait()
        self.texture_cache.bump(weakref.ref(self))
        return self.texture_id

    def bind(self, tmu, exit_stack):
        tc = TextureCache.instance()
        exit_stack.callback(lambda: self.tex.release(tmu))

class AsyncTextureUploadThread(QtCore.QThread):
    def __init__(self, work_queue):
        super().__init__()
        self.work_queue = work_queue

    def run(self):
        glsf = shared_resources.GL_QSURFACE_FORMAT()
        offscreen_surface = QtGui.QOffscreenSurface()
        offscreen_surface.setFormat(glsf)
        offscreen_surface.create()
        gl_context = QtGui.QOpenGLContext()
        gl_context.setFormat(glsf)
        if not gl_context.create():
            raise RuntimeError('Failed to create OpenGL context for background texture upload thread.')
        tex = self.work_queue.get()
        while tex is not None:
            tex = self.work_queue.get()

class TextureCache(QtCore.QObject):
    ASYNC_TEXTURE_UPLOAD_THREAD_COUNT = 4
    CACHE_SLOTS = 200
    _INSTANCE = None

    @staticmethod
    def instance():
        tmi = TextureCache._INSTANCE
        if tmi is None:
            tmi = TextureCache._INSTANCE = TextureCache(QtWidgets.QApplication.instance())
        return tmi

    def __init__(self):
        super().__init__()
        glsf = shared_resources.GL_QSURFACE_FORMAT()
        self.offscreen_surface = QtGui.QOffscreenSurface()
        self.offscreen_surface.setFormat(glsf)
        self.offscreen_surface.create()
        self.gl_context = QtGui.QOpenGLContext
        self.work_queue = queue.LifoQueue()
        # lru_cache: weakrefs to currently uploaded, non-bound (not currently used  textures
        self.lru_cache = collections.deque()
        self.async_texture_upload_threads = [
            AsyncTextureUploadThread(self.work_queue) for i in range(self.ASYNC_TEXTURE_UPLOAD_THREAD_COUNT)
        ]

    def push_texture(self, async_texture_wr):
        assert async_texture_wr not in self.lru_cache
        self.lru_cache.append(async_texture_wr)


    def remove_texture(self, async_texture_wr):
        try:
            self.lru_cache.remove(async_texture_wr)
        except ValueError:
            pass

    def bump(self, async_texture_wr):
        if self.lru_cache[-1] is not async_texture_wr:
            self.lru_cache.remove(async_texture_wr)
            self.lru_cache.append(async_texture_wr)

    def retire(self, async_texture_wr):
        self.lru_cache.remove(async_texture_wr)

    def _on_app_about_to_quit(self):
