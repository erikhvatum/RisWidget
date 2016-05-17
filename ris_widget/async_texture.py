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
from contextlib import ExitStack
import enum
import OpenGL
import OpenGL.GL as PyGL
from PyQt5 import Qt
import queue
import threading
import warnings
import weakref
from . import shared_resources

class AsyncTextureState(enum.Enum):
    NotUploaded = 0
    Uploading = 1
    Uploaded = 2
    UploadFailed = 3
    Bound = 4

class AsyncTexture:
    pixel_transfer_opts = Qt.QOpenGLPixelTransferOptions()
    pixel_transfer_opts.setAlignment(1)

    def __init__(self, data, format, source_format, source_type, upload_immediately):
        self.data = data
        self.format = format
        self.source_format = source_format
        self.source_type = source_type
        self.bottle = _AsyncTextureBottle(self)
        weakref.finalize(self, _texture_cache.on_async_texture_finalized, self.bottle)
        self.lock = threading.Lock()
        self.state_cv = threading.Condition(self.lock)
        self.bound_tmu = None
        if upload_immediately:
            self._state = AsyncTextureState.Uploading
            _texture_cache.upload(self)
        else:
            self._state = AsyncTextureState.NotUploaded

    @property
    def state(self):
        with self.lock:
            return self._state

    @property
    def tex(self):
        return self.bottle.tex

    @tex.setter
    def tex(self, v):
        self.bottle.tex = v

    def upload(self):
        with self.state_cv:
            if self._state not in (AsyncTextureState.NotUploaded, AsyncTextureState.UploadFailed):
                return
            self._state = AsyncTextureState.Uploading
            _texture_cache.upload(self)
            self.state_cv.notify_all()

    def bind(self, tmu, exit_stack):
        with self.state_cv:
            assert self._state != AsyncTextureState.Bound and self.bound_tmu is None
            if self._state == AsyncTextureState.UploadFailed:
                self._state = AsyncTextureState.NotUploaded
                self.state_cv.notify_all()
            while self._state != AsyncTextureState.Uploaded:
                if self._state == AsyncTextureState.NotUploaded:
                    self._state = AsyncTextureState.Uploading
                    _texture_cache.upload(self)
                    self.state_cv.notify_all()
                elif self._state == AsyncTextureState.UploadFailed:
                    if hasattr(self, '_upload_exception'):
                        raise self._upload_exception
                    raise RuntimeError('Texture upload failed for unknown reasons.')
                self.state_cv.wait()
            self.tex.bind(tmu, Qt.QOpenGLTexture.DontResetTextureUnit)
            exit_stack.callback(self._release)
            self.bound_tmu = tmu
            _texture_cache.on_async_texture_bound(self)
            self._state = AsyncTextureState.Bound
            self.state_cv.notify_all()

    def _release(self):
        """Do not call this method directly.  The responsibility for calling ._release at the appropriate time should be left with
        the ExitStack instance provided as the exit_stack argument to .bind, since .bind adds a ._release call to exit_stack."""
        with self.state_cv:
            assert self._state == AsyncTextureState.Bound and self.bound_tmu is not None
            self.tex.release(self.bound_tmu)
            self.bound_tmu = None
            _texture_cache.on_async_texture_released(self)
            self._state = AsyncTextureState.Uploaded
            self.state_cv.notify_all()

class _AsyncTextureBottle:
    def __init__(self, async_texture):
        self.async_texture_wr = weakref.ref(async_texture)
        self.tex = None

class _AsyncTextureUploadThread(Qt.QThread):
    def __init__(self, texture_cache, offscreen_surface):
        super().__init__()
        self.texture_cache = texture_cache
        self.offscreen_surface = offscreen_surface

    def run(self):
        gl_context = Qt.QOpenGLContext()
        gl_context.setShareContext(Qt.QOpenGLContext.globalShareContext())
        gl_context.setFormat(shared_resources.GL_QSURFACE_FORMAT())
        if not gl_context.create():
            raise RuntimeError('Failed to create OpenGL context for background texture upload thread.')
        gl_context.makeCurrent(self.offscreen_surface)
        PyGL.glPixelStorei(PyGL.GL_UNPACK_ALIGNMENT, 1)
        texture_cache = self.texture_cache
        async_texture_bottle = texture_cache.work_queue.get()
        try:
            while async_texture_bottle is not None:
                async_texture = async_texture_bottle.async_texture_wr()
                if async_texture is not None:
                    assert async_texture._state == AsyncTextureState.Uploading and async_texture.tex is None
                    if Qt.QThread.currentThread() is not gl_context.thread():
                        warnings.warn(
                            '_AsyncTextureUploadThread somehow managed to have its gl_context migrate to another thread, which '
                            'makes no sense and should never happen.')
                    try:
                        async_texture.tex = tex = Qt.QOpenGLTexture(Qt.QOpenGLTexture.Target2D)
                        tex.setFormat(async_texture.format)
                        tex.setWrapMode(Qt.QOpenGLTexture.ClampToEdge)
                        tex.setMipLevels(6)
                        tex.setAutoMipMapGenerationEnabled(True)
                        data = async_texture.data
                        tex.setSize(data.shape[0], data.shape[1], 1)
                        tex.allocateStorage()
                        tex.setMinMagFilters(Qt.QOpenGLTexture.LinearMipMapLinear, Qt.QOpenGLTexture.Nearest)
                        tex.bind()
                        try:
                            PyGL.glTexSubImage2D(
                                PyGL.GL_TEXTURE_2D, 0, 0, 0, data.shape[0], data.shape[1],
                                async_texture.source_format,
                                async_texture.source_type,
                                memoryview(data.T.flatten()))
                        finally:
                            tex.release()
                        texture_cache.on_upload_completion_in_upload_thread(async_texture)
                    except Exception as e:
                        async_texture._upload_exception = e
                        with async_texture.state_cv:
                            async_texture._state = AsyncTextureState.UploadFailed
                            async_texture.state_cv.notify_all()
                async_texture_bottle = texture_cache.work_queue.get()
            self.texture_cache = None
        finally:
            gl_context.doneCurrent()

_texture_cache = None

class _TextureCache(Qt.QObject):
    ASYNC_TEXTURE_UPLOAD_THREAD_COUNT = 4
    # Whenever the .apply_cache_constraint() method is called or an entry is appended to .lru_cache, the oldest entries in
    # .lru_cache are destroyed until either .lru_cache is empty or an additional constraint is met.  Which additional constraint
    # applies depends on the the host environment:
    # MIN_FREE_GPU_MEMORY_PORTION has an effect if the GL_NVX_gpu_memory_info extension is available
    MIN_FREE_GPU_MEMORY_PORTION = 0.25
    # MAX_LRU_CACHE_KIBIBYTES has an effect if the GL_NVX_gpu_memory_info extension is not available, and defaults to 128MiB.
    # Actual memory used is typically a multiple of this value - thus the conservative default.
    MAX_LRU_CACHE_KIBIBYTES = 128 << 10

    @staticmethod
    def init():
        global _texture_cache
        if _texture_cache is None:
            _texture_cache = _TextureCache()

    def __init__(self):
        super().__init__()
        glsf = shared_resources.GL_QSURFACE_FORMAT()
        if shared_resources.NVX_GPU_MEMORY_INFO_AVAILABLE:
            self._apply_constraint = self._apply_constraint_NV
        else:
            self._apply_constraint = self._apply_constraint_plain
        self.offscreen_surface = Qt.QOffscreenSurface()
        self.offscreen_surface.setFormat(glsf)
        self.offscreen_surface.create()
        self.gl_context = Qt.QOpenGLContext()
        self.gl_context.setShareContext(Qt.QOpenGLContext.globalShareContext())
        self.gl_context.setFormat(glsf)
        if not self.gl_context.create():
            raise RuntimeError('Failed to create OpenGL context for TextureCache.')
        self.work_queue = queue.Queue()
        # lru_cache: _AsyncTextureBottles of currently uploaded, non-bound textures in descending order of time since last
        # use (least recently used at lru_cache[0] and most recently used at lru_cache[-1]).
        self.lru_cache = collections.deque()
        self.lru_cache_lock = threading.Lock()
        self.async_texture_upload_threads = []
        for _ in range(_TextureCache.ASYNC_TEXTURE_UPLOAD_THREAD_COUNT):
            # Note that Qt docs state: "Note: Due to the fact that QOffscreenSurface is backed by a QWindow on some platforms,
            # cross-platform applications must ensure that create() is only called on the main (GUI) thread. The
            # QOffscreenSurface is then safe to be used with makeCurrent() on other threads, but the initialization and
            # destruction must always happen on the main (GUI) thread."
            offscreen_surface = Qt.QOffscreenSurface()
            offscreen_surface.setFormat(glsf)
            offscreen_surface.create()
            upload_thread = _AsyncTextureUploadThread(self, offscreen_surface)
            upload_thread.start()
            self.async_texture_upload_threads.append(upload_thread)
        Qt.QApplication.instance().aboutToQuit.connect(self.shut_down)

    # def __del__(self):
    #     try:
    #         self.shut_down()
    #     except (AttributeError, RuntimeError, TypeError):
    #         pass

    def append_async_texture_to_lru_cache(self, async_texture):
        with self.lru_cache_lock:
            self.lru_cache.append(async_texture.bottle)
            self._apply_constraint()

    def on_upload_completion_in_upload_thread(self, async_texture):
        # Called from async texture upload thread with the requirement that a GL context is current in that thread
        with async_texture.state_cv:
            self.append_async_texture_to_lru_cache(async_texture)
            async_texture._state = AsyncTextureState.Uploaded
            async_texture.state_cv.notify_all()

    def on_async_texture_bound(self, async_texture):
        with self.lru_cache_lock:
            self.lru_cache.remove(async_texture.bottle)

    def on_async_texture_released(self, async_texture):
        self.append_async_texture_to_lru_cache(async_texture)

    def upload(self, async_texture):
        assert async_texture.bottle not in self.work_queue.queue
        self.work_queue.put(async_texture.bottle)

    def apply_constraint(self):
        with self.lru_cache_lock:
            self._apply_constraint()

    def _apply_constraint_plain(self):
        lru_cache = self.lru_cache
        KiB = sum(atb.async_texture_wr().data.nbytes for atb in lru_cache) >> 10
        while KiB > _TextureCache.MAX_LRU_CACHE_KIBIBYTES and lru_cache:
            KiB -= lru_cache[0].async_texture_wr().data.nbytes >> 10
            self._pop_left()

    def _apply_constraint_NV(self):
        with ExitStack() as estack:
            if Qt.QOpenGLContext.currentContext() is None:
                self.gl_context.makeCurrent(self.offscreen_surface)
                estack.callback(self.gl_context.doneCurrent)
            import OpenGL.GL.NVX.gpu_memory_info as MI
            lru_cache = self.lru_cache
            min_KiB = int(PyGL.glGetInteger(MI.GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX) * _TextureCache.MIN_FREE_GPU_MEMORY_PORTION)
            while PyGL.glGetInteger(MI.GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX) < min_KiB and lru_cache:
                self._pop_left()

    def _pop_left(self):
        # Requirement: self.lru_cache_lock is held
        atb = self.lru_cache[0]
        at = atb.async_texture_wr()
        if at is None:
            self.lru_cache.popleft()
            atb.tex.destroy()
            atb.tex = None
        else:
            lock_acquired = at.lock.acquire(blocking=False, timeout=-1)
            # If the left-most texture in the LRU cache is locked at this juncture, there can be only one reason: the left-most texture
            # is currently being bound.  This can happen in two situations, the second of which is a generalization of the first:
            #   1) Calling async_texture_instance.bind(..) prompted uploading of a not-uploaded, not-uploading texture, and the texture
            #      cache was constrained such that it could not even nominally hold async_texture_instance.  So, adding
            #      async_texture_instance to the cache prompted attempted removal of async_texture_instance from the cache.
            #   2) While an AsyncTexture.bind(..) call was in progress, a texture upload completed, causing the uploaded texture to be
            #      added to the texture cache.  However, doing so caused the texture cache to exceed a constraint, in turn causing the
            #      AsyncTexture instance currently being bound to be targeted for destruction.
            # In both cases, a blocking lock acquisition with no timeout would result in a deadlock, and the obvious fix of moving
            # the line "self.append_async_texture_to_lru_cache(async_texture)" in _TextureCache.on_upload_completion_in_upload_thread
            # outside of the lock would result in the currently-binding texture being destroyed and then erroneously marked as uploaded.
            # The proper course of action is to not destroy the texture currently being bound.  This makes sense: if an AsyncTexture is
            # being bound, we really do want to remove it from the texture cache without destroying it.  In fact, AsyncTexture.bind does
            # just that.  So, if we could not acquire the lock, it's because binding is in progress, and we allow that ongoing bind call
            # to do the appropriate cache maintenance.
            if lock_acquired:
                self.lru_cache.popleft()
                try:
                    atb.tex.destroy()
                    atb.tex = None
                    at._state = AsyncTextureState.NotUploaded
                finally:
                    at.lock.release()

    def on_async_texture_finalized(self, async_texture_bottle):
        with self.lru_cache_lock:
            if async_texture_bottle.tex is not None:
                if Qt.QOpenGLContext.currentContext() is None and Qt.QThread.currentThread() is not self.gl_context.thread():
                    warnings.warn('_TextureCache.on_async_texture_finalized called from wrong thread.')
                    return
                with ExitStack() as estack:
                    if Qt.QOpenGLContext.currentContext() is None:
                        self.gl_context.makeCurrent(self.offscreen_surface)
                        estack.callback(self.gl_context.doneCurrent)
                    async_texture_bottle.tex.destroy()
                    async_texture_bottle.tex = None
                    async_texture = async_texture_bottle.async_texture_wr()
                    if async_texture is not None:
                        with async_texture.state_cv:
                            async_texture._state = AsyncTextureState.NotUploaded
                            async_texture.state_cv.notify_all()
            try:
                self.lru_cache.remove(async_texture_bottle)
            except ValueError:
                pass

    def shut_down(self):
        # Cancel any unstarted, queued uploads
        with self.work_queue.mutex:
            self.work_queue.queue.clear()
            self.work_queue.unfinished_tasks = 0
        # Destroy any uploaded textures, ensuring that an OpenGL context is current while doing so
        with ExitStack() as estack:
            if Qt.QOpenGLContext.currentContext() is None:
                self.gl_context.makeCurrent(self.offscreen_surface)
                estack.callback(self.gl_context.doneCurrent)
            for async_texture_bottle in self.lru_cache:
                async_texture_bottle.tex.destroy()
                async_texture_bottle.tex = None
                async_texture = async_texture_bottle.async_texture_wr()
                if async_texture is not None:
                    async_texture._state = AsyncTextureState.NotUploaded
        self.lru_cache.clear()
        # Gracefully stop all upload threads
        for thread in self.async_texture_upload_threads:
            self.work_queue.put(None)
        for thread in self.async_texture_upload_threads:
            thread.wait()
        self.async_texture_upload_threads = []