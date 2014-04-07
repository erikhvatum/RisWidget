# The MIT License (MIT)
#
# Copyright (c) 2014 Erik Hvatum
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

import threading
import time

from ris_widget.ris import Ris
from ris_widget.ris_exceptions import *
from ris_widget.size import Size

class LoopCacheSinkStream(Ris):
    '''As a sink: caches acquired images in memory.
    As a stream: outputs cached images in FiFo order, returning to the first image
    received after showing the last.'''

    def clearCache(self):
        with self.lock:
            self.imagesDataCache = []
            self.imageSize = None
            self.prevIdxSent = None

    ## Ris interface

    def __init__(self):
        super().__init__()
        self.lock = threading.RLock()
        with self.lock:
            self.ris = None
            self.imagesDataCache = []
            self.imageSize = None
            self.storeRisImages = None
            self.wantStop = True
            self.prevIdxSent = None

    def _doStart(self):
        with self.lock:
            if self.wantStop:
                self.wantStop = False
                self.acquire()

    def _doStop(self):
        with self.lock:
            self.wantStop = True

    def _doAcquire(self):
        with self.lock:
            if not self.wantStop and len(self.imagesDataCache) > 0:
                if self.prevIdxSent is None or self.prevIdxSent >= len(self.imagesDataCache) - 1:
                    self.prevIdxSent = -1
                self.prevIdxSent += 1
                self._signalImageAcquired(self.imagesDataCache[self.prevIdxSent])
#       time.sleep(0.1)

    def _imageAcquired(self, image):
        super()._imageAcquired(image)
        self.acquire()

    ## Sink interface

    def attachRis(self, ris, storeRisImages=True):
        with self.lock:
            if self.ris is not None:
                self.detachRis(self.ris)
            self.ris = ris
            self.storeRisImages = storeRisImages
            self.ris.attachSink(self)

    def detachRis(self):
        with self.lock:
            self.ris.detachSink(self)
            self.ris = None

    def risImageAcquired(self, ris, image):
        with self.lock:
            if self.storeRisImages:
                newImageSize = Size(image.shape[1], image.shape[0])
                if self.imageSize is not None:
                    if self.imageSize != newImageSize:
                        print('clearing! old: {} new: {}'.format(self.imageSize, newImageSize))
                        self.clearCache()
                        self.imageSize = newImageSize
                else:
                    self.imageSize = newImageSize
                self.imagesDataCache.append(image)

    def setStoreRisImages(self, storeRisImages):
        with self.lock:
            self.storeRisImages = storeRisImages
