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

import scipy
import scipy.optimize
import scipy.ndimage
import threading
import time

from ris_widget.ris import Ris
from ris_widget.ris_exceptions import *
from ris_widget.size import Size

class MicroManagerFocusSink:
    def __init__(self, mmc, ris):
        self.mmc = mmc
        self.lock = threading.RLock()
        self.newImageReadyCondVar = threading.Condition(self.lock)
        with self.lock:
            self.ris = ris
            self.newImageReady = False
            self.newImage = None
            self.focusThread = None
            self.wantStop = False
            self.ris.attachSink(self)

    def __del__(self):
        self.stopFocus()

    def startFocus(self):
        with self.lock:
            if self.focusThread is not None:
                raise FocusAlreadyInProgress()
            self.focusThread = _FocusThread(self)
            self.wantStop = False
            self.newImage = None
            self.newImageReady = False
            self.focusThread.start()

    def stopFocus(self):
        with self.lock:
            self.wantStop = True
        self.focusThread.join()
        self.focusThread = None

    def risImageAcquired(self, ris, image):
        # Multiple images may replace the current new image while the focus thread is processing an image.  This
        # is ok; the important things are that when the focus thread is ready for a new image, self.newImage is
        # the most recent of the images acquired and that self.newImage is a different image than processed by the
        # previous focus.
        with self.lock:
            if self.focusThread is not None:
                self.newImage = image
                self.newImageReady = True
                self.newImageReadyCondVar.notify()

class _FocusThread(threading.Thread):
    def __init__(self, mmfs):
        super().__init__()
        self.mmfs = mmfs

    def getNextImage(self):
        with self.mmfs.lock:
            while not self.mmfs.wantStop and not self.mmfs.newImageReady:
                self.mmfs.newImageReadyCondVar.wait()
            if self.mmfs.wantStop:
                raise FocusAborted
            self.mmfs.newImageReady = False
            return self.mmfs.newImage

    def evaluate(self, position):
        self.mmfs.mmc.setPosition('FocusDrive', position)
        image = self.getNextImage()
        ret = -scipy.ndimage.generic_gradient_magnitude(image, scipy.ndimage.sobel).mean()
        print(ret)
        return ret

    def run(self):
        try:
#           scipy.optimize.minimize_scalar(self.evaluate, bounds=(16398.577, 25749.579), method='bounded')
            scipy.optimize.brent(self.evaluate)
        except FocusAborted as e:
            return
        self.mmfs.focusThread = None
