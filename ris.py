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

from PyQt5 import QtCore, QtGui, QtWidgets, QtOpenGL

from ris_widget.ris_exceptions import *

class Ris:
    '''Rapid Image Stream.  Derive from this class to implement a new image stream
    source, and pass an instance of your class as the argument to RisWidget.attachRis(...)
    in order to display streamed images.  Note that you don't need this just to show an
    image or two.  For that, use RisWidget.showImage(..).'''
    def __init__(self, bufferCount):
        self._sinks = []
        self._bufferCount = bufferCount
        self._streamManager = None

    def attachSink(self, sink):
        pass

    def detachSink(self, sink):
        pass

    def start(self):
        pass

    def stop(self):
        pass

class _StreamWorker(QtCore.QObject):
    def __init__(self, ris):
        self.ris = ris

class _StreamManager(QtCore.QObject):
    '''Something derived from QObject and residing in the main thread must exist
    in order to receive notifications asyncronously from the _StreamWorker thread.
    The Ris class could serve this function, but that would require deriving it
    from QObject, potentially complicating things for users implementing Ris
    children.  Additionally, this threading functionality can be viewed as an
    internal implementation detail that is better abstracted away.  This way,
    Ris appears to provide a clean callback-style interface whose callbacks
    are smarter than your average callback in that they are actually handlers
    executed by the main thread's event loop function and run in the main thread.'''

    # It is through these signals that events in the stream thread are transported into
    # the main thread.
    workerStoppedSignal = QtCore.pyqtSignal()
    newImageSignal = QtCore.pyqtSignal(int)
    exceptionSignal = QtCore.pyqtSignal(list)

    def __init__(self, ris):
        super().__init__(None)
        self.streamWorker = _StreamWorker(ris)
        # Refer to the first example in the "detailed description" section of
        # http://qt-project.org/doc/qt-5/qthread.html for information on exactly
        # what is going on in the next few lines and why.
        self.streamWorkerThread = QtCore.QThread(self)
        # Equivalent to the connect(workerThread, &QThread::finished, worker, &QObject::deleteLater);
        # line in the link above.
        self.streamWorkerThread.finished.connect(self.streamWorkerThread.deleteLater, QtCore.Qt.QueuedConnection)
