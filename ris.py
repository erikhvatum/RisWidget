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
import threading

from ris_widget.ris_exceptions import *

class Ris:
    '''Rapid Image Stream.  Derive from this class to implement a new image stream
    source, and pass an instance of your class as the argument to RisWidget.attachRis(...)
    in order to display streamed images.  Note that you don't need this just to show an
    image or two.  For that, use RisWidget.showImage(..).'''
    def __init__(self, maxStreamAheadCount=2):
        self._maxStreamAheadCount = maxStreamAheadCount
        self._currStreamAheadCount = 0
        self._currStreamAheadCountLock = threading.Lock()
        self._currStreamAheadCountDecreased = threading.Condition(self._currStreamAheadCountLock)

        # self._sinks could be a set, but maintaining sink order offers no surprises, a
        # list is fast to iterate through in order, and the only case where fast lookup
        # would be desired (adding and removing streams when a large number of streams
        # are attached) is somewhat pathological, but possible to address by deriving from
        # Ris and overriding attachSink and detachSink.
        self._sinks = []
        self._streamManager = _StreamManager(self)

    def attachSink(self, sink):
        if sink in self._sinks:
            raise DuplicateSinkException()
        self._sinks.append(sink)

    def detachSink(self, sink):
        try:
            idx = self._sinks.index(sink)
        except ValueError as e:
            raise SpecifiedSinkNotAttachedException()
        del self._sinks[idx]

    def start(self):
        self._streamManager.startStream()

    def stop(self):
        self._streamManager.stopStream()

    def _doStart(self):
        raise NotImplementedError('A class inheriting Ris must implement the _doStart member function.')

    def _doStop(self):
        raise NotImplementedError('A class inheriting Ris must implement the _doStop member function.')

    def _doAcquire(self):
        raise NotImplementedError('A class inheriting Ris must implement the _doAcquire member function.')

    def _imageAcquired(self, image):
        for sink in self._sinks:
            sink.risImageAcquired(self, image)


class _StreamWorker(QtCore.QObject):
    # The stream notifies the main thread of changes by emitting these signals.
    newImageSignal = QtCore.pyqtSignal(list)
    exceptionSignal = QtCore.pyqtSignal(list)

    def __init__(self, manager):
        super().__init__(None)
        self.manager = manager

    def startStreamSlot(self):
        self.manager.ris._doStart()
        self.loop()

    def loop(self):
        with self.manager.ris._currStreamAheadCountLock:
            if self._maxStreamAheadCount == self._currStreamAheadCount:
                self._currStreamAheadCountDecreased.wait()
        self.manager.ris._doAcquire()


    def stopStreamSlot(self):
        self.manager.ris._doStop()

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

    # These signals are used to asynchronously control the stream.  That is, a _StreamWorker
    # listens to its _StreamManager for startStreamSignal and stopStreamSignal.
    startStreamSignal = QtCore.pyqtSignal()
    stopStreamSignal = QtCore.pyqtSignal()

    def __init__(self, ris):
        super().__init__(None)
        self.ris = ris
        self.streamWorker = _StreamWorker(self)
        # Refer to the first example in the "detailed description" section of
        # http://qt-project.org/doc/qt-5/qthread.html for information on exactly
        # what is going on in the next few lines and why.
        self.streamWorkerThread = QtCore.QThread(self)
        self.streamWorker.moveToThread(self.streamWorkerThread)
        # Equivalent to the connect(workerThread, &QThread::finished, worker, &QObject::deleteLater);
        # line in the link above.
        self.streamWorkerThread.finished.connect(self.streamWorkerThread.deleteLater, QtCore.Qt.QueuedConnection)
        self.startStreamSignal.connect(self.streamWorker.startStreamSlot, QtCore.Qt.QueuedConnection)
        self.stopStreamSignal.connect(self.streamWorker.stopStreamSlot, QtCore.Qt.QueuedConnection)
        self.streamWorker.newImageSignal.connect(self.ris._imageAcquired, QtCore.Qt.QueuedConnection)
        self.streamWorkerThread.start()

    def __del__(self):
        self.streamWorker.quit()
        self.streamWorker.wait()

    def startStream(self):
        self.startStreamSignal.emit()

    def stopStream(self):
        self.stopStreamSignal.emit()
