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

import concurrent.futures as futures
from PyQt5 import Qt
from ..om import SignalingList

#class Task:
#   __slots__ = ('func', 'va', 'kw')
#   def __init__(self, *va, **kw):
#       self.va = va
#       self.kw = kw

class ProgressThreadPool(Qt.QWidget):
    """Signals:
    task_completed(task)"""
    _TASK_COMPLETED_EVENT = Qt.QEvent.registerEventType()
    task_completed = Qt.pyqtSignal(object)

    def __init__(self, tasks, parent=None, show_cancel=True):
        super().__init__(parent)
        self._next_task_id = 0
        self._tasks = None
        self.tasks = tasks

    @property
    def tasks(self):
        return self._tasks

    @tasks.setter
    def tasks(self, v):
        """Note: Only the main (GUI) thread should modify .tasks."""
        # If v is not a SignalingList and also is missing at least one list modification signal that we need, convert v
        # to a SignalingList
        if self._tasks is not None:

        if not isinstance(v, SignalingList) and any(not hasattr(v, signal) for signal in ('inserted', 'removed', 'replaced')):
            v = SignalingList(v)
        v.inserted.connect(self._on_inserted)
        v.removed.connect(self._on_removed)
        v.replaced.connect(self._on_replaced)

    def event(self, event):
        if event.type() == ProgressThreadPool._TASK_COMPLETED_EVENT:
            self._on_task_completed_event()

class _task_obj:

class _task_completed_event(Qt.QEvent):
    def __init__(self):
        super().__init__(ProgressThreadPool._TASK_COMPLETED_EVENT)
