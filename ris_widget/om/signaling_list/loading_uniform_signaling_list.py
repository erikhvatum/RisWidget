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

from enum import Enum
from PyQt5 import Qt
from .uniform_signaling_list import UniformSignalingList

class LoaderStatus(Enum):
    New = 0       # Task is not in a UniformSignalingTaskableList
    Queued = 1    # Task is in a UniformSignalingTaskableList but has not been submitted to the thread pool
    Pooled = 2    # Task has been submitted to the thread pool, and is waiting for the thread pool to start running it
    Started = 3   # Task is running
    Completed = 4 # Task was submitted to the thread pool, started, and returned normally
    Failed = 5    # Either task was submitted to the thread pool, started, and raised an exception, or submission to the pool failed
    Cancelled = 6 # Task was cancelled before it finished

class Loader(Qt.QRunnable):
    def __init__(self, loader_element):
        self._loader_element = loader_element

class LoaderElement(Qt.QObject):
    pass

class TaskStatus(Enum):
    New = 0       # Task is not in a UniformSignalingTaskableList
    Queued = 1    # Task is in a UniformSignalingTaskableList but has not been submitted to the thread pool
    Pooled = 2    # Task has been submitted to the thread pool, and is waiting for the thread pool to start running it
    Started = 3   # Task is running
    Completed = 4 # Task was submitted to the thread pool, started, and returned normally
    Failed = 5    # Either task was submitted to the thread pool, started, and raised an exception, or submission to the pool failed
    Cancelled = 6 # Task was cancelled before it finished

class Task:
    def __init__(self, callable_, *callable_va, **callable_kw):
        self.callable_ = callable_
        self.callable_va = callable_va
        self.callable_kw = callable_kw
        self._status = TaskStatus.New
        self._progress_thread_pool = self._future = None
        self._instance_count = 0

    def _submit(self):
        assert self._progress_thread_pool is not None
        assert self._future is None
        assert Qt.QThread.currentThread() is Qt.QApplication.instance().thread()
        assert self._status == TaskStatus.Queued
        old_status = self._status
        self._status = TaskStatus.Pooled
        self._send_task_status_change_event(old_status)
        old_status = TaskStatus.Pooled
        try:
            self._future = self._progress_thread_pool._thread_pool_executor.submit(self._pool_thread_proc)
        except Exception as e:
            self._status = TaskStatus.Failed
            self._send_task_status_change_event(old_status)
            raise e
        self._future.add_done_callback(self._done_callback)

    def _cancel(self):
        if self._status in (TaskStatus.New, TaskStatus.Queued):
            self._status = TaskStatus.Cancelled
            self._send_task_status_change_event(TaskStatus.Queued)
        elif self._status in (TaskStatus.Pooled, TaskStatus.Started):
            self._future.cancel()

    def _done_callback(self, future):
        assert future is self._future
        old_status = self._status
        if future.cancelled():
            self._status = TaskStatus.Cancelled
        elif future.done():
            self._status = TaskStatus.Completed
        else:
            self._status = TaskStatus.Failed
        self._post_task_status_change_event(old_status)

    def _pool_thread_proc(self):
        self._status = TaskStatus.Started
        self._post_task_status_change_event(TaskStatus.Pooled)
        return self.callable_(*self.callable_va, **self.callable_kw)

    def _send_task_status_change_event(self, old_status):
        ptp = self._progress_thread_pool
        if ptp is not None:
            Qt.QApplication.instance().sendEvent(ptp, _TaskStatusChangeEvent(self, self.status, old_status))

    def _post_task_status_change_event(self, old_status):
        ptp = self._progress_thread_pool
        if ptp is not None:
            Qt.QApplication.instance().postEvent(ptp, _TaskStatusChangeEvent(self, self.status, old_status))

    @property
    def result(self):
        """If the following return statement raises an AttributeError, it is because this Task has not yet been pooled
        (submitted to self._progress_thread_pool._thread_pool_executor) or perhaps has not even been added to a
        UniformSignalingTaskableList."""
        return self._future.result()

    @property
    def progress_thread_pool(self):
        return self._progress_thread_pool

    @property
    def status(self):
        return self._status

_TASK_STATUS_CHANGED_EVENT = Qt.QEvent.registerEventType()

class _TaskStatusChangeEvent(Qt.QEvent):
    def __init__(self, task, new_status, old_status):
        super().__init__(_TASK_STATUS_CHANGED_EVENT)
        self.task = task
        self.new_status = new_status
        self.old_status = old_status

class UniformSignalingTaskableList(UniformSignalingList):
    pass