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
from enum import Enum
import multiprocessing
from PyQt5 import Qt
from .. import om

class Task:
    __slots__ = ('callable', 'callable_va', 'callable_kw', '_future', '_progress_thread_pool', '_prev_non_pooled', '_next_non_pooled')
    def __init__(self, callable, *callable_va, **callable_kw):
        self.callable = callable
        self.callable_va = callable_va
        self.callable_kw = callable_kw

    def _pool_thread_proc(self):
        Qt.QApplication.instance().postEvent(self._progress_thread_pool, _task_event(task=self, is_starting=True))
        try:
            r = self.callable(*self.callable_va, **self.callable_kw)
        except Exception as e:
            Qt.QApplication.instance().postEvent(self._progress_thread_pool, _task_event(task=self, is_starting=False))
            raise e
        Qt.QApplication.instance().postEvent(self._progress_thread_pool, _task_event(task=self, is_starting=False))

    @property
    def result(self):
        '''If the next line raises an AttributeError, it is because this Task has not yet been pooled (submitted to
        self._progress_thread_pool._thread_pool_executor) or perhaps has not even been added to a ProgressThreadPool.'''
        return self._future.result()

    @property
    def is_pooled(self):
        return hasattr(self, '_future')

_TASK_EVENT = Qt.QEvent.registerEventType()

class _task_event(Qt.QEvent):
    def __init__(self, task, is_starting):
        super().__init__(_TASK_EVENT)
        self.task = task
        self.is_starting = is_starting

class ProgressThreadPool(Qt.QWidget):
    """Signals:
    * task_count_changed(ProgressThreadPool)
    * task_started(Task)
    * task_done(Task)
    """
    task_count_changed = Qt.pyqtSignal(object)
    task_started = Qt.pyqtSignal(Task)
    task_done = Qt.pyqtSignal(Task)
    _x_thread_cancel = Qt.pyqtSignal()

    def __init__(self, tasks=tuple(), thread_pool_executor=None, max_workers=None, parent=None, show_cancel=True):
        '''If None is supplied for thread_pool_executor, a new concurrent.futures.ThreadPoolExecutor is created
        with max_workers.  If max_workers is None, max thread count is clamp(int(multiprocessing.cpu_count()/2),1,8)
        (IE, if None is supplied for max_workers, floor(cpu_count/2), bounded to the interval [1,8], is used).

        If a value is supplied for thread_pool_executor, the max_workers argument controls the maximum number of
        outstanding Tasks.  If you supply 2 for max_workers and futures.ThreadPoolExecutor(max_workers=8) for
        thread_pool_executor, the resulting ProgressThreadPool will only allow two of its Tasks to be in the
        thread pool at a time although the pool could ostensibly run eight in parallel.'''
        super().__init__(parent)
        self._ignore_task_list_change = False
        if not isinstance(tasks, om.SignalingList) and any(not hasattr(tasks, signal) for signal in ('inserted', 'removed', 'replaced')):
            tasks = om.SignalingList(tasks)
        for idx, task in enumerate(tasks):
            if not isinstance(task, Task):
                tasks[idx] = Task(task[0], task[1:])
        if max_workers is None:
            max_workers = max(1, int(multiprocessing.cpu_count() / 2))
            max_workers = min(8, max_workers)
        self._max_workers = max_workers
        if thread_pool_executor is None:
            thread_pool_executor = futures.ThreadPoolExecutor(max_workers)
        self._thread_pool_executor = thread_pool_executor
        l = Qt.QHBoxLayout()
        self.setLayout(l)
        self._progress_bar = Qt.QProgressBar()
        self._progress_bar.setMinimum(0)
        l.addWidget(self._progress_bar)
        self._cancel_button = Qt.QPushButton('Cancel')
        l.addWidget(self._cancel_button)
        self._cancel_button.clicked.connect(self._on_cancel)
        self._x_thread_cancel.connect(self._on_cancel, Qt.Qt.QueuedConnection)
        self._tasks = tasks
        self._non_pooled_tasks = set(tasks)
        self._done_tasks = set()
        self._started_tasks = set()
        self._first_non_pooled_task = None
        tasks.inserted.connect(self._on_inserted)
        tasks.replaced.connect(self._on_replaced)

    def submit(self, callable, *callable_va, **callable_kw):
        task = Task(callable, *callable_va, **callable_kw)
        task._progress_thread_pool = self
        self.tasks.append(task)
        return task

    def cancel(self):
        if self.thread() is Qt.QThread.currentThread():
            self._on_cancel()
        else:
            self._x_thread_cancel.emit()

    def _on_cancel(self):
        self._tasks.clear()

    @property
    def max_workers(self):
        return self._max_workers

    @property
    def tasks(self):
        assert Qt.QThread.currentThread() is Qt.QApplication.instance().thread(),\
               'ProgressThreadPool.tasks should only be accessed from the main/GUI thread.'
        return self._tasks

    @property
    def thread_pool_executor(self):
        return self._thread_pool_executor

    def event(self, event):
        if event.type() == _TASK_EVENT:
            assert isinstance(event, _task_event)
            if event.is_starting:
                self._on_task_starting(event.task)
            else:
                self._on_task_done(event.task)
            return True
        return super().event(event)

    def _on_task_starting(self, task):
        print('_on_task_starting')
        self.task_started.emit(task)

    def _on_task_done(self, task):
        self._done_tasks.add(task)
        self._update_progressbar()
        self._update_pool()
        self.task_done.emit(task)

    def _on_inserted(self, insertion_idx, tasks):
        assert all(isinstance(task, Task) for task in tasks)
        tasks = [task for task in tasks if not task.is_pooled]
        if tasks:
            if self._non_pooled_tasks:
                # Scan ._tasks list backward forward simultaneously from inserted region until a non-pooled Task is found
                # or it becomes impossible to advance both directions (which would indicate that we are in an inconsistent
                # state)
                bidx, fidx = insertion_idx - 1, insertion_idx + len(tasks)
                bE, fE = bidx < 0, fidx >= len(self._tasks)
                while not bE or not fE:
                    if not bE:
                        t = self._tasks[bidx]
                        if not t.is_pooled:
                            tn = t._next_non_pooled
                            t0 = tasks[0]
                            t1 = tasks[-1]
                            t0._prev_non_pooled = t
                            t._next_non_pooled = t0
                            t1._next_non_pooled = tn
                            if tn is not None:
                                tn._prev_non_pooled = t1
                            for ta, tb in zip(tasks, tasks[1:]):
                                ta._next_non_pooled = tb
                                tb._prev_non_pooled = ta
                            break
                        bidx -= 1
                        bE = bidx < 0
                    if not Fe:
                        t = self._tasks[fidx]
                        if not t.is_pooled:
                            tp = t._prev_non_pooled
                            t0 = tasks[0]
                            t1 = tasks[-1]
                            t1._next_non_pooled = t
                            t._prev_non_pooled = t1
                            t0._prev_non_pooled = tp
                            if tp is None:
                                # We scanned forward and the first non-pooled Task we found has no preceeding non-pooled Tasks.
                                # The inserted Tasks preceed it, making the first non-pooled, inserted Task the new non-pooled
                                # Task linked-list head.
                                if t is not self._first_non_pooled_task:
                                    raise RuntimeError('Inconsistent state: scanning forward found an non-pooled Task that has no preceeding '
                                                       'unpooled Task and yet somehow is not the non-pooled Task linked-list head.')
                                self._first_non_pooled_task = t
                            else:
                                tp._next_non_pooled = t0
                            for ta, tb in zip(tasks, tasks[1:]):
                                ta._next_non_pooled = tb
                                tb._prev_non_pooled = ta
                            break
                        fidx += 1
                        fE = fidx >= len(self._tasks)
                else:
                    raise RuntimeError(
                        'Inconsistent state: self._non_pooled_tasks is not empty, but self._tasks contains no unpooled Tasks '
                        'other than those just inserted.')
            else:
                for ta, tb in zip(tasks, tasks[1:]):
                    ta._next_non_pooled = tb
                    tb._prev_non_pooled = ta
                tasks[0]._prev_non_pooled = None
                tasks[-1]._next_non_pooled = None
                self._first_non_pooled_task = tasks[0]
            for t in tasks:
                self._non_pooled_tasks.add(t)
            self._update_progressbar()
            self._update_pool()

    def _on_replaced(self, idxs, replaced_tasks, tasks):
        assert all(isinstance(task, Task) for task in tasks)

    def _update_pool(self):
        add_task_count_to_pool = min(len(self._non_pooled_tasks) - len(self._done_tasks), self._max_workers)
        print('len(self._non_pooled_tasks)', len(self._non_pooled_tasks), 'add_task_count_to_pool', add_task_count_to_pool)
        if add_task_count_to_pool > 0:
            add_tasks = []
            task = self._first_non_pooled_task
            assert task._prev_non_pooled is None
            for _ in range(add_task_count_to_pool):
                add_tasks.append(task)
                task = task._next_non_pooled
            for task in add_tasks:
                task._prev_non_pooled = task._next_non_pooled = None
                self._non_pooled_tasks.remove(task)
                task._future = self._thread_pool_executor.submit(task._pool_thread_proc)

    def _update_progressbar(self):
        tc = len(self._tasks)
        self._progress_bar.setMaximum(tc)
        self._progress_bar.setValue(len(self._done_tasks))
