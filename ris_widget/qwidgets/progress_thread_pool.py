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
from PyQt5 import Qt
from .. import om

class TaskStatus(Enum):
    New = 0       # Task is not in a ProgressThreadPool's .tasks
    Queued = 1    # Task is in a ProgressThreadPool's .tasks but has not been submitted to its thread pool
    Pooled = 2    # Task has been submitted to a ProgressThreadPool's thread pool, and is waiting for the thread pool to start running it
    Started = 3   # Task is running
    Completed = 4 # Task was submitted to the thread pool, started, and returned normally
    Failed = 5    # Either task was submitted to the thread pool, started, and raised an exception, or submission to the pool failed
    Cancelled = 6 # Task was cancelled before it finished

class Task:
    __slots__ = ('callable', 'callable_va', 'callable_kw', '_instance_count', '_status', '_future', '_progress_thread_pool', '_prev_queued', '_next_queued')
    def __init__(self, callable, *callable_va, **callable_kw):
        self.callable = callable
        self.callable_va = callable_va
        self.callable_kw = callable_kw
        self._instance_count = 0
        self._status = TaskStatus.New
        self._next_queued = self._prev_queued = self._progress_thread_pool = self._future = None

    def _submit(self):
        assert self._progress_thread_pool is not None
        assert self._future is None
        assert Qt.QThread.currentThread() is Qt.QApplication.instance().thread()
        assert self._status == TaskStatus.Queued
        old_status = self._status
        self._status = TaskStatus.Pooled
        try:
            self._future = self._progress_thread_pool._thread_pool_executor.submit(self._pool_thread_proc)
        except Exception as e:
            self._status = TaskStatus.Failed
            self._post_task_status_change_event(old_status)
            raise e
        self._post_task_status_change_event(old_status)
        self._future.add_done_callback(self._done_callback)

    def _cancel(self):
        if self._status in (TaskStatus.New, TaskStatus.Queued):
            self._status = TaskStatus.Cancelled
            self._post_task_status_change_event(TaskStatus.Queued)
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
        assert(self._status == TaskStatus.Pooled)
        self._status = TaskStatus.Started
        self._post_task_status_change_event(TaskStatus.Pooled)
        self.callable(*self.callable_va, **self.callable_kw)

    def _post_task_status_change_event(self, old_status):
        ptp = self._progress_thread_pool
        if ptp is not None:
            Qt.QApplication.instance().postEvent(ptp, _task_status_change_event(self, self.status, old_status))

    @property
    def result(self):
        '''If the following return statement raises an AttributeError, it is because this Task has not yet been pooled
        (submitted to self._progress_thread_pool._thread_pool_executor) or perhaps has not even been added to a
        ProgressThreadPool.'''
        return self._future.result(0)

    @property
    def progress_thread_pool(self):
        return self._progress_thread_pool

    @property
    def status(self):
        return self._status

_TASK_STATUS_CHANGED_EVENT = Qt.QEvent.registerEventType()

class _task_status_change_event(Qt.QEvent):
    def __init__(self, task, new_status, old_status):
        super().__init__(_TASK_STATUS_CHANGED_EVENT)
        self.task = task
        self.new_status = new_status
        self.old_status = old_status

class ProgressThreadPool(Qt.QWidget):
    '''ProgressThreadPool: A Qt widget for running an ordered collection of tasks in a thread pool, with a cancel
    button and a progress bar showing the percentage of tasks in the collection that have completed.  The collection
    of tasks is accessible via the .tasks property and is a SignalingList containing Task instances.  A Task may be
    queued for execution by inserting it anywhere in .tasks, or by calling .submit, which is provided for convenience.
    When it comes time to send a Task into the threadpool executor, the first Task in .tasks with a .status of Queued
    is sent.  Likewise, a Task may be cancelled by removing it from .tasks.

    In detail:

    (For the remainder of this text, let ptp be a ProgressThreadPool instance and t be a Task instance.)

    "ptp.submit(foo_func, 1, 2, 3, foo_arg=4)" is exactly equivalent to
    "ptp.tasks.append(Task(foo_func, 1, 2, 3, foo_arg=4))", except that ptp.submit may be called from any thread.

    ptp responds to modification of ptp.tasks, whether by insertion, append, removal, or replacement.  You may
    perform any of these operations on ptp.tasks at any time, with the following limitations:
    * Only instances of Task may be added to ptp.tasks.
    * ptp.tasks may only be manipulated by the thread that owns ptp, and this is necessarily the GUI thread, as
      ProgressThreadPool is a QWidget, a GUI thing.
    * In general, t can not be in more than one ProgressThreadPool instance's .tasks, although it may appear in
      ptp.tasks more than once (EG, it is OK if "ptp.tasks[0] is ptp.tasks[1]" evaluates to True).
    * If t.status is Queued, Pooled or Started, t may only be added to ptp.tasks if it is already in ptp.tasks.

    Removing the last remaining instance of t from ptp.tasks (IE, the last instance of the Task instance that is
    t; other Task instances for which "other_task_instance is t" is False may remain in ptp.tasks) causes
    t.progress_thread_pool to become None.  If t.status was Queued when t was removed, t.status changes to New.
    If t.status was Pooled, it usually changes to New, but there exists a small window during which the thread
    pool executor has started the task and we do not yet know it, which we learn is the case if the Tasks's future
    rejects it cancel call.  If this happens, t.status changes to Started, eventually changing to Completed or Failed
    if t's callable ever exits.  If t.status was Started, Completed, Failed, or Cancelled, removal of t from ptp.tasks
    does not change the value of t.status.

    Signals:

    * task_count_changed(progress_thread_pool): len(progress_thread_pool_instance.tasks) changed.
    * task_status_changed(task, task_status): task.status changed from task_status.'''
    task_count_changed = Qt.pyqtSignal(object)
    task_status_changed = Qt.pyqtSignal(Task, TaskStatus)
    _x_thread_submit = Qt.pyqtSignal(Task)
    _x_thread_cancel = Qt.pyqtSignal()

    def __init__(self, thread_pool_executor=None, max_workers=None, parent=None, show_cancel=True):
        '''If None is supplied for thread_pool_executor, a new concurrent.futures.ThreadPoolExecutor is created
        with max_workers.  If max_workers is None, max thread count is clamp(int(multiprocessing.cpu_count()/2),1,8)
        (IE, if None is supplied for max_workers, floor(cpu_count/2), bounded to the interval [1,8], is used).

        If a value is supplied for thread_pool_executor, the max_workers argument controls the maximum number of
        outstanding Tasks.  If you supply 2 for max_workers and futures.ThreadPoolExecutor(max_workers=8) for
        thread_pool_executor, the resulting ProgressThreadPool will only allow two of its Tasks to be in the
        thread pool at a time although the pool could ostensibly run eight in parallel.'''
        super().__init__(parent)
        self._ignore_task_list_change = False
        if max_workers is None:
            import multiprocessing
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
        self._cancel_button.clicked.connect(self._cancel)
        if not show_cancel:
            self._cancel_button.hide()
        self._x_thread_submit.connect(self._on_x_thread_submit, Qt.Qt.QueuedConnection)
        self._x_thread_cancel.connect(self._cancel, Qt.Qt.QueuedConnection)
        self._tasks = om.SignalingList()
        self._queued_tasks = set()
        self._pooled_tasks = set()
        self._started_tasks = set()
        self._retired_tasks = set() # A task that is completed, failed, or cancelled is considered retired
        self._task_status_sets = {
            TaskStatus.Queued : self._queued_tasks,
            TaskStatus.Pooled : self._pooled_tasks,
            TaskStatus.Started : self._started_tasks,
            TaskStatus.Completed : self._retired_tasks,
            TaskStatus.Failed : self._retired_tasks,
            TaskStatus.Cancelled : self._retired_tasks}
        self._first_queued_task = None # Head of Queued Task linked list
        self._tasks.inserting.connect(self._on_tasks_inserting)
        self._tasks.inserted.connect(self._on_tasks_inserted)
        self._tasks.replacing.connect(self._on_tasks_replacing)
        self._tasks.replaced.connect(self._on_tasks_replaced)
        self._tasks.removed.connect(self._on_tasks_removed)

    def submit(self, callable, *callable_va, **callable_kw):
        task = Task(callable, *callable_va, **callable_kw)
        if Qt.QThread.currentThread() is Qt.QApplication.instance().thread():
            self._tasks.append()
        else:
            self._x_thread_submit.emit(task)
        return task

    def _on_x_thread_submit(self, task):
        self._tasks.append(task)

    def cancel(self):
        '''Attempts to cancel any running and queued tasks.  Safe to call from any thread.'''
        if Qt.QThread.currentThread() is Qt.QApplication.instance().thread():
            self._cancel()
        else:
            self._x_thread_cancel.emit()

    def _cancel(self):
        for task in self._queued_tasks:
            task._cancel()
        for task in self._started_tasks:
            task._cancel()

    @property
    def max_workers(self):
        return self._max_workers

    @property
    def tasks(self):
        assert Qt.QThread.currentThread() is Qt.QApplication.instance().thread(),\
               'ProgressThreadPool.tasks should only be accessed and/or manipulated from the main/GUI thread.'
        return self._tasks

    @property
    def thread_pool_executor(self):
        return self._thread_pool_executor

    def event(self, event):
        if event.type() == _TASK_STATUS_CHANGED_EVENT:
            assert isinstance(event, _task_change_status_event)
            task = event.task
            status = task.status
            assert status != TaskStatus.New
            old_status = event.old_status
            if status != old_status:
                self._on_task_status_changed_ev(task, old_status)
                self.task_status_changed.emit(task, old_status)
            return True
        return super().event(event)

    _EXTERNALLY_SET_STATUSES = frozenset(TaskStatus.Started, TaskStatus.Completed, TaskStatus.Failed, TaskStatus.Cancelled)
    def _on_task_status_changed_ev(self, task, old_status):
        if task.status in ProgressThreadPool._EXTERNALLY_SET_STATUSES:
            self._task_status_sets[old_status].remove(task)
            self._task_status_sets[task.status].add(task)
            self._update_progressbar()
            self._update_pool()

    _ACCEPTABLE_ADDED_DETACHED_TASK_STATUSES = frozenset(TaskStatus.New, TaskStatus.Completed, TaskStatus.Failed, TaskStatus.Cancelled)
    def _on_tasks_inserting(self, insertion_idx, tasks):
        for task in tasks:
            if not isinstance(task, Task):
                raise ValueError("Only instances of Task (or subclasses thereof) may be inserted into a ProgressThreadPool's .tasks list.")
            if task._progress_thread_pool is None:
                assert task._status not in (TaskStatus.Queued, TaskStatus.Pooled), "Inconsistent state: a detached Task is Queued or Pooled."
                if task._status is not in ProgressThreadPool._ACCEPTABLE_ADDED_DETACHED_TASK_STATUSES:
                    raise ValueError(
                        "A running Task (IE, with .status of TaskStatus.Started) may only be inserted into a ProgressThreadPool's .tasks "
                        "list if it is already a member of that ProgressThreadPool's .tasks list.  That is, there may be multiple entries "
                        "referring to a single running Task in a ProgressThreadPool's .tasks, but a running Task can not appear in more "
                        "than one ProgressThreadPool's .tasks list.")


    def _on_tasks_inserted(self, insertion_idx, tasks):
        # Note: It is expected that _on_tasks_inserting has run without raising an exception if we are here.
        for task in tasks:
            if :
        if not tasks:
            # All 
        if tasks:
            if self._queued_tasks:
                # Scan ._tasks list backward forward simultaneously from inserted region until a non-pooled Task is found
                # or it becomes impossible to advance both directions (which would indicate that we are in an inconsistent
                # state)
                bidx, fidx = insertion_idx - 1, insertion_idx + len(tasks)
                bE, fE = bidx < 0, fidx >= len(self._tasks)
                while not bE or not fE:
                    if not bE:
                        t = self._tasks[bidx]
                        if not t.is_pooled:
                            tn = t._next_queued
                            t0 = tasks[0]
                            t1 = tasks[-1]
                            t0._prev_queued = t
                            t._next_queued = t0
                            t1._next_queued = tn
                            if tn is not None:
                                tn._prev_queued = t1
                            for ta, tb in zip(tasks, tasks[1:]):
                                ta._next_queued = tb
                                tb._prev_queued = ta
                            break
                        bidx -= 1
                        bE = bidx < 0
                    if not Fe:
                        t = self._tasks[fidx]
                        if not t.is_pooled:
                            tp = t._prev_queued
                            t0 = tasks[0]
                            t1 = tasks[-1]
                            t1._next_queued = t
                            t._prev_queued = t1
                            t0._prev_queued = tp
                            if tp is None:
                                # We scanned forward and the first non-pooled Task we found has no preceeding non-pooled Tasks.
                                # The inserted Tasks preceed it, making the first non-pooled, inserted Task the new non-pooled
                                # Task linked-list head.
                                if t is not self._first_queued_task:
                                    raise RuntimeError('Inconsistent state: scanning forward found an non-pooled Task that has no preceeding '
                                                       'unpooled Task and yet somehow is not the non-pooled Task linked-list head.')
                                self._first_queued_task = t
                            else:
                                tp._next_queued = t0
                            for ta, tb in zip(tasks, tasks[1:]):
                                ta._next_queued = tb
                                tb._prev_queued = ta
                            break
                        fidx += 1
                        fE = fidx >= len(self._tasks)
                else:
                    raise RuntimeError(
                        'Inconsistent state: self._queued_tasks is not empty, but self._tasks contains no unpooled Tasks '
                        'other than those just inserted.')
            else:
                for ta, tb in zip(tasks, tasks[1:]):
                    ta._next_queued = tb
                    tb._prev_queued = ta
                tasks[0]._prev_queued = None
                tasks[-1]._next_queued = None
                self._first_queued_task = tasks[0]
            for t in tasks:
                self._queued_tasks.add(t)
            self._update_progressbar()
            self._update_pool()

    def _on_tasks_replacing(self, idxs, replaced_tasks, tasks):
        assert tasks is self._tasks
        if not all(isinstance(task) for task in tasks):
            raise ValueError("Only instances of Task (or subclasses thereof) may be added to a ProgressThreadPool's .tasks list.")

    def _on_tasks_replaced(self, idxs, replaced_tasks, tasks):
        pass

    def _on_tasks_removed(self, idxs, tasks):
        assert all(isinstance(task, Task) for task in tasks)

    def _update_pool(self):
        add_task_count_to_pool = min(len(self._queued_tasks) - len(self._done_tasks), self._max_workers)
        print('len(self._queued_tasks)', len(self._queued_tasks), 'add_task_count_to_pool', add_task_count_to_pool)
        if add_task_count_to_pool > 0:
            add_tasks = []
            task = self._first_queued_task
            assert task._prev_queued is None
            for _ in range(add_task_count_to_pool):
                add_tasks.append(task)
                task = task._next_queued
            for task in add_tasks:
                task._prev_queued = task._next_queued = None
                self._queued_tasks.remove(task)
                task._future = self._thread_pool_executor.submit(task._pool_thread_proc)

    def _update_progressbar(self):
        tc = len(self._tasks)
        self._progress_bar.setMaximum(tc)
        self._progress_bar.setValue(len(self._retired_tasks))
