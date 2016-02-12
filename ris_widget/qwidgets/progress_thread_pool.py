# The MIT License (MIT)
#
# Copyright (c) 2015-2016 WUSTL ZPLAB
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
        ProgressThreadPool."""
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

# NOTE: There are only two situations in which a Task's status is changed from a thread other than the GUI thread:
# 1. When the Task's _pool_thread_proc is begins executing, the Task's status changes from Pooled to Started
# 2. When the Task's _pool_thread_proc exits, whether by return or exception, status changes from Started to Completed or Failed
# Given the above, the need for locking in order to ensure data structure consistency in ProgressThreadPool can be avoided
# by carefully arranging ProgressThreadPool's logic.  One particularly expedient: the main thread is responsible for
# submission of queued tasks to the thread pool executor; as this submission occurs, the main thread updates the Tasks's
# status from Queued to Pooled.
#
# However, this leaves a gap that can be surprising: in the interval between reading a Task's status as being Pooled
# and attempting to cancel that Task, the Task's status may switch to Started.  Such a Task will transition to Completed or
# Failed depending on the outcome of its procedure; it will not become Cancelled.
class ProgressThreadPool(Qt.QWidget):
    """ProgressThreadPool: A Qt widget for running an ordered collection of tasks in a thread pool, with a cancel
    button and a progress bar showing the percentage of tasks in the collection that have completed.  The collection
    of tasks is accessible via the .tasks property and is a SignalingList containing Task instances.  It is acceptable
    to place non-Task objects in .tasks; ProgressThreadPool simply ignores any elements of .tasks that are not Task
    instances. A Task may be queued for execution by inserting it anywhere in .tasks, or by calling .submit, which is
    provided for convenience. When it comes time to send a Task into the threadpool executor, the first Task in .tasks
    with a .status of Queued is sent.  Likewise, a Task may be cancelled by removing it from .tasks.

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
    if t's callable_ ever exits.  If t.status was Started, Completed, Failed, or Cancelled, removal of t from ptp.tasks
    does not change the value of t.status.

    Signals:

    * task_status_changed(task, task_status): task.status changed from task_status."""
    task_status_changed = Qt.pyqtSignal(Task, TaskStatus)
    all_tasks_retired = Qt.pyqtSignal()
    cancelled = Qt.pyqtSignal()
    _x_thread_submit = Qt.pyqtSignal(Task)
    _x_thread_cancel = Qt.pyqtSignal()

    def __init__(self, tasks=None, thread_pool_executor=None, max_workers=None, parent=None, show_cancel=True):
        """If None is supplied for thread_pool_executor, a new concurrent.futures.ThreadPoolExecutor is created
        with max_workers.  If max_workers is None, max thread count is multiprocessing.cpu_count() + 1.

        If a value is supplied for thread_pool_executor, the max_workers argument controls the maximum number of
        outstanding Tasks.  If you supply 2 for max_workers and futures.ThreadPoolExecutor(max_workers=8) for
        thread_pool_executor, the resulting ProgressThreadPool will only allow two of its Tasks to be in the
        thread pool at a time although the pool could ostensibly run eight in parallel."""
        super().__init__(parent)
        self._updating_pool = False
        if max_workers is None:
            import multiprocessing
            max_workers = multiprocessing.cpu_count() + 1
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
        if tasks is None:
            tasks = om.SignalingList()
        else:
            self._validate_tasks_for_adding(tasks)
            if not isinstance(tasks, om.SignalingList):
                tasks = om.SignalingList(tasks)
        self._tasks = tasks
        self._first_queued_task_idx_hint = None
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
        self._tasks.inserting.connect(self._on_tasks_inserting)
        self._tasks.inserted.connect(self._on_tasks_inserted)
        self._tasks.replacing.connect(self._on_tasks_replacing)
        self._tasks.replaced.connect(self._on_tasks_replaced)
        self._tasks.removed.connect(self._on_tasks_removed)
        if tasks:
            self._on_tasks_inserted(0, tasks)

    def submit(self, callable_, *callable_va, **callable_kw):
        task = Task(callable_, *callable_va, **callable_kw)
        if Qt.QThread.currentThread() is Qt.QApplication.instance().thread():
            self._tasks.append(task)
        else:
            self._x_thread_submit.emit(task)
        return task

    def _on_x_thread_submit(self, task):
        self._tasks.append(task)

    def cancel(self):
        """Attempts to cancel any running and queued tasks.  Safe to call from any thread."""
        if Qt.QThread.currentThread() is Qt.QApplication.instance().thread():
            self._cancel()
        else:
            self._x_thread_cancel.emit()

    def _cancel(self):
        try:
            self._updating_pool = True
            for task in list(self._queued_tasks):
                task._cancel()
            for task in list(self._started_tasks):
                task._cancel()
        finally:
            self._updating_pool = False
        self.cancelled.emit()

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
            assert isinstance(event, _TaskStatusChangeEvent)
            task = event.task
            status = task.status
            assert status is not TaskStatus.New
            old_status = event.old_status
            new_status = event.new_status
            if new_status is not old_status:
                self._on_task_status_changed_ev(task, new_status, old_status)
                self.task_status_changed.emit(task, old_status)
            return True
        return super().event(event)

    def _on_task_status_changed_ev(self, task, new_status, old_status):
        if old_status is not TaskStatus.New:
            self._task_status_sets[old_status].remove(task)
        self._task_status_sets[new_status].add(task)
        self._update_progressbar()
        self._update_pool()

    _ACCEPTABLE_ADDED_DETACHED_TASK_STATUSES = frozenset((TaskStatus.New, TaskStatus.Completed, TaskStatus.Failed, TaskStatus.Cancelled))

    def _validate_tasks_for_adding(self, tasks):
        for task in tasks:
            if isinstance(task, Task):
                if task._progress_thread_pool is None:
                    assert task._status not in (TaskStatus.Queued, TaskStatus.Pooled), "Inconsistent state: a detached Task is Queued or Pooled."
                    assert task._instance_count == 0, "Inconsistent state: a detached Task has instance count other than 0."
                    assert task not in self._queued_tasks, "Inconsistent state: a detached Task is already in queued tasks set."
                    if task._status not in ProgressThreadPool._ACCEPTABLE_ADDED_DETACHED_TASK_STATUSES:
                        raise ValueError(
                            "A running Task (IE, with .status of TaskStatus.Started) may only be inserted into a ProgressThreadPool's .tasks "
                            "list if it is already a member of that ProgressThreadPool's .tasks list.  That is, there may be multiple entries "
                            "referring to a single running Task in a ProgressThreadPool's .tasks, but a running Task can not appear in more "
                            "than one ProgressThreadPool's .tasks list.")
                elif task._progress_thread_pool is not self:
                    raise ValueError(
                        "A queued, pooled, or running task may only inserted into a ProgressThreadPool's .tasks list if it is already a "
                        "member of that ProgressThreadPool's .tasks list.  In order to move a queued or pooled Task from one ProgressThreadPool "
                        "to another, store the Task in a temporary variable, remove it from the source ProgressThreadPool, then insert it into "
                        "the destination ProgressThreadPool's .tasks list.")

    def _on_tasks_inserting(self, insertion_idx, tasks):
        self._validate_tasks_for_adding(tasks)

    def _on_tasks_inserted(self, insertion_idx, tasks):
        # Note: It is expected that self._validate_tasks_for_adding(tasks) has run without raising an exception if we are here
        prev_qqlen = len(self._queued_tasks)
        for tidx, t in enumerate(tasks, insertion_idx):
            if isinstance(t, Task)and t._status in (TaskStatus.New, TaskStatus.Queued):
                if self._first_queued_task_idx_hint is None or tidx < self._first_queued_task_idx_hint:
                    self._first_queued_task_idx_hint = tidx
                t._instance_count += 1
                if t._instance_count == 1 and t._status is TaskStatus.New:
                    t._progress_thread_pool = self
                    self._queued_tasks.add(t)
                    t._status = TaskStatus.Queued
                    self.task_status_changed.emit(TaskStatus.New)
        self._update_progressbar()
        if prev_qqlen != len(self._queued_tasks):
            self._update_pool()

    def _on_tasks_replacing(self, idxs, replaced_tasks, tasks):
        assert tasks is self._tasks
        if not all(isinstance(task) for task in tasks):
            raise ValueError("Only instances of Task (or subclasses thereof) may be added to a ProgressThreadPool's .tasks list.")
        raise NotImplementedError()

    def _on_tasks_replaced(self, idxs, replaced_tasks, tasks):
        raise NotImplementedError()

    def _on_tasks_removed(self, idxs, tasks):
        for tidx, t in zip(idxs, tasks):
            if isinstance(t, Task):
                t_status = t._status
                t._progress_thread_pool = None
                if t_status is TaskStatus.Queued:
                    if self._first_queued_task_idx_hint is not None and tidx <= self._first_queued_task_idx_hint:
                        self._first_queued_task_idx_hint = None
                if t.status is TaskStatus.Pooled:
                    t._cancel()

    def _update_pool(self):
        if self._updating_pool:
            return
        try:
            self._updating_pool = True
            add_task_count_to_pool = min(
                self._max_workers - (len(self._pooled_tasks) + len(self._started_tasks)),
                len(self._queued_tasks))
#           print('len(self._queued_tasks)', len(self._queued_tasks), 'add_task_count_to_pool', add_task_count_to_pool)
            if add_task_count_to_pool <= 0:
                return
            for _ in range(add_task_count_to_pool):
                n = self._task_qhead
                assert n is not None
                assert n.prev_queued is None
                n.task._submit()
        finally:
            self._updating_pool = False

    def _update_progressbar(self):
        rtc = len(self._retired_tasks)
        ttc = len(self._queued_tasks) + len(self._pooled_tasks) + len(self._started_tasks) + rtc
        self._progress_bar.setMaximum(ttc)
        self._progress_bar.setValue(rtc)
        if ttc > 0 and ttc == rtc:
            self.all_tasks_retired.emit()
