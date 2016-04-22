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
# Authors: Erik Hvatum <ice.rikh@gmail.com> and Zach Pincus <zpincus@wustl.edu>

import concurrent.futures as futures
import multiprocessing
import threading
import traceback

from PyQt5 import Qt


class UpdateEvent(Qt.QEvent):
    TYPE = Qt.QEvent.registerEventType()

    def __init__(self):
        super().__init__(self.TYPE)

    def post(self, sender):
        Qt.QApplication.instance().postEvent(sender, self)

class ProgressThreadPool(Qt.QWidget):
    def __init__(self, cancel_jobs, attached_layout, parent=None):
        super().__init__(parent)
        self.thread_pool = futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()-1)
        self.task_count_lock = threading.Lock()
        self._queued_tasks = 0
        self._retired_tasks = 0

        l = Qt.QHBoxLayout()
        self.setLayout(l)
        self._progress_bar = Qt.QProgressBar()
        self._progress_bar.setMinimum(0)
        l.addWidget(self._progress_bar)
        self._cancel_button = Qt.QPushButton('Cancel')
        l.addWidget(self._cancel_button)
        self._cancel_button.clicked.connect(cancel_jobs)
        attached_layout().addWidget(self)
        self.hide()

    def _task_done(self, future):
        self.increment_retired()
        try:
            future.result()
        except futures.CancelledError:
            pass
        except:
            if future.on_error is not None:
                future.on_error(*future.on_error_args)
            traceback.print_exc()

    def submit(self, task, *args, on_error=None, on_error_args=[], **kws):
        self.increment_queued()
        future = self.thread_pool.submit(task, *args, **kws)
        future.on_error = on_error
        future.on_error_args = on_error_args
        future.add_done_callback(self._task_done)
        return future

    def increment_queued(self):
        with self.task_count_lock:
            self._queued_tasks += 1
        UpdateEvent().post(self)

    def increment_retired(self):
        with self.task_count_lock:
            self._retired_tasks += 1
        UpdateEvent().post(self)

    def event(self, event):
        if event.type() == UpdateEvent.TYPE:
            self._update_progressbar()
            return True
        return super().event(event)

    def _update_progressbar(self):
        self._progress_bar.setMaximum(self._queued_tasks)
        self._progress_bar.setValue(self._retired_tasks)
        with self.task_count_lock:
            if self._queued_tasks == self._retired_tasks:
                self.hide()
                self._queued_tasks = 0
                self._retired_tasks = 0
            elif self.isHidden():
                self.show()

