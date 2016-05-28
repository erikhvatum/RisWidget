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

import numpy
from PyQt5 import Qt
import time

class FPSDisplay(Qt.QWidget):
    """A widget displaying interval since last .notify call and 1 / the interval since last .notify call.
    FPSDisplay collects data and refreshes only when visible, reducing the cost of having it constructed
    and hidden with a signal attached to .notify."""
    def __init__(self, rate_str='Framerate', rate_suffix_str='fps', interval_str='Interval', parent=None):
        super().__init__(parent)
        l = Qt.QGridLayout()
        self.setLayout(l)
        self.intervals = None
        self.fpss = None
        self._sample_count = None
        self.acquired_sample_count = 0
        self.prev_t = None
        r = 0
        self.sample_count_label = Qt.QLabel('Sample count: ')
        self.sample_count_spinbox = Qt.QSpinBox()
        self.sample_count_spinbox.setRange(2, 1024)
        self.sample_count_spinbox.valueChanged[int].connect(self._on_sample_count_spinbox_value_changed)
        l.addWidget(self.sample_count_label, r, 0, Qt.Qt.AlignRight)
        l.addWidget(self.sample_count_spinbox, r, 1)
        r += 1
        self.rate_label = Qt.QLabel(rate_str + ': ')
        self.rate_field = Qt.QLabel()
        self.rate_suffix_label = Qt.QLabel(rate_suffix_str)
        l.addWidget(self.rate_label, r, 0, Qt.Qt.AlignRight)
        l.addWidget(self.rate_field, r, 1, Qt.Qt.AlignRight)
        l.addWidget(self.rate_suffix_label, r, 2, Qt.Qt.AlignLeft)
        r += 1
        self.interval_label = Qt.QLabel(interval_str + ': ')
        self.interval_field = Qt.QLabel()
        self.interval_suffix_label = Qt.QLabel()
        l.addWidget(self.interval_label, r, 0, Qt.Qt.AlignRight)
        l.addWidget(self.interval_field, r, 1, Qt.Qt.AlignRight)
        l.addWidget(self.interval_suffix_label, r, 2, Qt.Qt.AlignLeft)
        r += 1
        self.rate_stddev_label = Qt.QLabel(rate_str + ' \N{GREEK SMALL LETTER SIGMA}: ')
        self.rate_stddev_field = Qt.QLabel()
        self.rate_stddev_suffix_label = Qt.QLabel(rate_suffix_str)
        l.addWidget(self.rate_stddev_label, r, 0, Qt.Qt.AlignRight)
        l.addWidget(self.rate_stddev_field, r, 1, Qt.Qt.AlignRight)
        l.addWidget(self.rate_stddev_suffix_label, r, 2, Qt.Qt.AlignLeft)
        r += 1
        self.interval_stddev_label = Qt.QLabel(interval_str + ' \N{GREEK SMALL LETTER SIGMA}: ')
        self.interval_stddev_field = Qt.QLabel()
        self.interval_stddev_suffix_label = Qt.QLabel()
        l.addWidget(self.interval_stddev_label, r, 0, Qt.Qt.AlignRight)
        l.addWidget(self.interval_stddev_field, r, 1, Qt.Qt.AlignRight)
        l.addWidget(self.interval_stddev_suffix_label, r, 2, Qt.Qt.AlignLeft)
        r += 1
        l.addItem(
            Qt.QSpacerItem(
                0, 0, Qt.QSizePolicy.MinimumExpanding, Qt.QSizePolicy.MinimumExpanding
            ),
            r, 0,
            1, -1
        )
        l.setColumnStretch(0, 0)
        l.setColumnStretch(1, 1)
        l.setColumnStretch(2, 0)
        self.stddev_widgets = (
            self.rate_stddev_label,
            self.rate_stddev_field,
            self.rate_stddev_suffix_label,
            self.interval_stddev_label,
            self.interval_stddev_field,
            self.interval_stddev_suffix_label
        )
        self.sample_count = 20

    @property
    def sample_count(self):
        return self._sample_count

    @sample_count.setter
    def sample_count(self, v):
        assert 2 <= v <= 1024
        if v != self._sample_count:
            self._sample_count = v
            self.intervals = numpy.empty((self._sample_count - 1,), dtype=numpy.float64)
            self.fpss = numpy.empty((self._sample_count - 1,), dtype=numpy.float64)
            self.sample_count_spinbox.setValue(self.sample_count)
            self.clear()
            show_stddev = v >= 3
            for stddev_widget in self.stddev_widgets:
                stddev_widget.setVisible(show_stddev)

    def notify(self, end_interval=False):
        if not self.isVisible():
            return
        t = time.time()
        if self.prev_t is None:
            if not end_interval:
                self.prev_t = t
                self.acquired_sample_count += 1
        else:
            wrap_idx = (self.acquired_sample_count-1) % self.intervals.shape[0]
            self.intervals[wrap_idx] = v = t - self.prev_t
            self.fpss[wrap_idx] = 0 if v == 0 else 1 / v
            if end_interval:
                self.prev_t = None
            else:
                self.acquired_sample_count += 1
                self.prev_t = t
            self._refresh()

    def clear(self):
        self.acquired_sample_count = 0
        self.prev_t = None
        if self.isVisible():
            self._refresh()

    def _refresh(self):
        if self.acquired_sample_count < 2:
            self.rate_field.setText('')
            self.interval_field.setText('')
            self.rate_stddev_field.setText('')
            self.interval_stddev_field.setText('')
        else:
            end = min(self._sample_count, self.acquired_sample_count - 1)
            intervals = self.intervals[:end]
            interval = intervals.mean()
            fpss = self.fpss[:end]
            fps = fpss.mean()
            self.rate_field.setText('{:f}'.format(fps))
            if interval > 1:
                self.interval_field.setText('{:f}'.format(interval))
                self.interval_suffix_label.setText('s')
            else:
                self.interval_field.setText('{:f}'.format(interval * 1000))
                self.interval_suffix_label.setText('ms')
            if self._sample_count >= 3:
                if self.acquired_sample_count < 3:
                    self.rate_stddev_field.setText('')
                    self.interval_stddev_field.setText('')
                else:
                    fps_stddev = fpss.std()
                    interval_stddev = intervals.std()
                    self.rate_stddev_field.setText('{:f}'.format(fps_stddev))
                    if interval_stddev > 1:
                        self.interval_stddev_field.setText('{:f}'.format(interval_stddev))
                        self.interval_stddev_suffix_label.setText('s')
                    else:
                        self.interval_stddev_field.setText('{:f}'.format(interval_stddev * 1000))
                        self.interval_stddev_suffix_label.setText('ms')

    def _on_sample_count_spinbox_value_changed(self, sample_count):
        self.sample_count = sample_count

    def hideEvent(self, event):
        super().hideEvent(event)
        self.clear()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh()