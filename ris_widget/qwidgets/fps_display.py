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

from PyQt5 import Qt
import time

class FPSDisplay(Qt.QWidget):
    """A widget displaying interval since last .notify call and 1 / the interval since last .notify call.
    FPSDisplay updates only when visible, reducing the cost of having it constructed and hidden with a signal
    attached to .notify."""
    def __init__(self, rate_str='framerate', rate_suffix_str='fps', interval_str='interval', parent=None):
        super().__init__(parent)
        l = Qt.QGridLayout()
        self.setLayout(l)
        self.rate_label = Qt.QLabel(rate_str)
        self.rate_field = Qt.QLabel()
        self.rate_suffix_label = Qt.QLabel(rate_suffix_str)
        l.addWidget(self.rate_label, 0, 0, Qt.Qt.AlignRight)
        l.addWidget(self.rate_field, 0, 1, Qt.Qt.AlignRight)
        l.addWidget(self.rate_suffix_label, 0, 2, Qt.Qt.AlignLeft)
        self.interval_label = Qt.QLabel(interval_str)
        self.interval_field = Qt.QLabel()
        self.interval_suffix_label = Qt.QLabel()
        l.addWidget(self.interval_label, 1, 0, Qt.Qt.AlignRight)
        l.addWidget(self.interval_field, 1, 1, Qt.Qt.AlignRight)
        l.addWidget(self.interval_suffix_label, 1, 2, Qt.Qt.AlignLeft)
        self.prev_notify_t = None
        self.latest_notify_t = None

    def notify(self):
        t = time.time()
        self.prev_notify_t = self.latest_notify_t
        self.latest_notify_t = t
        self._refresh()

    def clear(self):
        self.prev_notify_t = self.latest_notify_t = None
        self._refresh()

    def _refresh(self):
        if not self.isVisible():
            return
        if self.prev_notify_t is None or self.latest_notify_t is None or self.latest_notify_t <= self.prev_notify_t:
            self.rate_field.setText('')
            self.interval_field.setText('')
        else:
            delta = self.latest_notify_t - self.prev_notify_t
            self.rate_field.setText('{:f}'.format(1 / delta))
            if delta > 1:
                self.interval_field.setText('{:f}'.format(delta))
                self.interval_suffix_label.setText('s')
            else:
                self.interval_field.setText('{:f}'.format(delta * 1000))
                self.interval_suffix_label.setText('ms')

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh()