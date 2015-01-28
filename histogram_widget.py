# The MIT License (MIT)
#
# Copyright (c) 2014 WUSTL ZPLAB
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

class HistogramContainerWidget(Qt.QWidget):
    def __init__(self, parent, qsurface_format):
        super().__init__(parent)
        self.setLayout(Qt.QHBoxLayout())
        self.splitter = Qt.QSplitter(self)
        self.layout().addWidget(self.splitter)
        sp_expanding = Qt.QSizePolicy(Qt.QSizePolicy.Expanding, Qt.QSizePolicy.Expanding)
        sp_expanding.setHorizontalStretch(0)
        sp_expanding.setVerticalStretch(0)
        self.histogram_frame = Qt.QFrame(self.splitter)
        self.histogram_frame.setSizePolicy(sp_expanding)
        self.histogram_frame.setMinimumSize(Qt.QSize(120, 60))
        self.histogram_frame.setFrameShape(Qt.QFrame.StyledPanel)
        self.histogram_frame.setFrameShadow(Qt.QFrame.Sunken)
        self.histogram_frame.setLayout(Qt.QHBoxLayout())
        self.histogram_frame.layout().setSpacing(0)
        self.histogram_frame.layout().setContentsMargins(Qt.QMargins(0,0,0,0))
        self.histogram_widget = HistogramWidget(self.histogram_frame, qsurface_format)
        self.histogram_widget.setSizePolicy(sp_expanding)
        self.histogram_frame.layout().addWidget(self.histogram_widget)
        self.ctrls = Qt.QWidget(self)
        l = Qt.QGridLayout()
        self.ctrls.setLayout(l)
        self.splitter.addWidget(self.ctrls)
        self.splitter.addWidget(self.histogram_frame)
        self.gamma_gamma_label = Qt.QLabel('\u03b3\u03b3:', self.ctrls)
        self.gamma_gamma_slider = Qt.QSlider(Qt.Qt.Horizontal, self.ctrls)
        self.gamma_gamma_edit = Qt.QLineEdit(self.ctrls)
        l.addWidget(self.gamma_gamma_label, 0, 0, Qt.Qt.AlignRight)
        l.addWidget(self.gamma_gamma_slider, 0, 1)
        l.addWidget(self.gamma_gamma_edit, 0, 2)
        self.gamma_label = Qt.QLabel('\u03b3', self.ctrls)
        self.gamma_slider = Qt.QSlider(Qt.Qt.Horizontal, self.ctrls)
        self.gamma_edit = Qt.QLineEdit(self.ctrls)
        l.addWidget(self.gamma_label, 1, 0, Qt.Qt.AlignRight)
        l.addWidget(self.gamma_slider, 1, 1)
        l.addWidget(self.gamma_edit, 1, 2)


class HistogramWidget(Qt.QOpenGLWidget):
    def __init__(self, parent, qsurface_format):
        super().__init__(parent)
        self.setFormat(qsurface_format)

    def initializeGL(self):
        pass

    def paintGL(self):
        pass

    def resizeGL(self, x, y):
        pass
