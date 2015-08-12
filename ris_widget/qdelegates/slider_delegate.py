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

from PyQt5 import Qt

class SliderDelegate(Qt.QStyledItemDelegate):
    def __init__(self, min_value, max_value, parent=None):
        super().__init__(parent)
        self.min_value = min_value
        self.max_value = max_value
        self._drag_grabber = None

    def sizeHint(self, option, midx):
        return Qt.QSize(100,10)

    def paint(self, qpainter, option, midx):
        if not midx.isValid():
            return
        d = midx.data()
        if isinstance(d, Qt.QVariant):
            d = d.value()
        pbo = Qt.QStyleOptionProgressBar()
        pbo.minimum, pbo.maximum = 0, 100
        pbo.progress = int( (d-self.min_value)/(self.max_value-self.min_value) * 100.0 )
        pbo.text = '{}%'.format(pbo.progress)
        pbo.textVisible = True
        pbo.textAlignment = Qt.Qt.AlignCenter
        pbo.rect = option.rect
        style = option.widget.style() if option.widget is not None else Qt.QApplication.style()
        style.drawControl(Qt.QStyle.CE_ProgressBar, pbo, qpainter)

    def editorEvent(self, event, model, option, midx):
        print(option.rect, option.widget.visualRect(midx), option.widget.viewport().geometry())
        if not midx.isValid() or not event.type() == Qt.QEvent.MouseButtonPress or event.buttons() != Qt.Qt.LeftButton:
            return False
        if self._drag_grabber is not None:
            self._drag_grabber.deleteLater()
        offset = option.widget.viewport().geometry().topLeft()
        rect = Qt.QRect(
            option.rect.left() + offset.x(),
            option.rect.top() + offset.y(),
            option.rect.size().width(),
            option.rect.size().height())
        self._drag_grabber = DragGrabber(option.widget, model, rect, midx)
        self._drag_grabber.destroyed.connect(self.on_drag_grabber_destroyed)
        self._drag_grabber.drag_x_changed.connect(self.on_drag_x_changed)
        return self.on_drag_x_changed(event.localPos().x(), option.rect, model, midx)

    def on_drag_x_changed(self, x, r, model, midx):
        sl, sw = r.left(), r.width()
        v = ((x - sl) / sw) * (self.max_value - self.min_value) + self.min_value
        if v < self.min_value:
            v = self.min_value
        elif v > self.max_value:
            v = self.max_value
        return model.setData(midx, Qt.QVariant(v), Qt.Qt.EditRole)

    def on_drag_grabber_destroyed(self):
        self._drag_grabber = None

class DragGrabber(Qt.QWidget):
    drag_x_changed = Qt.pyqtSignal(int, Qt.QRect, Qt.QAbstractItemModel, Qt.QModelIndex)

    def __init__(self, parent, model, rect, midx):
        super().__init__(parent)
        self._ignore_next_no_button_mouse_move_event = True
        self.model = model
        self.midx = midx
        self.setMouseTracking(True)
        # DragGrabber exactly covers the cell it manipulates, so it must be transparent (or simply
        # not paint its background in the first place) in order for the slider to be visible
        # while it is dragged.
        self.setAutoFillBackground(False)
        self.setGeometry(rect)
        self.show()
        self.grabMouse()

    def mouseReleaseEvent(self, event):
        event.accept()
        self.releaseMouse()
        self.deleteLater()

    def mouseMoveEvent(self, event):
        event.accept()
        if event.buttons() != Qt.Qt.LeftButton:
            if event.buttons() == Qt.Qt.NoButton and self._ignore_next_no_button_mouse_move_event:
                # In the definition of QApplicationPrivate::sendSyntheticEnterLeave(..), line 2788 of 
                # qtbase-opensource-src-5.4.2/src/widgets/kernel/qapplication.cpp, is called when
                # widget visibility changes, including as a result of the widget's first .show() call.
                #
                # The last two lines of sendSyntheticEnterLeave(..):
                # QMouseEvent e(QEvent::MouseMove, pos, windowPos, globalPos, Qt::NoButton, Qt::NoButton, Qt::NoModifier);
                # sendMouseEvent(widgetUnderCursor, &e, widgetUnderCursor, tlw, &qt_button_down, qt_last_mouse_receiver);
                #
                # So, even though sendSyntheticEnterLeave may be being called because a widget was shown
                # in reponse to a mouse click, the first mouse event the newly created widget will receive
                # if a number of conditions are met, and they are in this case, is the synthetic move event
                # with flags set to indicate that it occurred with no mouse buttons depressed.  We consider
                # dragging to have ended when the left mouse button is released, and so we must ignore this
                # synethetic event if our DragGrabber is to exist for more than one event loop iteration.
                self._ignore_next_no_button_mouse_move_event = False
            else:
                self.releaseMouse()
                self.deleteLater()
        else:
            self.drag_x_changed.emit(event.x(), self.rect(), self.model, self.midx)

    def focusOutEvent(self, event):
        self.releaseMouse()
        self.deleteLater()

