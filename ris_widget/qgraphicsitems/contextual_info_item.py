# The MIT License (MIT)
#
# Copyright (c) 2014-2015 WUSTL ZPLAB
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

from contextlib import ExitStack
from pathlib import Path
from PyQt5 import Qt
from ..shared_resources import UNIQUE_QGRAPHICSITEM_TYPE

class ContextualInfoItem(Qt.QGraphicsObject):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()
    FIXED_POSITION_IN_VIEW = Qt.QPoint(10, 5)

    text_changed = Qt.pyqtSignal()

    def __init__(self, parent_item=None):
        super().__init__(parent_item)
        self.setFlag(Qt.QGraphicsItem.ItemIgnoresTransformations)
        self._font = Qt.QFont('Courier', pointSize=16, weight=Qt.QFont.Bold)
        self._font.setKerning(False)
        self._font.setStyleHint(Qt.QFont.Monospace, Qt.QFont.OpenGLCompatible | Qt.QFont.PreferQuality)
        self._pen = Qt.QPen(Qt.QColor(Qt.Qt.black))
        self._pen.setWidth(2)
        self._pen.setCosmetic(True)
        self._brush = Qt.QBrush(Qt.QColor(45,255,70,255))
        self._text = None
        self._text_flags = Qt.Qt.AlignLeft | Qt.Qt.AlignTop | Qt.Qt.AlignAbsolute
        self._picture = None
        self._bounding_rect = None
        # Necessary to prevent context information from disappearing when mouse pointer passes over
        # context info text
        self.setAcceptHoverEvents(False)
        self.setAcceptedMouseButtons(Qt.Qt.NoButton)
        # Info text generally should appear over anything else rather than z-fighting
        self.setZValue(10)
        self.hide()

    def type(self):
        return ContextualInfoItem.QGRAPHICSITEM_TYPE

    def boundingRect(self):
        if self._text:
            self._update_picture()
            return self._bounding_rect
        else:
            return Qt.QRectF(0,0,1,1)

    def paint(self, qpainter, option, widget):
        self._update_picture()
        self._picture.play(qpainter)

    def return_to_fixed_position(self, view):
        """Maintain position self.FIXED_POSITION_IN_VIEW relative to view's top left corner."""
        topleft = self.FIXED_POSITION_IN_VIEW
        if view.mapFromScene(self.pos()) != topleft:
            self.setPos(view.mapToScene(topleft))

    def _update_picture(self):
        if self._picture is None:
            self._picture = Qt.QPicture()
            ppainter = Qt.QPainter()
            with ExitStack() as stack:
                ppainter.begin(self._picture)
                stack.callback(ppainter.end)
                # The Qt API calls required for formatting multiline text such that it can be rendered to
                # a path are private, as can be seen in the implementation of
                # QGraphicsSimpleTextItem::paint, pasted below.  (Specifically, QStackTextEngine is a private
                # component and thus not available through PyQt).
                #
                # void QGraphicsSimpleTextItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
                # {
                #     Q_UNUSED(widget);
                #     Q_D(QGraphicsSimpleTextItem);
                #
                #     painter->setFont(d->font);
                #
                #     QString tmp = d->text;
                #     tmp.replace(QLatin1Char('\n'), QChar::LineSeparator);
                #     QStackTextEngine engine(tmp, d->font);
                #     QTextLayout layout(&engine);
                #
                #     QPen p;
                #     p.setBrush(d->brush);
                #     painter->setPen(p);
                #     if (d->pen.style() == Qt::NoPen && d->brush.style() == Qt::SolidPattern) {
                #         painter->setBrush(Qt::NoBrush);
                #     } else {
                #         QTextLayout::FormatRange range;
                #         range.start = 0;
                #         range.length = layout.text().length();
                #         range.format.setTextOutline(d->pen);
                #         QList<QTextLayout::FormatRange> formats;
                #         formats.append(range);
                #         layout.setAdditionalFormats(formats);
                #     }
                #
                #     setupTextLayout(&layout);
                #     layout.draw(painter, QPointF(0, 0));
                #
                #     if (option->state & (QStyle::State_Selected | QStyle::State_HasFocus))
                #         qt_graphicsItem_highlightSelected(this, painter, option);
                # }
                #
                # We would just use QGraphicsSimpleTextItem directly, but it is not derived from QGraphicsObject, so
                # it lacks the QObject base class required for emitting signals.  It is not possible to add a QObject
                # base to a QGraphicsItem derivative in Python (it can be done in C++ - this is how QGraphicsObject
                # is implemented).
                #
                # Total lack of signal support is not acceptable; it would greatly complicate the task of
                # positioning a non-child item relative to a ContextualInfoItem.
                # 
                # However, it's pretty easy to use that very paint function to generate paint commands that
                # we cache in self._picture, so we do that.  For strings large enough that the relayout
                # performed on each refresh by QGraphicsSimpleTextItem exceeds the CPython interpreter overhead 
                # for initiating the QPicture replay, our paint function is faster.
                #
                # Additionally, QGraphicsTextItem is very featureful, has a QObject base, and would be the first
                # choice, but the one thing it can not do is outline text, so it's out.
                i = Qt.QGraphicsSimpleTextItem(self._text)
                i.setFont(self._font)
                # Disabling brush/pen via setBrush/Pen(Qt.QBrush/Pen(Qt.Qt.NoBrush/Pen)) ought to be more intelligent
                # than disablind via setting to transparent color.  However, using NoBrush or NoPen here seems to
                # cause extreme painting slowdowns on OS X.
                transparent_color = Qt.QColor(Qt.Qt.transparent)
                if self._pen is None or self._brush is None:
                    i.setPen(Qt.QPen(transparent_color) if self._pen is None else self._pen)
                    i.setBrush(Qt.QBrush(transparent_color) if self._brush is None else self._brush)
                    i.paint(ppainter, Qt.QStyleOptionGraphicsItem(), None)
                else:
                    # To ensure that character outlines never obscure the entirety of character interior, outline
                    # is drawn first and interior second.  If both brush and pen are nonempty, Qt draws interior first
                    # and outline second.
                    i.setBrush(Qt.QBrush(transparent_color))
                    i.setPen(self._pen)
                    i.paint(ppainter, Qt.QStyleOptionGraphicsItem(), None)
                    i.setBrush(self._brush)
                    i.setPen(Qt.QPen(transparent_color))
                    i.paint(ppainter, Qt.QStyleOptionGraphicsItem(), None)
                self._bounding_rect = i.boundingRect()

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, v):
        if self._text != v:
            if v:
                self.prepareGeometryChange()
            self._text = v
            self._picture = None
            if self._text:
                self.show()
                self.update()
            else:
                self.hide()
            self.text_changed.emit()

    @property
    def font(self):
        return self._font

    @font.setter
    def font(self, v):
        assert isinstance(v, Qt.QFont)
        self._font = v
        self._picture = None
        self.update()

    @property
    def pen(self):
        """The pen used to draw text outline to provide contrast against any background.  If None,
        outline is not drawn."""
        return self._pen

    @pen.setter
    def pen(self, v):
        assert isinstance(v, Qt.QPen) or v is None
        self._pen = v
        self._picture = None
        self.update()

    @property
    def brush(self):
        """The brush used to fill text.  If None, text is not filled."""
        return self._brush

    @brush.setter
    def brush(self, v):
        assert isinstance(v, Qt.QBrush) or v is None
        self._brush = v
        self._picture = None
        self.update()
