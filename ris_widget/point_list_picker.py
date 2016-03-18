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
import weakref
from . import om
from .shared_resources import UNIQUE_QGRAPHICSITEM_TYPE

class Point(Qt.QObject):
    changed = Qt.pyqtSignal(object)
    x_changed = Qt.pyqtSignal(object)
    y_changed = Qt.pyqtSignal(object)

    def __init__(self, x=0.0, y=0.0, parent=None):
        super().__init__(parent)
        self._x = x
        self._y = y

    def __repr__(self):
        return '[{}, {}]'.format(self._x, self._y)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        x = float(x)
        if x != self._x:
            self._x = x
            self.x_changed.emit(self)
            self.changed.emit(self)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        y = float(y)
        if y != self._y:
            self._y = y
            self.y_changed.emit(self)
            self.changed.emit(self)

class PointList(om.UniformSignalingList):
    def take_input_element(self, obj):
        if isinstance(obj, Point):
            if hasattr(self, '_list') and obj in self:
                return Point(obj.x, obj.y)
            return obj
        elif isinstance(obj, (Qt.QPointF, Qt.QPoint)):
            return Point(obj.x(), obj.y())
        else:
            i = iter(obj)
            return Point(next(i), next(i))

class PointItemMixin:
    def __init__(self, point, point_list_picker):
        self.point_wr = weakref.ref(point)
        self.point_list_picker_wr = weakref.ref(point_list_picker)
        self.item_change_handlers = {
            Qt.QGraphicsItem.ItemSceneHasChanged : self._on_item_scene_has_changed,
            Qt.QGraphicsItem.ItemParentHasChanged : self._on_item_parent_has_changed,
            Qt.QGraphicsItem.ItemPositionHasChanged : self._on_item_position_has_changed,
            Qt.QGraphicsItem.ItemSelectedHasChanged : self._on_item_selected_has_changed
        }

    def itemChange(self, change, value):
        if self.point_wr() is None or self.point_list_picker_wr() is None:
            return super().itemChange(change, value)
        return self.item_change_handlers.get(change, super().itemChange)(change, value)

    def keyPressEvent(self, event):
        if event.key() == Qt.Qt.Key_Delete and event.modifiers() == Qt.Qt.NoModifier:
            point_list_picker = self.point_list_picker_wr()
            if point_list_picker is not None:
                point_list_picker.delete_selected()

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.point_list_picker_wr()._on_point_item_focused(self.point_wr())

    def _on_item_scene_has_changed(self, change, value):
        self.point_list_picker_wr()._on_point_item_removed(self.point_wr())

    def _on_item_parent_has_changed(self, change, value):
        self.point_list_picker_wr()._on_point_item_removed(self.point_wr())

    def _on_item_position_has_changed(self, change, value):
        self.point_list_picker_wr()._on_point_item_moved(self.point_wr())

    def _on_item_selected_has_changed(self, change, value):
        self.point_list_picker_wr()._on_point_selected_has_changed(self.point_wr(), self.isSelected())

class PointListRectItem(PointItemMixin, Qt.QGraphicsRectItem):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()
    def __init__(self, point, point_list_picker, parent_item=None):
        Qt.QGraphicsRectItem.__init__(self, parent_item)
        PointItemMixin.__init__(self, point, point_list_picker)
        self.setRect(-7,-7,15,15)

class _Empty:
    pass

class PointListPicker(Qt.QGraphicsObject):
    """PointListPicker is a utility class intended to be used directly or extended via inheritance.  A plain, unextended
    PointListPicker is instantiated with a view and an item to be annotated in that view as constructor parameters.

    After instantiation, right clicking in the view appends the item-relative coordinate clicked to .points, which is
    a replaceable SignalingList, and this results in creation of an attached PointListRectItem which becomes visible
    in the view.

    In general, modification to .points results in corresponding updates to the point items in the view, and modification
    to the point items in the view results in updates to .points:

    * Dragging or otherwise moving a point item causes the associated entry in .points to update, and modifying an
      entry in .points causes its point item to move within the view to the newly assinged position.

    * Deleting a point item, which can be done by hitting the delete key with the item in focus (click on it to give it
      focus), results in the associated .points entry being removed.  Deleting an entry in .points causes the associated
      point item to be removed.

    * Inserting an entry into .points causes a corresponding point item to appear in the view.  Any iterable of at least
      two elements convertable to float may be inserted into .points.  Qt.QPoint and Qt.QPointF instances are also
      accepted.

    Although PointListPicker's implementation is not trivial, using it and extending it very much are.  See
    RisWidget.make_point_list_picker(..) for a basic usage example and RisWidget.make_point_list_picker_with_table_view(..)
    for something slightly more advanced.  For information regarding extending PointListPicker, see 
    examples.poly_line_picker for a basic example and examples.quadratic_compound_bezier_picker for one that's more
    ambitious."""
    point_list_replaced = Qt.pyqtSignal(object)
    point_list_contents_changed = Qt.pyqtSignal()
    point_is_selected_changed = Qt.pyqtSignal(object, bool)
    point_focused = Qt.pyqtSignal(object)

    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()
    POINT_LIST_TYPE = PointList

    def __init__(self, general_view, parent_item, points=None):
        super().__init__(parent_item)
        self.view = general_view
        self.view.scene_region_changed.connect(self._on_scene_region_changed)
        self.PointListType = self.POINT_LIST_TYPE
        self.pen = Qt.QPen(Qt.Qt.red)
        self.pen.setWidth(2)
        color = Qt.QColor(Qt.Qt.yellow)
        color.setAlphaF(0.5)
        self.brush = Qt.QBrush(color)
        self.brush_selected = Qt.QBrush(Qt.QColor(255, 0, 255, 127))
        self._ignore_point_and_item_moved = False
        self._ignore_point_and_item_removed = False
        self.point_items = dict()
        self._points = None
        self.points = self.PointListType() if points is None else points
        self.parentItem().installSceneEventFilter(self)

    def instantiate_point_item(self, point, idx):
        item = PointListRectItem(point, self, self.parentItem())
        item.setPen(self.pen)
        item.setBrush(self.brush)
        return item

    def sceneEventFilter(self, watched, event):
        if watched is self.parentItem() and event.type() == Qt.QEvent.GraphicsSceneMousePress and event.button() == Qt.Qt.RightButton:
            self._points.append(event.pos())
            return True
        return False

    def delete_selected(self):
        # "run" as in consecutive indexes specified as range rather than individually
        runs = []
        run_start_idx = None
        run_end_idx = None
        for idx, point in enumerate(self._points):
            item = self.point_items[point]
            if item.isSelected():
                if run_start_idx is None:
                    run_end_idx = run_start_idx = idx
                elif idx - run_end_idx == 1:
                    run_end_idx = idx
                else:
                    runs.append((run_start_idx, run_end_idx))
                    run_end_idx = run_start_idx = idx
        if run_start_idx is not None:
            runs.append((run_start_idx, run_end_idx))
        for run_start_idx, run_end_idx in reversed(runs):
            del self._points[run_start_idx:run_end_idx+1]

    def boundingRect(self):
        return Qt.QRectF()

    def paint(self, QPainter, QStyleOptionGraphicsItem, QWidget_widget=None):
        pass

    def type(self):
        return self.QGRAPHICSITEM_TYPE

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        if not isinstance(points, self.PointListType):
            points = self.PointListType(points)
        self._detach_point_list()
        self._points = points
        self._attach_point_list()
        self.point_list_replaced.emit(self)
        self.point_list_contents_changed.emit()

    def _attach_point(self, point, idx):
        assert point not in self.point_items
        point.changed.connect(self._on_point_changed)
        point_item = self.instantiate_point_item(point, idx)
        self.point_items[point] = point_item
        point_item.setScale(1 / self.view.transform().m22())
        point_item.setPos(point.x, point.y)
        flags = point_item.flags()
        point_item.setFlags(
            flags |
            Qt.QGraphicsItem.ItemIsFocusable | # Necessary in order for item to receive keyboard events
            Qt.QGraphicsItem.ItemIsSelectable |
            Qt.QGraphicsItem.ItemIsMovable |
            Qt.QGraphicsItem.ItemSendsGeometryChanges
        )

    def _detach_point(self, point):
        self._ignore_point_and_item_removed = True
        try:
            try:
                point.changed.disconnect(self._on_point_changed)
            except TypeError:
                pass
            try:
                point_item = self.point_items.pop(point)
                point_item.point_wr = weakref.ref(_Empty())
                scene = point_item.scene()
                if scene is not None:
                    scene.removeItem(point_item)
            except KeyError:
                pass
        finally:
            self._ignore_point_and_item_removed = False

    def _attach_point_list(self):
        self._points.inserted.connect(self._on_points_inserted)
        self._points.removed.connect(self._on_points_removed)
        self._points.replaced.connect(self._on_points_replaced)
        for idx, point in enumerate(self._points):
            self._attach_point(point, idx)

    def _detach_point_list(self):
        if self._points is None:
            return
        sig_hands = [
            (self._points.inserted, self._on_points_inserted),
            (self._points.removed, self._on_points_removed),
            (self._points.replaced, self._on_points_replaced)
        ]
        # If we somehow enter a bad state in which not all signals are connected as we expect, ignoring disconnection
        # failures on assignment of a new list allows escape from the bad state by assignment to .points
        for sig, hand in sig_hands:
            try:
                sig.disconnect(hand)
            except TypeError:
                pass
        for point in self._points:
            self._detach_point(point)

    def _on_points_inserted(self, idx_, points):
        for idx, point in enumerate(points, idx_):
            self._attach_point(point, idx)
        self.point_list_contents_changed.emit()

    def _on_points_removed(self, idxs, points):
        if self._ignore_point_and_item_removed:
            return
        for point in points:
            self._detach_point(point)
        self.point_list_contents_changed.emit()

    def _on_points_replaced(self, idxs, replaced_points, points):
        for point in replaced_points:
            self._detach_point(point)
        for idx, point in zip(idxs, points):
            self._attach_point(point)
        self.point_list_contents_changed.emit()

    def _on_point_focused(self, point):
        self.point_focused.emit(point)

    def _on_point_changed(self, point):
        if self._ignore_point_and_item_moved:
            return
        self._ignore_point_and_item_moved = True
        try:
            point_item = self.point_items[point]
            point_item.setPos(point.x, point.y)
        finally:
            self.point_list_contents_changed.emit()
            self._ignore_point_and_item_moved = False

    def _on_point_item_removed(self, point):
        if self._ignore_point_and_item_removed:
            return
        self._ignore_point_and_item_removed = True
        try:
            self._detach_point(point)
            del self._points[self._points.index(point)]
        finally:
            self.point_list_contents_changed.emit()
            self._ignore_point_and_item_removed = False

    def _on_point_item_moved(self, point):
        if self._ignore_point_and_item_moved:
            return
        self._ignore_point_and_item_moved = True
        try:
            pos = self.point_items[point].pos()
            point.x, point.y = pos.x(), pos.y()
        finally:
            self.point_list_contents_changed.emit()
            self._ignore_point_and_item_moved = False

    def _on_point_selected_has_changed(self, point, isSelected):
        self.point_items[point].setBrush(self.brush_selected if isSelected else self.brush)
        self.point_is_selected_changed.emit(point, isSelected)

    def _on_point_item_focused(self, point):
        self.point_focused.emit(point)

    def _on_scene_region_changed(self, view):
        assert view is self.view
        scale = 1 / self.view.transform().m22()
        for point_item in self.point_items.values():
            point_item.setScale(scale)