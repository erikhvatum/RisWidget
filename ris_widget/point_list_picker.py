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
        self._container_ref_count = 0
        self._container_id = None

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
            Qt.QGraphicsItem.ItemPositionHasChanged : self._on_item_position_has_changed
        }

    def itemChange(self, change, value):
        if self.point_wr() is None or self.point_list_picker_wr() is None:
            return super().itemChange(change, value)
        return self.item_change_handlers.get(change, super().itemChange)(change, value)

    def keyPressEvent(self, event):
        if event.key() == Qt.Qt.Key_Delete:
            self.scene().removeItem(self)
        else:
            event.ignore()

    @property
    def center_pos(self):
        return self.boundingRect().center() + self.pos()

    @center_pos.setter
    def center_pos(self, pos):
        self.setPos(pos - self.boundingRect().center())

    def _on_item_scene_has_changed(self, change, value):
        self.point_list_picker_wr()._on_point_item_removed(self.point_wr())

    def _on_item_parent_has_changed(self, change, value):
        self.point_list_picker_wr()._on_point_item_removed(self.point_wr())

    def _on_item_position_has_changed(self, change, value):
        self.point_list_picker_wr()._on_point_item_moved(self.point_wr())

class PointListRectItem(PointItemMixin, Qt.QGraphicsRectItem):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()
    def __init__(self, point, point_list_picker, parent_item=None):
        Qt.QGraphicsRectItem.__init__(self, parent_item)
        PointItemMixin.__init__(self, point, point_list_picker)
        self.setRect(0,0,15,15)

class _Empty:
    pass

class PointListPicker(Qt.QObject):
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

    def __init__(self, general_view, point_item_parent, points=None, PointListType=PointList, parent=None):
        super().__init__(parent)
        self.view = general_view
        self.view.scene_region_changed.connect(self._on_scene_region_changed)
        self.point_item_parent = point_item_parent
        self.PointListType = PointListType
        self._ignore_point_and_item_moved = False
        self._ignore_point_and_item_removed = False
        self.point_items = dict()
        self._points = None
        self.points = self.PointListType() if points is None else points
        self.view.mouse_event_signal.connect(self._on_mouse_event_in_view)

    def instantiate_point_item(self, point, idx):
        item = PointListRectItem(point, self, self.point_item_parent)
        pen = Qt.QPen(Qt.Qt.red)
        pen.setWidth(2)
        item.setPen(pen)
        item.setBrush(Qt.QBrush(Qt.Qt.yellow))
        return item

    def _on_mouse_event_in_view(self, event_type, event, scene_pos):
        if event_type == 'press' and event.buttons() == Qt.Qt.RightButton:
            self._points.append(self.point_item_parent.mapFromScene(scene_pos))
            event.accept()

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

    def _attach_point(self, point, idx):
        if point._container_ref_count > 0:
            assert point in self.point_items
            if point._container_id != id(self):
                raise ValueError('A Point instance may not be a member of two different PointLists.')
        else:
            assert point not in self.point_items
            point._container_ref_count = 1
            point._container_id = id(self)
            point.changed.connect(self._on_point_changed)
            point_item = self.instantiate_point_item(point, idx)
            self.point_items[point] = point_item
            point_item.setScale(1 / self.view.transform().m22())
            point_item.center_pos = Qt.QPointF(point.x, point.y)
            flags = point_item.flags()
            point_item.setFlags(
                flags |
                Qt.QGraphicsItem.ItemIsSelectable |
                Qt.QGraphicsItem.ItemIsFocusable |
                Qt.QGraphicsItem.ItemIsMovable |
                Qt.QGraphicsItem.ItemSendsGeometryChanges
            )

    def _detach_point(self, point):
        assert point._container_id == id(self)
        assert point._container_ref_count > 0
        point._container_ref_count -= 1
        if point._container_ref_count == 0:
            self._ignore_point_and_item_removed = True
            try:
                point._container_id = None
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
        # pacs = []
        # if idx_ > 0:
        #     if len(points)
        # if pacs:
        #     self.point_adjacency_changes.emit(pacs)

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

    def _on_point_changed(self, point):
        if self._ignore_point_and_item_moved:
            return
        self._ignore_point_and_item_moved = True
        try:
            point_item = self.point_items[point]
            point_item.center_pos = Qt.QPointF(point.x, point.y)
        finally:
            self._ignore_point_and_item_moved = False

    def _on_point_item_removed(self, point):
        if self._ignore_point_and_item_removed:
            return
        assert point._container_id == id(self)
        self._ignore_point_and_item_removed = True
        try:
            while point._container_ref_count > 0:
                self._detach_point(point)
        finally:
            self.point_list_contents_changed.emit()
            self._ignore_point_and_item_removed = False

    def _on_point_item_moved(self, point):
        if self._ignore_point_and_item_moved:
            return
        self._ignore_point_and_item_moved = True
        try:
            pos = self.point_items[point].center_pos
            point.x, point.y = pos.x(), pos.y()
        finally:
            self._ignore_point_and_item_moved = False

    def _on_scene_region_changed(self, view):
        assert view is self.view
        scale = 1 / self.view.transform().m22()
        for point_item in self.point_items.values():
            point_item.setScale(scale)