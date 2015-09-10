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

from PyQt5 import Qt
from .. import om
from ..image import Image
from ..layer import Layer
from ..shared_resources import FREEIMAGE, ICONS
from .progress_thread_pool import ProgressThreadPool, Task, TaskStatus

class Flipbook(Qt.QWidget):
    """Flipbook: a widget containing a list box showing the name property values of the elements of its pages property.
    Changing which row is selected in the list box causes the current_page_changed signal to be emitted with the newly
    selected page's index and the page itself as parameters.

    The pages property of Flipbook is an SignalingList, a container with a list interface, containing a sequence 
    of elements.  The pages property should be manipulated via the standard list interface, which it implements
    completely.  So, for example, if you have a Flipbook of Images and wish to add an Image to the end of that Flipbook:
    
    image_flipbook.pages.append(Image(numpy.zeros((400,400,3), dtype=numpy.uint8)))

    Signals:
    * current_page_changed(Flipbook instance, page #)"""
    current_page_changed = Qt.pyqtSignal(object, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        l = Qt.QVBoxLayout()
        self.setLayout(l)
        self.pages_model = PagesModel(om.SignalingList())
        self.pages_model.handle_dropped_files = self._handle_dropped_files
        self.pages_view = PagesView(self.pages_model)
        self.pages_view.setModel(self.pages_model)
        self.pages_view.selectionModel().currentRowChanged.connect(self._on_pages_current_idx_changed)
        l.addLayout(self._make_behavior_grid())
        l.addWidget(self.pages_view)
        self.progress_thread_pool = None

    def _make_behavior_grid(self):
        icons = ICONS()
        l = Qt.QGridLayout()

        w = self.stack_label = Qt.QLabel()
        w.setPixmap(icons['layer_stack_icon'].pixmap(60, 60))
        w.setToolTip('Layer stack')
        l.addWidget(w, 0, 1, Qt.Qt.AlignCenter)

        w = self.layer_label = Qt.QLabel()
        w.setPixmap(icons['layer_icon'].pixmap(60, 60))
        w.setToolTip('Layer')
        l.addWidget(w, 0, 2, Qt.Qt.AlignCenter)

        w = self.image_label = Qt.QLabel()
        w.setPixmap(icons['image_icon'].pixmap(60, 60))
        w.setToolTip('Image')
        l.addWidget(w, 0, 3, Qt.Qt.AlignCenter)


        bg = self.pages_model.drop_as_button_group = self.drop_as_button_group = Qt.QButtonGroup(self)
        bg.setExclusive(True)
        w = self.drop_as_label = Qt.QLabel('Drop as:')
        l.addWidget(w, 1, 0, Qt.Qt.AlignLeft)

        w = self.drop_as_stack_radio = Qt.QRadioButton()
        bg.addButton(w, 0)
        w.setToolTip('Drop as Layer stack')
        l.addWidget(w, 1, 1, Qt.Qt.AlignCenter)

        w = self.drop_as_layer_radio = Qt.QRadioButton()
        bg.addButton(w, 1)
        w.setChecked(True)
        w.setToolTip('Drop as Layer')
        l.addWidget(w, 1, 2, Qt.Qt.AlignCenter)

        w = self.drop_as_image_radio = Qt.QRadioButton()
        bg.addButton(w, 2)
        w.setToolTip('Drop as Image')
        l.addWidget(w, 1, 3, Qt.Qt.AlignCenter)


        bg = self.pages_model.show_as_button_group = self.show_as_button_group = Qt.QButtonGroup(self)
        bg.setExclusive(True)
        w = self.show_as_label = Qt.QLabel('Show as:')
        l.addWidget(w, 2, 0, Qt.Qt.AlignLeft)

        w = self.show_as_stack_radio = Qt.QRadioButton()
        bg.addButton(w, 0)
        w.setChecked(True)
        w.setToolTip('Show as Layer stack')
        l.addWidget(w, 2, 1, Qt.Qt.AlignCenter)

        w = self.show_as_layer_radio = Qt.QRadioButton()
        bg.addButton(w, 1)
        w.setToolTip('Show as Layer')
        l.addWidget(w, 2, 2, Qt.Qt.AlignCenter)

        w = self.show_as_image_radio = Qt.QRadioButton()
        bg.addButton(w, 2)
        w.setToolTip('Show as Image')
        l.addWidget(w, 2, 3, Qt.Qt.AlignCenter)

        return l

    def _handle_dropped_files(self, fpaths, dst_row, dst_column, dst_parent):
        freeimage = FREEIMAGE(show_messagebox_on_error=True, error_messagebox_owner=None)
        if freeimage is None:
            return False
        drop_as = self.drop_as_button_group.checkedId()
        if drop_as > 0 and self.progress_thread_pool is None:
            self.progress_thread_pool = ProgressThreadPool()
            self.progress_thread_pool.task_status_changed.connect(self._on_progress_thread_pool_task_status_changed)
            self.progress_thread_pool.all_tasks_retired.connect(self._on_all_progress_thread_pool_tasks_retired)
            self.layout().addWidget(self.progress_thread_pool)
        if drop_as == 0:
            layers = [Layer(Image(freeimage.read(fpath_str), name=fpath_str), name=fpath_str) for fpath_str in (str(fpath) for fpath in fpaths)]
            self.pages.insert(dst_row, om.SignalingList(layers))
        elif drop_as == 1:
            def read_layer(fpath):
                fpath = str(fpath)
                return Layer(Image(freeimage.read(fpath), name=fpath), name=fpath)
            tasks = []
            for fpath in fpaths:
                task = Task(read_layer, fpath)
                task.drop_as = drop_as
                tasks.append(task)
            self.pages[dst_row:dst_row] = tasks
            self.progress_thread_pool.tasks.extend(tasks)
        else:
            def read_image(fpath):
                fpath = str(fpath)
                return Image(freeimage.read(fpath_str), name=fpath_str)
            tasks = [self.progress_thread_pool.submit(read_image, fpath) for fpath in fpaths]
            self.pages[dst_row:dst_row] = tasks
        return True

    def _on_progress_thread_pool_task_status_changed(self, task, old_status):
        try:
            element_inst_count = self.pages_model._instance_counts[task]
        except KeyError:
            # We received queued notification informing us that something already removed from Tasks
            # changed to Completed status before being removed.
            return
        if task.status is TaskStatus.Completed:
            next_idx = 0
            pages = self.pages
            for _ in range(element_inst_count):
                idx = pages.index(task, next_idx)
                next_idx = idx + 1
                pages[idx] = task.result
                task._progress_thread_pool = None
        else:
            next_idx = 0
            pages = self.pages
            m = self.pages_model
            for _ in range(element_inst_count):
                idx = pages.index(task, next_idx)
                next_idx = idx + 1
                m.dataChanged.emit(m.createIndex(idx, 0), m.createIndex(idx, 0))

    def _on_all_progress_thread_pool_tasks_retired(self):
        self.layout().removeWidget(self.progress_thread_pool)
        self.progress_thread_pool.deleteLater()
        self.progress_thread_pool = None

    @property
    def pages(self):
        return self.pages_model.signaling_list

    @pages.setter
    def pages(self, pages):
        assert isinstance(pages, om.SignalingList)
        self.pages_model.signaling_list = pages
        self.current_page_changed.emit(self, self.selectionModel().currentIndex().row())

    def _on_pages_current_idx_changed(self, midx, old_midx):
        self.current_page_changed.emit(self, midx.row())

@om.item_view_shortcuts.with_selected_rows_deletion_shortcut
class PagesView(Qt.QTableView):
    def __init__(self, pages_model, parent=None):
        super().__init__(parent)
        self.setModel(pages_model)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setHighlightSections(False)
        self.verticalHeader().setSectionsClickable(False)
        self.verticalHeader().setHighlightSections(False)
        self.verticalHeader().setSectionsClickable(False)
        self.setTextElideMode(Qt.Qt.ElideMiddle)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(Qt.QAbstractItemView.DragDrop)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(Qt.Qt.LinkAction)
        self.horizontalHeader().setSectionResizeMode(Qt.QHeaderView.ResizeToContents)
        self.setSelectionBehavior(Qt.QAbstractItemView.SelectRows)
        self.setSelectionMode(Qt.QAbstractItemView.ExtendedSelection)

class PagesModel(om.signaling_list.DragDropModelBehavior, om.signaling_list.PropertyTableModel):
    PROPERTIES = (
        'name',
        )

    def __init__(self, signaling_list=None, parent=None):
        super().__init__(self.PROPERTIES, signaling_list, parent)
        icons = ICONS()
        self.icons = {
            Image : icons['image_icon'],
            Layer : icons['layer_icon'],
            om.SignalingList : icons['layer_stack_icon']
            }

    def flags(self, midx):
        if midx.isValid() and midx.column() == self.PROPERTIES.index('name'):
            element = self.signaling_list[midx.row()]
            if isinstance(element, Task):
                return super().flags(midx) & ~Qt.Qt.ItemIsEnabled & ~Qt.Qt.ItemIsEditable
        return super().flags(midx)

    def data(self, midx, role=Qt.Qt.DisplayRole):
        if midx.isValid() and midx.column() == self.PROPERTIES.index('name'):
            element = self.signaling_list[midx.row()]
            if element is None:
                return Qt.QVariant()
            if isinstance(element, Task):
                if role == Qt.Qt.DecorationRole:
                    return Qt.QVariant(ICONS()[('layer_stack_icon', 'layer_icon', 'image_icon')[element.drop_as]])
                if role == Qt.Qt.DisplayRole:
                    return Qt.QVariant('({}) {}'.format(element.status.name, element.callable_va[0]))
            if role == Qt.Qt.DecorationRole:
                return Qt.QVariant(self.icons.get(type(element)))
        return super().data(midx, role)
