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

import json
import numpy
from pathlib import Path
from PyQt5 import Qt
from ..image import Image
from .flipbook import ImageList, _X_THREAD_ADD_IMAGE_FILES_EVENT, _X_THREAD_ADD_IMAGE_FILE_STACKS_EVENT
from .flipbook import _XThreadAddImageFilesEvent, _XThreadAddImageFileStacksEvent
from ..layer import Layer
from ..layers import LayerList
from .. import om
from .progress_thread_pool import ProgressThreadPool, Task, TaskStatus
from ..shared_resources import FREEIMAGE

class LayerStackPageList(om.UniformSignalingList):
    def take_input_element(self, obj):
        if isinstance(obj, (LayerList, Task)):
            return obj
        if isinstance(obj, (numpy.ndarray, Image, Layer)):
            obj = [obj]
        return LayerList(obj)

#TODO: feed entirety of .pages to ProgressThreadPool and make ProgressThreadPool entirely ignore non-Task elements
#rather than raising exceptions
class LayerStackFlipbook(Qt.QWidget):
    # TODO: update the following docstring
    """"""

    def __init__(self, layer_stack, parent=None):
        super().__init__(parent)
        self.layer_stack = layer_stack
        l = Qt.QVBoxLayout()
        self.setLayout(l)
        self.pages_model = LayerStackPagesModel(LayerStackPageList())
        self.pages_model.handle_dropped_files = self._handle_dropped_files
        self.pages_model.rowsInserted.connect(self._on_model_rows_inserted, Qt.Qt.QueuedConnection)
        self.pages_view = LayerStackPagesView(self.pages_model)
        self.pages_view.setModel(self.pages_model)
        self.pages_view.selectionModel().currentRowChanged.connect(self._on_page_focus_changed)
        l.addWidget(self.pages_view)
        self.progress_thread_pool = None

    def add_json_and_image_files(self, fpaths):
        if Qt.QThread.currentThread() is Qt.QApplication.instance().thread():
            self._add_json_and_image_files(fpaths)
        else:
            Qt.QApplication.instance().postEvent(self, _XThreadAddImageFilesEvent(fpaths))

    def add_image_file_stacks(self, image_fpath_stacks):
        if Qt.QThread.currentThread() is Qt.QApplication.instance().thread():
            self._add_image_file_stacks(image_fpath_stacks)
        else:
            Qt.QApplication.instance().postEvent(self, _XThreadAddImageFileStacksEvent(image_fpath_stacks))

    def event(self, event):
        if event.type() == _X_THREAD_ADD_IMAGE_FILES_EVENT:
            assert isinstance(event, _XThreadAddImageFilesEvent)
            self._add_image_files(event.image_fpaths)
            return True
        if event.type() == _X_THREAD_ADD_IMAGE_FILE_STACKS_EVENT:
            assert isinstance(event, _XThreadAddImageFileStacksEvent)
            self._add_image_file_stacks(event.image_fpath_stacks)
            return True
        return super().event(event)

    @property
    def pages(self):
        return self.pages_model.signaling_list

    @pages.setter
    def pages(self, pages):
        if not isinstance(pages, LayerStackPageList):
            pages = LayerStackPageList(pages)
        self.pages_model.signaling_list = pages
        self._on_page_focus_changed()

    @property
    def focused_page_idx(self):
        midx = self.pages_view.selectionModel().currentIndex()
        if midx.isValid():
            return midx.row()

    @property
    def focused_page(self):
        focused_page_idx = self.focused_page_idx
        if focused_page_idx is not None:
            return self.pages[focused_page_idx]

    def ensure_page_selected(self):
        """If no page is selected and .pages is not empty:
           If there is a "current" page, IE highlighted but not selected, select it.
           If there is no "current" page, make .pages[0] current and select it."""
        if not self.pages:
            return
        sm = self.pages_view.selectionModel()
        if not sm.currentIndex().isValid():
            sm.setCurrentIndex(
                    self.pages_model.index(0, 0),
                    Qt.QItemSelectionModel.SelectCurrent | Qt.QItemSelectionModel.Rows)
        if len(sm.selectedRows()) == 0:
            sm.select(
                sm.currentIndex(),
                Qt.QItemSelectionModel.SelectCurrent | Qt.QItemSelectionModel.Rows)

    def _on_model_rows_inserted(self, _, __, ___):
        self.pages_view.resizeRowsToContents()

    def _on_page_focus_changed(self, midx=None, old_midx=None):
        focused_page = self.focused_page
        if not isinstance(focused_page, LayerList):
            return
        self.layer_stack.layers = focused_page

    def _add_json_and_image_files(self, fpaths, dst_row=None):
        irs = self._load_json_and_make_image_readers(fpaths)
        if irs is not None:
            if dst_row is None:
                self.pages.extend(irs)
            else:
                self.pages[dst_row:dst_row] = irs
            self.ensure_page_selected()

    def _add_image_file_stacks(self, image_fpath_stacks, dst_row=None):
        isrs = self._make_image_stack_readers(image_fpath_stacks)
        if isrs is not None:
            if dst_row is None:
                self.pages.extend(isrs)
            else:
                self.pages[dst_row:dst_row] = isrs
            self.ensure_page_selected()

    def _handle_dropped_files(self, fpaths, dst_row, dst_column, dst_parent):
        self._add_json_and_image_files(fpaths, dst_row)
        return True

    def _on_progress_thread_pool_task_status_changed(self, task, old_status):
        try:
            element_inst_count = self.pages_model._instance_counts[task]
        except KeyError:
            # We received queued notification informing us that something already removed from Tasks
            # changed to Completed status before being removed.
            return
        if task.status is TaskStatus.Completed:
            pages = self.pages
            name = task.result_name
            if isinstance(task.result, numpy.ndarray):
                image_stack = ImageList([Image(task.result, name=name)])
            else:
                image_stack = ImageList(Image(image_data, name=image_name) for image_data, image_name in task.result)
            image_stack.name = name
            task._progress_thread_pool = None
            next_idx = 0
            current_midx = self.pages_view.selectionModel().currentIndex()
            current_idx = current_midx.row() if current_midx.isValid() else None
            for _ in range(element_inst_count):
                idx = pages.index(task, next_idx)
                next_idx = idx + 1
                pages[idx] = image_stack
                if idx == current_idx:
                    self._on_page_focus_changed()
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

    def _load_json_and_make_image_readers(self, fpaths):
        assert Qt.QThread.currentThread() is Qt.QApplication.instance().thread()
        freeimage_load_failed = False
        freeimage = None
        pages = LayerStackPageList()
        for fpath in fpaths:
            fpath = Path(fpath)
            if fpath.suffix in ('.txt', '.json'):
                try:
                    with open(str(fpath), 'r') as f:
                        prop_stackds = json.load(f)['layer property stacks']
                        for prop_stackd in prop_stackds:
                            layers = LayerList()
                            layers.name = prop_stackd['name']
                            for props in prop_stackd['layers']:
                                layer = Layer()
                                for pname, pval, in props.items():
                                    setattr(layer, pname, pval)
                                layers.append(layer)
                            pages.append(layers)
                except (FileNotFoundError, ValueError) as e:
                    Qt.QMessageBox.information(None, 'JSON Error', '{} : {}'.format(type(e).__name__, e))
            else:
                if freeimage_load_failed:
                    continue
                if freeimage is None:
                    freeimage = FREEIMAGE(show_messagebox_on_error=True)
                    if freeimage is None:
                        freeimage_load_failed = True
                        continue
                if self.progress_thread_pool is None:
                    self.progress_thread_pool = ProgressThreadPool()
                    self.progress_thread_pool.task_status_changed.connect(self._on_progress_thread_pool_task_status_changed)
                    self.progress_thread_pool.all_tasks_retired.connect(self._on_all_progress_thread_pool_tasks_retired)
                    self.layout().addWidget(self.progress_thread_pool)
                reader = self.progress_thread_pool.submit(freeimage.read, str(fpath))
                reader.result_name = str(fpath)
                pages.append(reader)
        return pages

    def _make_image_stack_readers(self, image_fpath_stacks):
        assert Qt.QThread.currentThread() is Qt.QApplication.instance().thread()
        freeimage = FREEIMAGE(show_messagebox_on_error=True, error_messagebox_owner=self)
        if freeimage:
            if self.progress_thread_pool is None:
                self.progress_thread_pool = ProgressThreadPool()
                self.progress_thread_pool.task_status_changed.connect(self._on_progress_thread_pool_task_status_changed)
                self.progress_thread_pool.all_tasks_retired.connect(self._on_all_progress_thread_pool_tasks_retired)
                self.layout().addWidget(self.progress_thread_pool)
            stack_readers = []
            def read_stack(image_fpaths):
                return [(freeimage.read(str(image_fpath)), str(image_fpath)) for image_fpath in image_fpaths]
            for image_fpaths in image_fpath_stacks:
                stack_reader = self.progress_thread_pool.submit(read_stack, image_fpaths)
                stack_reader.result_name = ', '.join(Path(image_fpath).stem for image_fpath in image_fpaths)
                stack_readers.append(stack_reader)
            return stack_readers

@om.item_view_shortcuts.with_selected_rows_deletion_shortcut
class LayerStackPagesView(Qt.QTableView):
    def __init__(self, pages_model, parent=None):
        super().__init__(parent)
        self.setModel(pages_model)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setHighlightSections(False)
        self.horizontalHeader().setSectionsClickable(False)
        self.verticalHeader().setHighlightSections(False)
        self.verticalHeader().setSectionsClickable(False)
        self.setTextElideMode(Qt.Qt.ElideLeft)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(Qt.QAbstractItemView.DragDrop)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(Qt.Qt.LinkAction)
        self.horizontalHeader().setSectionResizeMode(Qt.QHeaderView.ResizeToContents)
        self.setSelectionBehavior(Qt.QAbstractItemView.SelectRows)
        self.setSelectionMode(Qt.QAbstractItemView.ExtendedSelection)
        self.setWordWrap(False)

class LayerStackPagesModelDragDropBehavior(om.signaling_list.DragDropModelBehavior):
    def can_drop_rows(self, src_model, src_rows, dst_row, dst_column, dst_parent):
        return isinstance(src_model, LayerStackPagesModel)

    def mimeTypes(self):
        ret = super().mimeTypes()
        ret.append('text/plain')
        return ret

    def mimeData(self, midxs):
        mime_data = super().mimeData(midxs)
        pages = self.signaling_list
        lpsss = \
        [
            {
                'name' : page.name,
                'layers' :
                [
                    layer.get_savable_properties_dict() for layer in page
                ]
            }
            for page in 
            (
                pages[midx.row()] for midx in midxs if midx.isValid()
            ) if isinstance(page, LayerList)
        ]
        mime_data.setText(json.dumps({'layer property stacks' : lpsss}, ensure_ascii=False, indent=1))
        return mime_data

    # def canDropMimeData(self, mime_data, drop_action, row, column, parent):
    #     if not mime_data.hasUrls() and mime_data.hasText():
    #         return self._from_json(mime_data.text()) is not None
    #     return super().canDropMimeData(mime_data, drop_action, row, column, parent)

    # def dropMimeData(self, mime_data, drop_action, row, column, parent):
    #     if mime_data.hasUrls:
    #         print(mime_data.urls())

    def _from_json(self, json_str):
        try:
            lpsss = json.loads(json_str)['layer property stacks']
            pages = LayerStackPageList()
            for lpss in lpsss:
                layers = LayerList()
                layers.name = lpss['name']
                for lps in lpss:
                    layer = Layer()
                    for lpn, lpv in lpss['layers'].items():
                        setattr(layer, lpn, lpv)
                    layers.append(layer)
                pages.append(layers)
        except:
            return
        return pages

class LayerStackPagesModel(LayerStackPagesModelDragDropBehavior, om.signaling_list.PropertyTableModel):
    PROPERTIES = (
        'name',
        )

    def __init__(self, signaling_list=None, parent=None):
        super().__init__(self.PROPERTIES, signaling_list, parent)

    def flags(self, midx):
        if midx.isValid() and midx.column() == self.PROPERTIES.index('name'):
            element = self.signaling_list[midx.row()]
            if isinstance(element, Task):
                return super().flags(midx) & ~Qt.Qt.ItemIsEditable
        return super().flags(midx)

    def data(self, midx, role=Qt.Qt.DisplayRole):
        if midx.isValid() and midx.column() == self.PROPERTIES.index('name'):
            element = self.signaling_list[midx.row()]
            if isinstance(element, map):
                pass
            if element is None:
                return Qt.QVariant()
            if isinstance(element, Task):
                if role == Qt.Qt.DisplayRole:
                    return Qt.QVariant('{} ({})'.format(element.result_name, element.status.name))
                if role == Qt.Qt.ForegroundRole:
                    return Qt.QVariant(Qt.QApplication.palette().brush(Qt.QPalette.Disabled, Qt.QPalette.WindowText))
        return super().data(midx, role)

