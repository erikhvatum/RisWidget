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

import numpy
from pathlib import Path
import collections

from PyQt5 import Qt

from .. import om
from ..image import Image
from ..shared_resources import FREEIMAGE
from .progress_thread_pool import ProgressThreadPool, Task, TaskStatus

class ImageList(om.UniformSignalingList):
    def take_input_element(self, obj):
        return obj if isinstance(obj, Image) else Image(obj)

class PageList(om.UniformSignalingList):
    def take_input_element(self, obj):
        if isinstance(obj, (ImageList, Task)):
            return obj
        if isinstance(obj, (numpy.ndarray, Image)):
            ret = ImageList((obj,))
            if hasattr(obj, 'name'):
                ret.name = obj.name
            return ret
        return ImageList(obj)

_X_THREAD_ADD_IMAGE_FILES_EVENT = Qt.QEvent.registerEventType()

class _XThreadAddImageFilesEvent(Qt.QEvent):
    def __init__(self, image_fpaths):
        super().__init__(_X_THREAD_ADD_IMAGE_FILES_EVENT)
        self.image_fpaths = image_fpaths

#TODO: feed entirety of .pages to ProgressThreadPool and make ProgressThreadPool entirely ignore non-Task elements
#rather than raising exceptions
class Flipbook(Qt.QWidget):
    # TODO: update the following docstring
    """"""

    def __init__(self, layer_stack, parent=None):
        super().__init__(parent)
        self.layer_stack = layer_stack
        l = Qt.QVBoxLayout()
        self.setLayout(l)
        self.pages_model = PagesModel(PageList())
        self.pages_model.handle_dropped_files = self._handle_dropped_files
        self.pages_model.rowsInserted.connect(self._on_model_rows_inserted, Qt.Qt.QueuedConnection)
        self.pages_view = PagesView(self.pages_model)
        self.pages_view.setModel(self.pages_model)
        self.pages_view.selectionModel().currentRowChanged.connect(self._on_page_focus_changed)
        l.addWidget(self.pages_view)
        self.progress_thread_pool = None
        self._attached_page = None
        self.delete_selected_action = Qt.QAction(self)
        self.delete_selected_action.setText('Delete pages')
        self.delete_selected_action.setToolTip('Delete currently selected main flipbook pages')
        self.delete_selected_action.setShortcut(Qt.Qt.Key_Delete)
        self.delete_selected_action.setShortcutContext(Qt.Qt.WidgetWithChildrenShortcut)
        self.delete_selected_action.triggered.connect(self.delete_selected)
        self.addAction(self.delete_selected_action)
        self.consolidate_selected_action = Qt.QAction(self)
        self.consolidate_selected_action.setText('Consolidate pages')
        self.consolidate_selected_action.setToolTip('Consolidate selected main flipbook pages (combine them into one page)')
        self.consolidate_selected_action.setShortcut(Qt.Qt.Key_Return)
        self.consolidate_selected_action.setShortcutContext(Qt.Qt.WidgetWithChildrenShortcut)
        self.consolidate_selected_action.triggered.connect(self.merge_selected)
        self.addAction(self.consolidate_selected_action)
        self.pages_view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self._on_selection_changed()
        self._on_page_focus_changed()

    def add_image_files(self, image_fpaths):
        """image_fpaths: An iterable of filenames and/or iterables of filenames, with
        a filename being either a pathlib.Path object or a string.  For example, the
        following would append 7 pages to the flipbook, with 1 image in the first
        appended page, 4 in the second, 1 in the third, 4 in the fourth, 4 in the
        fifth, and 1 in the sixth and seventh pages:

        rw.flipbook.add_image_files(
        [
            '/home/me/nofish_control.png',
            [
                '/home/me/monkey_fish0/cheery_chartreuse.png',
                '/home/me/monkey_fish0/serious_celadon.png',
                '/home/me/monkey_fish0/uber_purple.png',
                '/home/me/monkey_fish0/ultra_violet.png'
            ],
            '/home/me/onefish_muchcontrol.png',
            [
                pathlib.Path('/home/me/monkey_fish1/cheery_chartreuse.png'),
                '/home/me/monkey_fish1/serious_celadon.png',
                '/home/me/monkey_fish1/uber_purple.png',
                '/home/me/monkey_fish1/ultra_violet.png'
            ],
            [
                '/home/me/monkey_fish2/cheery_chartreuse.png',
                '/home/me/monkey_fish2/serious_celadon.png',
                '/home/me/monkey_fish2/uber_purple.png',
                '/home/me/monkey_fish2/ultra_violet.png'
            ]
            '/home/me/somefish_somecontrol.png',
            '/home/me/allfish_nocontrol.png'
        ])

        Flipbook.add_image_files(..) is safe to call from any thread."""

        # make sure image_fpaths contains either str/Path entities, or iterables of the same...
        for p in image_fpaths:
            if not isinstance(p, (str, Path)):
                for i in p:
                    assert(isinstance(i, (str, Path)))
        if Qt.QThread.currentThread() is Qt.QApplication.instance().thread():
            self._add_image_files(image_fpaths)
        else:
            Qt.QApplication.instance().postEvent(self, _XThreadAddImageFilesEvent(image_fpaths))

    def add_image_stacks(self, file_paths, get_stack_name, get_image_name, image_name_order):
        """Given a list of file names, parse them out into stacks of related images
        (e.g. multiple color channels for each image position), and add the stacks
        to the flipbook.

        Parameters:
            file_paths: iterable of file-names of image files
            get_stack_name: a function that, given a file-name stem (the name of
                the file only, without parent directories or the extension),
                will return the name of the stack to which the image belongs.
                (e.g. the name of the image position).
            get_image_name: a function that, given a file-name stem, will return
                the name of the image class (e.g. which color channel).
            image_name_order: order of the image-class names in the final stacks
                (hint: brightfield images should go first to be at the bottom.)

        Example:
        file_paths = ['well1-GFP.tif', 'well1-BF.tif', 'well1-mCherry.tif',
            'well2-GFP.tif', 'well2-BF.tif', 'well2-mCherry.tif']
        def well_name(stem):
            return stem.split('-')[0] # return the 'wellN' portion of the name
        def fluor_name(stem):
            return stem.split('-')[1]
        rw.add_image_stacks(file_paths, well_name, fluor_name, ['BF', 'GFP',
            'mCherry'])
        """
        file_paths = map(Path, file_paths)
        stacks = collections.defaultdict(dict)
        for f in file_paths:
            if f.name.startswith('.'):
                # ignore any OS X dot-files
                continue
            stack_name = get_stack_name(f.stem)
            image_name = get_image_name(f.stem)
            image_dict = stacks[stack_name]
            image_dict[image_name] = f
        stack_list = []
        for stack_name, image_dict in sorted(stacks.items()):
            image_list = om.SignalingList((image_dict[name] for name in image_name_order), name=stack_name)
            stack_list.append(image_list)
        self.add_image_files(stack_list)

    def _handle_dropped_files(self, fpaths, dst_row, dst_column, dst_parent):
        freeimage = FREEIMAGE(show_messagebox_on_error=True, error_messagebox_owner=self)
        if freeimage is None:
            return False
        select_idx = len(self.pages)
        self.add_image_files(fpaths)
        if select_idx != len(self.pages):
            self.focused_page_idx = select_idx
        return True

    def event(self, event):
        if event.type() == _X_THREAD_ADD_IMAGE_FILES_EVENT:
            assert isinstance(event, _XThreadAddImageFilesEvent)
            self._add_image_files(event.image_fpaths, event.need_threadpool)
            return True
        return super().event(event)

    def contextMenuEvent(self, event):
        menu = Qt.QMenu(self)
        menu.addAction(self.consolidate_selected_action)
        menu.addAction(self.delete_selected_action)
        menu.exec(event.globalPos())

    def delete_selected(self):
        sm = self.pages_view.selectionModel()
        m = self.pages_model
        if None in (m, sm):
            return
        midxs = sorted(sm.selectedRows(), key=lambda midx: midx.row())
        # "run" as in consecutive indexes specified as range rather than individually
        runs = []
        run_start_idx = None
        run_end_idx = None
        for midx in midxs:
            if midx.isValid():
                idx = midx.row()
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
            m.removeRows(run_start_idx, run_end_idx - run_start_idx + 1)

    def merge_selected(self):
        """The contents of the currently selected pages (by ascending index order in .pages
        and excluding the target page) are appended to the target page.  The target page is
        the selected page with the lowest index.  After their contents are appended to target,
        the non-target selected pages are removed from .pages."""
        sm = self.pages_view.selectionModel()
        m = self.pages_model
        if None in (m, sm):
            return
        midxs = sm.selectedRows()
        if len(midxs) < 2:
            return
        midxs = sorted(midxs, key=lambda _midx: _midx.row())
        target_midx = midxs.pop(0)
        pages = self.pages
        target_page = pages[target_midx.row()]
        extension = []
        runs = []
        run_start_idx = None
        run_end_idx = None
        for midx in midxs:
            if midx.isValid():
                idx = midx.row()
                if not isinstance(self.pages[idx], ImageList):
                    if run_start_idx is not None:
                        runs.append((run_start_idx, run_end_idx))
                        run_end_idx = run_start_idx = None
                    continue
                if run_start_idx is None:
                    run_end_idx = run_start_idx = idx
                elif idx - run_end_idx == 1:
                    run_end_idx = idx
                else:
                    runs.append((run_start_idx, run_end_idx))
                    run_end_idx = run_start_idx = idx
                extension.extend(pages[idx])
        if run_start_idx is not None:
            runs.append((run_start_idx, run_end_idx))
        for run_start_idx, run_end_idx in reversed(runs):
            m.removeRows(run_start_idx, run_end_idx - run_start_idx + 1)
        target_page.extend(extension)
        self._on_page_focus_changed()

    def _on_selection_changed(self, newly_selected_midxs=None, newly_deselected_midxs=None):
        midxs = self.pages_view.selectionModel().selectedRows()
        self.delete_selected_action.setEnabled(len(midxs) >= 1)
        self.consolidate_selected_action.setEnabled(len(midxs) >= 2)

    @property
    def pages(self):
        return self.pages_model.signaling_list

    @pages.setter
    def pages(self, pages):
        if not isinstance(pages, PageList):
            pages = PageList(pages)
        self.pages_model.signaling_list = pages
        self._on_page_focus_changed()

    @property
    def focused_page_idx(self):
        midx = self.pages_view.selectionModel().currentIndex()
        if midx.isValid():
            return midx.row()

    @focused_page_idx.setter
    def focused_page_idx(self, idx):
        if idx is None:
            self.pages_view.selectionModel().clear()
        else:
            if not 0 <= idx < len(self.pages):
                raise IndexError('The value assigned to focused_pages_idx must either be None or a value >= 0 and < page count.')
            sm = self.pages_view.selectionModel()
            midx = self.pages_model.index(idx, 0)
            sm.setCurrentIndex(midx, sm.ClearAndSelect)

    @property
    def focused_page(self):
        focused_page_idx = self.focused_page_idx
        if focused_page_idx is not None:
            return self.pages[focused_page_idx]

    def ensure_page_focused(self):
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
        self.ensure_page_focused()

    def _on_page_focus_changed(self, midx=None, old_midx=None):
        focused_page = self.focused_page
        if not isinstance(focused_page, ImageList):
            return
        layer_stack = self.layer_stack
        if layer_stack.layers is None:
            layer_stack.layers = []
        layers = layer_stack.layers
        lfp = len(focused_page)
        lls = len(layers)
        for idx in range(max(lfp, lls)):
            if idx >= lfp:
                layers[idx].image = None
            elif idx >= lls:
                layers.append(focused_page[idx])
            else:
                layers[idx].image = focused_page[idx]

    @staticmethod
    def _read_stack(freeimage, image_fpaths):
        return [(freeimage.read(str(image_fpath)), str(image_fpath)) for image_fpath in image_fpaths]

    def _add_image_files(self, image_fpaths):
        assert Qt.QThread.currentThread() is Qt.QApplication.instance().thread()
        freeimage = FREEIMAGE(show_messagebox_on_error=True, error_messagebox_owner=self)
        if freeimage:
            if self.progress_thread_pool is None:
                self.progress_thread_pool = ProgressThreadPool()
                self.progress_thread_pool.task_status_changed.connect(self._on_progress_thread_pool_task_status_changed)
                self.progress_thread_pool.all_tasks_retired.connect(self._on_all_progress_thread_pool_tasks_retired)
                self.layout().addWidget(self.progress_thread_pool)
            for p in image_fpaths:
                if isinstance(p, (str, Path)):
                    reader = self.progress_thread_pool.submit(freeimage.read, str(p))
                    # TODO: is the below a race condition? E.g. is there a possibility that
                    # _on_progress_thread_pool_task_status_changed() will get called in a
                    # different thread before reader.name gets set? If so, that's
                    # a problem as _on_progress_thread_pool_task_status_changed()
                    # requires the .name attribute. However, I assume that that
                    # function will only ever get called by the main event loop in
                    # this thread?
                    reader.name = str(p)
                    self.pages.append(reader)
                else:
                    if len(p) == 0:
                        pages.append(p)
                    else:
                        stack_reader = self.progress_thread_pool.submit(self._read_stack, freeimage, p)
                        if hasattr(p, 'name'):
                            stack_reader.name = p.name
                        else:
                            stack_reader.name = ', '.join(Path(image_fpath).stem for image_fpath in p)
                        self.pages.append(stack_reader)
        self.ensure_page_focused()

    def _on_progress_thread_pool_task_status_changed(self, task, old_status):
        try:
            element_inst_count = self.pages_model._instance_counts[task]
        except KeyError:
            # We received queued notification informing us that something already removed from Tasks
            # changed to Completed status before being removed.
            return
        if task.status is TaskStatus.Completed:
            pages = self.pages
            name = task.name
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

class PagesView(Qt.QTableView):
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

class PagesModelDragDropBehavior(om.signaling_list.DragDropModelBehavior):
    def can_drop_rows(self, src_model, src_rows, dst_row, dst_column, dst_parent):
        return isinstance(src_model, PagesModel)

    # def dropMimeData(self, mime_data, drop_action, row, column, parent):
    #     print(super().dropMimeData(mime_data, drop_action, row, column, parent))

class PagesModel(PagesModelDragDropBehavior, om.signaling_list.PropertyTableModel):
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
                    return Qt.QVariant('{} ({})'.format(element.name, element.status.name))
                if role == Qt.Qt.ForegroundRole:
                    return Qt.QVariant(Qt.QApplication.palette().brush(Qt.QPalette.Disabled, Qt.QPalette.WindowText))
        return super().data(midx, role)
