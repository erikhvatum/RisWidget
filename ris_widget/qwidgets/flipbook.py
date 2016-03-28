# The MIT License (MIT)
#
# Copyright (c) 2014-2016 WUSTL ZPLAB
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
from PyQt5 import Qt
from .. import om
from ..image import Image
from ..shared_resources import FREEIMAGE
from .default_table import DefaultTable
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
    def __init__(self, image_fpaths, completion_callback):
        super().__init__(_X_THREAD_ADD_IMAGE_FILES_EVENT)
        self.image_fpaths = image_fpaths
        self.completion_callback = completion_callback

_DELAYED_CALLBACKS_EVENT = Qt.QEvent.registerEventType()

class _DelayedCallbacksEvent(Qt.QEvent):
    def __init__(self, callbacks):
        super().__init__(_DELAYED_CALLBACKS_EVENT)
        self.callbacks = callbacks

_FLIPBOOK_PAGES_DOCSTRING = ("""
    The list of pages represented by a Flipbook instance's list view is available via a that
    Flipbook instance's .pages property.

    An individual page is itself a list of Images, but single Image pages need not be inserted
    as single element lists.  Single Image or array-like object insertions into the list of
    pages (.pages) are wrapped in an automatically created single element list if needed.
    Likewise, although a Flipbook instance's .pages property always and exclusively contains
    lists of pages, and pages, in turn, always and exclusively contain Image instances, 2D
    and 3D array-like objects (typically numpy.ndarray instances) inserted are always wrapped
    or copied into a new Image (an ndarray with appropriate striding and dtype is wrapped rather
    than copied).

    Ex:

    import numpy
    from ris_widget.ris_widget import RisWidget
    rw = RisWidget()
    rw.show()

    print(rw.flipbook.pages)
    # <ris_widget.qwidgets.flipbook.PageList object at 0x7fa93e38f678>

    rw.flipbook.pages.append(numpy.zeros((600,800), dtype=numpy.uint8).T)
    print(rw.flipbook.pages)
    # <ris_widget.qwidgets.flipbook.PageList object at 0x7fa93e38f678
    # [
    #         <ris_widget.qwidgets.flipbook.ImageList object at 0x7fa93e399b88
    #     [
    #         <ris_widget.image.Image object at 0x7fa93e399d38; unnamed, 800x600, 1 channel (G)>
    #     ]>
    # ]>

    print(rw.flipbook.pages)
    # <ris_widget.qwidgets.flipbook.PageList object at 0x7fa93e38f678
    # [
    #         <ris_widget.qwidgets.flipbook.ImageList object at 0x7fa93e399b88
    #     [
    #         <ris_widget.image.Image object at 0x7fa93e399d38; unnamed, 800x600, 1 channel (G)>
    #     ]>,
    #         <ris_widget.qwidgets.flipbook.ImageList object at 0x7fa945dd6048
    #     [
    #         <ris_widget.image.Image object at 0x7fa945dd60d8; unnamed, 640x480, 1 channel (G)>,
    #         <ris_widget.image.Image object at 0x7fa945dd6168; unnamed, 320x200, 1 channel (G)>
    #     ]>
    # ]>

    """)

class Flipbook(Qt.QWidget):
    """
    Flipbook: A Qt widget with a list view containing pages.  Calling a Flipbook instance's
    .add_image_files method is the easiest way in which to load a number of image files as pages
    into a Flipbook - see help(ris_widget.qwidgets.flipbook.Flipbook) for more information
    regarding this method.

    """

    __doc__ += _FLIPBOOK_PAGES_DOCSTRING

    page_focus_changed = Qt.pyqtSignal(object)
    page_selection_changed = Qt.pyqtSignal(object)

    def __init__(self, layer_stack, parent=None):
        super().__init__(parent)
        self.layer_stack = layer_stack
        self.setLayout(Qt.QVBoxLayout())
        self.views_splitter = Qt.QSplitter(Qt.Qt.Vertical)
        self.layout().addWidget(self.views_splitter)
        self.pages_groupbox = Qt.QGroupBox('Pages')
        self.pages_groupbox.setLayout(Qt.QHBoxLayout())
        self.pages_view = PagesView()
        self.pages_groupbox.layout().addWidget(self.pages_view)
        self.pages_model = PagesModel(PageList(), self.pages_view)
        self.pages_model.handle_dropped_files = self._handle_dropped_files
        self.pages_model.rowsInserted.connect(self._on_model_rows_inserted, Qt.Qt.QueuedConnection)
        self.pages_view.setModel(self.pages_model)
        self.pages_view.selectionModel().currentRowChanged.connect(self.apply)
        self.pages_view.selectionModel().selectionChanged.connect(self._on_page_selection_changed)
        self.views_splitter.addWidget(self.pages_groupbox)
        self.page_content_groupbox = Qt.QGroupBox('Page Contents')
        self.page_content_groupbox.setLayout(Qt.QHBoxLayout())
        self.page_content_view = DefaultTable()
        h = self.page_content_view.horizontalHeader()
        h.setStretchLastSection(True)
        h.setHighlightSections(False)
        h.setSectionsClickable(False)
        h = self.page_content_view.verticalHeader()
        h.setHighlightSections(False)
        h.setSectionsClickable(False)
        self.page_content_groupbox.layout().addWidget(self.page_content_view)
        self.page_content_model = PageContentModel(parent=self.page_content_view)
        self.page_content_model.rowsInserted.connect(self._on_content_model_rows_inserted, Qt.Qt.QueuedConnection)
        self.page_content_model.modelReset.connect(self._on_content_model_rows_inserted, Qt.Qt.QueuedConnection)
        self.page_content_view.setModel(self.page_content_model)
        self.views_splitter.addWidget(self.page_content_groupbox)
        self.views_splitter.setStretchFactor(0, 4)
        self.views_splitter.setStretchFactor(0, 1)
        self.views_splitter.setSizes((1, 0))
        self.progress_thread_pool = None
        self.progress_thread_pool_completion_callbacks = []
        self._attached_page = None
        self.delete_selected_action = Qt.QAction(self)
        self.delete_selected_action.setText('Delete pages')
        self.delete_selected_action.setToolTip('Delete currently selected main flipbook pages')
        self.delete_selected_action.setShortcut(Qt.Qt.Key_Delete)
        self.delete_selected_action.setShortcutContext(Qt.Qt.WidgetShortcut)
        self.delete_selected_action.triggered.connect(self.delete_selected)
        self.pages_view.addAction(self.delete_selected_action)
        self.consolidate_selected_action = Qt.QAction(self)
        self.consolidate_selected_action.setText('Consolidate pages')
        self.consolidate_selected_action.setToolTip('Consolidate selected main flipbook pages (combine them into one page)')
        self.consolidate_selected_action.setShortcut(Qt.Qt.Key_Return)
        self.consolidate_selected_action.setShortcutContext(Qt.Qt.WidgetWithChildrenShortcut)
        self.consolidate_selected_action.triggered.connect(self.merge_selected)
        self.addAction(self.consolidate_selected_action)
        self.pages_view.selectionModel().selectionChanged.connect(self._on_page_selection_changed)
        self._on_page_selection_changed()
        self.apply()

    def apply(self):
        """Replace the image fields of the layers in .layer_stack with the images contained in the currently
        focused flipbook page, creating new layers as required, or clearing the image field of any excess
        layers.  This method is called automatically when focus moves to a different page and when
        the contents of the current page change."""
        focused_page = self.focused_page
        if not isinstance(focused_page, ImageList):
            self.page_content_groupbox.setEnabled(False)
            self.page_content_model.signaling_list = None
            self._detach_page()
            return
        if focused_page is not self._attached_page:
            self._detach_page()
            focused_page.inserted.connect(self.apply)
            focused_page.removed.connect(self.apply)
            focused_page.replaced.connect(self.apply)
            self.page_content_groupbox.setEnabled(True)
            self.page_content_model.signaling_list = focused_page
            self._attached_page = focused_page
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
        self.page_focus_changed.emit(self)

    def _detach_page(self):
        if self._attached_page is not None:
            self._attached_page.inserted.disconnect(self.apply)
            self._attached_page.removed.disconnect(self.apply)
            self._attached_page.replaced.disconnect(self.apply)
            self._attached_page = None

    def add_image_files(self, image_fpaths, completion_callback=None):
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

        Flipbook.add_image_files(..) is safe to call from any thread.

        A callable supplied for completion_callback is executed when bulk image loading
        completes.  If you call .add_image_files(..) before the bulk image loading
        initiated by earlier .add_image_files(..) calls completes, callback execution
        is postponed until the additional images have been loaded.  When loading of
        all images has ended, regardless of whether all images loaded successfully or
        not, all callbacks installed by .add_image_files calls during that period of
        continuous bulk image loading are executed."""
        
        image_fpaths_l = []
        for p in image_fpaths:
            if isinstance(p, (str, Path)):
                image_fpaths_l.append(p)
            else:
                i_s = []
                for i in p:
                    assert(isinstance(i, (str, Path)))
                    i_s.append(i)
                image_fpaths_l.append(i_s)
        if image_fpaths_l:
            if Qt.QThread.currentThread() is Qt.QApplication.instance().thread():
                self._add_image_files(image_fpaths_l, completion_callback)
            else:
                Qt.QApplication.instance().postEvent(self, _XThreadAddImageFilesEvent(image_fpaths_l, completion_callback))

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
            self._add_image_files(event.image_fpaths, event.completion_callback)
            return True
        elif event.type() == _DELAYED_CALLBACKS_EVENT:
            for callback in event.callbacks:
                callback()
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
        the non-target selected pages are removed from .pages.  Any page still loading (e.g.
        added by .add_image_files() and not yet complete) is ignored.  If the target page
        is still loading, .merge_selected() is a no-op."""
        sm = self.pages_view.selectionModel()
        m = self.pages_model
        if None in (m, sm):
            return
        midxs = sm.selectedRows()
        midxs = sorted(
            (midx for midx in midxs if midx.isValid() and not isinstance(midx.data(), Task)),
            key=lambda _midx: _midx.row())
        if len(midxs) < 2:
            return
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
        self.apply()

    def _on_page_selection_changed(self, newly_selected_midxs=None, newly_deselected_midxs=None):
        midxs = self.pages_view.selectionModel().selectedRows()
        self.delete_selected_action.setEnabled(len(midxs) >= 1)
        self.consolidate_selected_action.setEnabled(len(midxs) >= 2)
        self.page_selection_changed.emit(self)

    def _on_pages_replaced(self, idxs, replaced_pages, pages):
        if self.focused_page_idx in idxs:
            self.apply()

    def focus_prev_page(self):
        """Advance to the previous page, if there is one."""
        idx = self.focused_page_idx
        if idx is None:
            selected_idxs = self.selected_page_idxs
            if not selected_idxs:
                self.ensure_page_focused()
                return
            idx = selected_idxs[0]
        self.focused_page_idx = max(idx - 1, 0)

    def focus_next_page(self):
        """Advance to the next page, if there is one."""
        idx = self.focused_page_idx
        if idx is None:
            selected_idxs = self.selected_page_idxs
            if not selected_idxs:
                self.ensure_page_focused()
                return
            idx = selected_idxs[0]
        self.focused_page_idx = min(idx + 1, len(self.pages) - 1)

    @property
    def pages(self):
        return self.pages_model.signaling_list

    @pages.setter
    def pages(self, pages):
        if not isinstance(pages, PageList):
            pages = PageList(pages)
        sl = self.pages_model.signaling_list
        if sl is not None:
            try:
                sl.replaced.disconnect(self._on_pages_replaced)
            except TypeError:
                pass
        self.pages_model.signaling_list = pages
        if pages is not None:
            pages.replaced.connect(self._on_pages_replaced)
        self.ensure_page_focused()
        self.page_selection_changed.emit(self)
        self.apply()

    try:
        pages.__doc__ = _FLIPBOOK_PAGES_DOCSTRING
    except AttributeError:
        pass

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

    @property
    def selected_page_idxs(self):
        return sorted(midx.row() for midx in self.pages_view.selectionModel().selectedRows() if midx.isValid())

    @selected_page_idxs.setter
    def selected_page_idxs(self, idxs):
        idxs = sorted(idxs)
        if not idxs:
            self.pages_view.selectionModel().clearSelection()
            return
        m = self.pages_model
        sm = self.pages_view.selectionModel()
        page_count = len(self.pages)
        idxs = [idx for idx in idxs if 0 <= idx < page_count]
        # "run" as in consecutive indexes specified as range rather than individually
        runs = []
        run_start_idx = None
        run_end_idx = None
        for idx in idxs:
            if run_start_idx is None:
                run_end_idx = run_start_idx = idx
            elif idx - run_end_idx == 1:
                run_end_idx = idx
            else:
                runs.append((run_start_idx, run_end_idx))
                run_end_idx = run_start_idx = idx
        if run_start_idx is not None:
            runs.append((run_start_idx, run_end_idx))
        focused_idx = self.focused_page_idx
        item_selection = Qt.QItemSelection()
        for run_start_idx, run_end_idx in runs:
            item_selection.append(Qt.QItemSelectionRange(m.index(run_start_idx, 0), m.index(run_end_idx, 0)))
        sm.select(item_selection, Qt.QItemSelectionModel.ClearAndSelect)
        if focused_idx not in idxs:
            sm.setCurrentIndex(m.index(idxs[0], 0), Qt.QItemSelectionModel.Current)

    @property
    def selected_pages(self):
        pages = self.pages
        return [pages[idx] for idx in self.selected_page_idxs]

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

    def _on_model_rows_inserted(self):
        self.pages_view.resizeRowsToContents()
        self.ensure_page_focused()

    def _on_content_model_rows_inserted(self):
        self.page_content_view.resizeRowsToContents()

    def _read_stack(self, freeimage, image_fpaths):
        return [(freeimage.read(str(image_fpath)), str(image_fpath)) for image_fpath in image_fpaths]

    def _add_image_files(self, image_fpaths, completion_callback):
        assert Qt.QThread.currentThread() is Qt.QApplication.instance().thread()
        pages = []
        freeimage = FREEIMAGE(show_messagebox_on_error=True, error_messagebox_owner=self)
        if freeimage:
            if self.progress_thread_pool is None:
                self.progress_thread_pool = ProgressThreadPool()
                self.progress_thread_pool.task_status_changed.connect(self._on_progress_thread_pool_task_status_changed)
                self.progress_thread_pool.all_tasks_retired.connect(self._on_all_progress_thread_pool_tasks_retired)
                self.layout().addWidget(self.progress_thread_pool)
            if completion_callback is not None:
                self.progress_thread_pool_completion_callbacks.append(completion_callback)
            for p in image_fpaths:
                if isinstance(p, (str, Path)):
                    reader = self.progress_thread_pool.submit(freeimage.read, str(p))
                    reader.name = str(p)
                    pages.append(reader)
                else:
                    if len(p) == 0:
                        pages.append(p)
                    else:
                        stack_reader = self.progress_thread_pool.submit(self._read_stack, freeimage, p)
                        stack_reader.name = ', '.join(Path(image_fpath).stem for image_fpath in p)
                        pages.append(stack_reader)
            self.pages.extend(pages)
            self.ensure_page_focused()
        else:
            if completion_callback is not None:
                completion_callback()

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
                    self.apply()
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
        Qt.QApplication.instance().postEvent(self, _DelayedCallbacksEvent(self.progress_thread_pool_completion_callbacks))
        self.progress_thread_pool_completion_callbacks = []

class PagesView(Qt.QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
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

class PageContentModelDragDropBehavior(om.signaling_list.DragDropModelBehavior):
    def can_drop_rows(self, src_model, src_rows, dst_row, dst_column, dst_parent):
        return isinstance(src_model, PageContentModel)

    def handle_dropped_qimage(self, qimage, name, dst_row, dst_column, dst_parent):
        image = Image.from_qimage(qimage=qimage, name=name)
        if image is not None:
            self.signaling_list[dst_row:dst_row] = [image]
            return True
        return False

    def handle_dropped_files(self, fpaths, dst_row, dst_column, dst_parent):
        freeimage = FREEIMAGE(show_messagebox_on_error=True, error_messagebox_owner=None)
        if freeimage is None:
            return False
        images = ImageList()
        for fpath in fpaths:
            fpath_str = str(fpath)
            images.append(Image(freeimage.read(fpath_str), name=fpath_str))
        self.signaling_list[dst_row:dst_row] = images
        return True

class PageContentModel(PageContentModelDragDropBehavior, om.signaling_list.PropertyTableModel):
    PROPERTIES = (
        'name',
        )

    def __init__(self, signaling_list=None, parent=None):
        super().__init__(self.PROPERTIES, signaling_list, parent)