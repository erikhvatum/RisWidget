﻿# The MIT License (MIT)
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
from . import progress_thread_pool

class ImageList(om.UniformSignalingList):
    def take_input_element(self, obj):
        return obj if isinstance(obj, Image) else Image(obj, immediate_texture_upload=False)

class PageList(om.UniformSignalingList):
    def take_input_element(self, obj):
        if isinstance(obj, ImageList):
            return obj
        if isinstance(obj, (numpy.ndarray, Image)):
            ret = ImageList((obj,))
            if hasattr(obj, 'name'):
                ret.name = obj.name
            return ret
        return ImageList(obj)

class _ReadPageTaskDoneEvent(Qt.QEvent):
    TYPE = Qt.QEvent.registerEventType()
    def __init__(self, task_page, error=False):
        super().__init__(self.TYPE)
        self.task_page = task_page
        self.error = error

class _ReadPageTaskPage:
    __slots__ = ["page", "im_fpaths", "im_names", "ims"]

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
    _play_advance_frame = Qt.pyqtSignal()

    def __init__(self, layer_stack, parent=None):
        super().__init__(parent)
        self.layer_stack = layer_stack
        self.setLayout(Qt.QVBoxLayout())
        self.views_splitter = Qt.QSplitter(Qt.Qt.Vertical)
        self.layout().addWidget(self.views_splitter)
        self.pages_groupbox = Qt.QGroupBox('Pages')
        l = Qt.QVBoxLayout()
        self.pages_groupbox.setLayout(l)
        self.pages_view = PagesView()
        l.addWidget(self.pages_view)
        ll = Qt.QHBoxLayout()
        self.toggle_playing_action = Qt.QAction(self)
        self.toggle_playing_action.setText('Play')
        self.toggle_playing_action.setShortcut(Qt.Qt.Key_P)
        self.toggle_playing_action.setCheckable(True)
        self.toggle_playing_action.setChecked(False)
        self.toggle_playing_action.setEnabled(False)
        self.toggle_playing_action.toggled.connect(self._on_toggle_play_action_toggled)
        self.toggle_playing_button = Qt.QPushButton('\N{BLACK RIGHT-POINTING POINTER}')
        self.toggle_playing_button.setCheckable(True)
        self.toggle_playing_button.setEnabled(False)
        self.toggle_playing_button.clicked.connect(self._on_toggle_play_button_toggled)
        self._play_advance_frame.connect(self._on_play_advance_frame, Qt.Qt.QueuedConnection)
        ll.addSpacerItem(Qt.QSpacerItem(0, 0, Qt.QSizePolicy.Expanding, Qt.QSizePolicy.Minimum))
        ll.addWidget(self.toggle_playing_button)
        ll.addSpacerItem(Qt.QSpacerItem(0, 0, Qt.QSizePolicy.Expanding, Qt.QSizePolicy.Minimum))
        l.addLayout(ll)
        self.pages_model = PagesModel(PageList(), self.pages_view)
        self.pages_model.handle_dropped_files = self._handle_dropped_files
        self.pages_model.rowsInserted.connect(self._on_model_rows_inserted)
        self.pages_model.rowsRemoved.connect(self._on_model_reset_or_rows_removed)
        self.pages_model.modelReset.connect(self._on_model_reset_or_rows_removed)
        self.pages_model.rowsInserted.connect(self._on_model_reset_or_rows_inserted_indirect, Qt.Qt.QueuedConnection)
        self.pages_model.modelReset.connect(self._on_model_reset_or_rows_inserted_indirect, Qt.Qt.QueuedConnection)
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
        self.page_content_model = PageContentModel(layer_stack, self.page_content_view)
        self.page_content_model.rowsInserted.connect(self._on_content_model_rows_inserted, Qt.Qt.QueuedConnection)
        self.page_content_model.modelReset.connect(self._on_content_model_rows_inserted, Qt.Qt.QueuedConnection)
        self.page_content_view.setModel(self.page_content_model)
        self.views_splitter.addWidget(self.page_content_groupbox)
        self.views_splitter.setStretchFactor(0, 4)
        self.views_splitter.setStretchFactor(0, 1)
        self.views_splitter.setSizes((1, 0))
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
        self._on_page_selection_changed()
        self.freeimage = FREEIMAGE(show_messagebox_on_error=True, error_messagebox_owner=self)
        self.apply()

    def apply(self):
        """Replace the image fields of the layers in .layer_stack with the images contained in the currently
        focused flipbook page, creating new layers as required, or clearing the image field of any excess
        layers.  This method is called automatically when focus moves to a different page and when
        the contents of the current page change."""
        focused_page_idx = self.focused_page_idx
        if focused_page_idx is None:
            self.page_content_groupbox.setEnabled(False)
            self.page_content_model.signaling_list = None
            self._detach_page()
            return
        pages = self.pages
        focused_page = pages[focused_page_idx]
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
        hint_pages = []
        if focused_page_idx > 0:
            hint_pages.append(pages[focused_page_idx-1])
        if focused_page_idx < len(pages) - 1:
            hint_pages.append(pages[focused_page_idx+1])
        for hint_page in hint_pages:
            for image in hint_page:
                image.async_texture.upload()
        self.page_focus_changed.emit(self)

    def _detach_page(self):
        if self._attached_page is not None:
            self._attached_page.inserted.disconnect(self.apply)
            self._attached_page.removed.disconnect(self.apply)
            self._attached_page.replaced.disconnect(self.apply)
            self._attached_page = None

    def add_image_files(self, image_fpaths, page_names=None, image_names=None, insertion_point=-1):
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

        page_names: iterable of same length as image_fpaths, containing
            names for each entry to display in the flipbook. Optional.

        image_names: iterable of same structure as image_fpaths, containing
            the desired image name for each loaded image. Optional.

        Returns list of futures objects corresponding to the page-IO tasks.
        To wait until read is done, call concurrent.futures.wait() on this list.
        """
        if not self.freeimage:
            return
        task_pages = []
        for i, p in enumerate(image_fpaths):
            task_page = _ReadPageTaskPage()
            task_page.page = page = ImageList()
            if page_names is not None:
                page.name = page_names[i]
            if isinstance(p, (str, Path)):
                if page_names is None:
                    page.name = str(p)
                if image_names is None:
                    task_page.im_names = [str(p)]
                else:
                    task_page.im_names = [image_names[i]]
                task_page.im_fpaths = [Path(p)]
            else:
                task_page.im_fpaths = []
                for j, im_fpaths in enumerate(p):
                    assert(isinstance(im_fpaths, (str, Path)))
                    task_page.im_fpaths.append(Path(im_fpaths))
                if page_names is None:
                    page.name = ', '.join(image_fpath.stem for image_fpath in task_page.im_fpaths)
                if image_names is None:
                    task_page.im_names = [str(image_fpath) for image_fpath in task_page.im_fpaths]
                else:
                    task_page.im_names = image_names[i]
            assert len(task_page.im_names) == len(task_page.im_fpaths)
            task_pages.append(task_page)
        return self.queue_page_creation_tasks(insertion_point, task_pages)

    def _handle_dropped_files(self, fpaths, dst_row, dst_column, dst_parent):
        if self.freeimage is None:
            return False
        if dst_row in (-1, None):
            dst_row = len(self.pages)
        self.add_image_files(fpaths, insertion_point=dst_row)
        if dst_row < len(self.pages):
            self.focused_page_idx = dst_row
        return True

    def event(self, e):
        if e.type() == _ReadPageTaskDoneEvent.TYPE:
            if e.error:
                e.task_page.page.name += ' (ERROR)'
            else:
                e.task_page.page.extend(Image(im, name=im_name, mask=self.layer_stack.imposed_image_mask, immediate_texture_upload=False) for
                                        (im, im_name) in zip(e.task_page.ims, e.task_page.im_names))
            return True
        return super().event(e)

    def _read_page_task(self, task_page):
        task_page.ims = [self.freeimage.read(str(image_fpath)) for image_fpath in task_page.im_fpaths]
        Qt.QApplication.instance().postEvent(self, _ReadPageTaskDoneEvent(task_page))

    def _on_task_error(self, task_page):
        Qt.QApplication.instance().postEvent(self, _ReadPageTaskDoneEvent(task_page, error=True))

    def queue_page_creation_tasks(self, insertion_point, task_pages):
        if not hasattr(self, 'thread_pool'):
            self.thread_pool = progress_thread_pool.ProgressThreadPool(self.cancel_page_creation_tasks, self.layout)
        new_pages = []
        page_futures = []
        for task_page in task_pages:
            future = self.thread_pool.submit(self._read_page_task, task_page, on_error=self._on_task_error, on_error_args=task_page)
            task_page.page.on_removal = future.cancel
            new_pages.append(task_page.page)
            page_futures.append(future)
        self.pages[insertion_point:insertion_point] = new_pages
        self.ensure_page_focused()
        return page_futures

    def cancel_page_creation_tasks(self):
        for i, image_list in reversed(list(enumerate(self.pages))):
            if len(image_list) == 0:
                # page removal calls the on_removal function, which as above is the future's cancel()
                self.pages_model.removeRows(i, 1)

    def contextMenuEvent(self, event):
        menu = Qt.QMenu(self)
        menu.addAction(self.consolidate_selected_action)
        menu.addAction(self.delete_selected_action)
        menu.exec(event.globalPos())

    def delete_selected(self):
        sm = self.pages_view.selectionModel()
        m = self.pages_model
        if sm is None or m is None:
            return
        selected_rows = self.selected_page_idxs[::-1]
        # "run" as in consecutive indexes specified as range rather than individually
        run_start_idx = selected_rows[0]
        run_length = 1
        for idx in selected_rows[1:]:
            if idx == run_start_idx - 1:
                # if the previous selected row is adjacent to the current "start"
                # of the run, extend the run one back
                run_start_idx = idx
                run_length += 1
            else:
                # delete one run and start recording the next
                m.removeRows(run_start_idx, run_length)
                run_start_idx = idx
                run_length = 1
        m.removeRows(run_start_idx, run_length)

    def merge_selected(self):
        """The contents of the currently selected pages (by ascending index order in .pages
        and excluding the target page) are appended to the target page. The target page is
        the selected page with the lowest index."""
        mergeable_rows = list(self.selected_page_idxs)
        if len(mergeable_rows) < 2:
            return
        target_row = mergeable_rows.pop(0)
        target_page = self.pages[target_row]
        to_add = [self.pages[row] for row in mergeable_rows]
        midx = self.pages_model.createIndex(target_row, 0)
        self.pages_view.selectionModel().select(midx, Qt.QItemSelectionModel.Deselect)
        self.delete_selected()
        for image_list in to_add:
            target_page.extend(image_list)
        self.pages_view.selectionModel().select(midx, Qt.QItemSelectionModel.Select)
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
        sm = self.pages_view.selectionModel()
        if idx is None:
            sm.clear()
        else:
            if not 0 <= idx < len(self.pages):
                raise IndexError('The value assigned to focused_pages_idx must either be None or a value >= 0 and < page count.')
            midx = self.pages_model.index(idx, 0)
            sm.setCurrentIndex(midx, Qt.QItemSelectionModel.ClearAndSelect)

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
        page_count = len(self.pages)
        idxs = [idx for idx in sorted(idxs) if 0 <= idx < page_count]
        # "run" as in consecutive indexes specified as range rather than individually
        run_start_idx = idxs[0]
        run_end_idx = idxs[0]
        runs = []
        for idx in idxs[1:]:
            if idx == run_end_idx + 1:
                run_end_idx = idx
            else:
                runs.append((run_start_idx, run_end_idx))
                run_end_idx = run_start_idx = idx
        runs.append((run_start_idx, run_end_idx))
        m = self.pages_model
        item_selection = Qt.QItemSelection()
        for run_start_idx, run_end_idx in runs:
            item_selection.append(Qt.QItemSelectionRange(m.index(run_start_idx, 0), m.index(run_end_idx, 0)))
        sm = self.pages_view.selectionModel()
        sm.select(item_selection, Qt.QItemSelectionModel.ClearAndSelect)
        if self.focused_page_idx not in idxs:
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
        e = len(self.pages) >= 2
        self.toggle_playing_action.setEnabled(e)
        self.toggle_playing_button.setEnabled(e)

    def _on_model_reset_or_rows_removed(self):
        if len(self.pages) < 2:
            a = self.toggle_playing_action
            a.setChecked(False)
            a.setEnabled(False)
            self.toggle_playing_button.setEnabled(False)
        else:
            self.toggle_playing_action.setEnabled(True)
            self.toggle_playing_button.setEnabled(True)

    def _on_model_reset_or_rows_inserted_indirect(self):
        self.pages_view.resizeRowsToContents()
        self.ensure_page_focused()

    def _on_content_model_rows_inserted(self):
        self.page_content_view.resizeRowsToContents()

    @property
    def is_playing(self):
        return self.toggle_playing_action.isEnabled() and self.toggle_playing_action.isChecked()

    @is_playing.setter
    def is_playing(self, v):
        if self.toggle_playing_action.isEnabled():
            self.toggle_playing_action.setChecked(v)

    def play(self):
        if self.is_playing or len(self.pages) < 2:
            return
        self.is_playing = True
        self.toggle_playing_button.setChecked(True)
        self._on_play_advance_frame()

    def pause(self):
        self.is_playing = False
        self.toggle_playing_button.setChecked(False)

    def _on_toggle_play_button_toggled(self, v):
        self.toggle_playing_action.setChecked(v)

    def _on_toggle_play_action_toggled(self, v):
        self.toggle_playing_button.setChecked(v)
        self._on_play_advance_frame()

    def _on_play_advance_frame(self):
        if not self.is_playing:
            return
        page_count = len(self.pages)
        if page_count == 0:
            return
        focused_page_idx = self.focused_page_idx
        if focused_page_idx + 1 == page_count:
            self.focused_page_idx = 0
        else:
            self.focused_page_idx = focused_page_idx + 1
        self._play_advance_frame.emit()

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

class ImageListListener(Qt.QObject):
    def __init__(self, image_list, pages_model, parent=None):
        super().__init__(parent)
        self.image_list = image_list
        self.pages_model = pages_model
        self.image_list.inserted.connect(self._on_change)
        self.image_list.replaced.connect(self._on_change)
        self.image_list.removed.connect(self._on_change)

    def remove(self):
        self.image_list.inserted.disconnect(self._on_change)
        self.image_list.replaced.disconnect(self._on_change)
        self.image_list.removed.disconnect(self._on_change)

    def _on_change(self, *args, **kws):
        idx = self.pages_model.signaling_list.index(self.image_list)
        index = self.pages_model.createIndex(idx, 0)
        self.pages_model.dataChanged.emit(index, index)

class PagesModel(PagesModelDragDropBehavior, om.signaling_list.PropertyTableModel):
    PROPERTIES = (
        'name',
        )

    def __init__(self, signaling_list=None, parent=None):
        self.listeners = {}
        super().__init__(self.PROPERTIES, signaling_list, parent)
        self.modelAboutToBeReset.connect(self._on_model_about_to_be_reset)
        self.modelReset.connect(self._on_model_reset)

    def flags(self, midx):
        if midx.isValid() and midx.column() == self.PROPERTIES.index('name'):
            image_list = self.signaling_list[midx.row()]
            if len(image_list) == 0:
                return super().flags(midx) & ~Qt.Qt.ItemIsEditable
        return super().flags(midx)

    def data(self, midx, role=Qt.Qt.DisplayRole):
        if midx.isValid() and midx.column() == self.PROPERTIES.index('name'):
            image_list = self.signaling_list[midx.row()]
            if image_list is None:
                return Qt.QVariant()
            if len(image_list) == 0:
                if role == Qt.Qt.ForegroundRole:
                    return Qt.QVariant(Qt.QApplication.palette().brush(Qt.QPalette.Disabled, Qt.QPalette.WindowText))
        return super().data(midx, role)

    def removeRows(self, row, count, parent=Qt.QModelIndex()):
        try:
            to_remove = self.signaling_list[row:row+count]
        except IndexError:
            return False
        for row_entry in to_remove:
            # call on-removal callback if present
            on_removal = getattr(row_entry, 'on_removal', None)
            if on_removal:
                on_removal()
        return super().removeRows(row, count, parent)

    def _add_listeners(self, image_lists):
        for image_list in image_lists:
            self.listeners[image_list] = ImageListListener(image_list, self)

    def _remove_listeners(self, image_lists):
        for image_list in image_lists:
            listener = self.listeners.pop(image_list)
            listener.remove()

    def _on_inserted(self, idx, elements):
        super()._on_inserted(idx, elements)
        self._add_listeners(elements)

    def _on_replaced(self, idxs, replaced_elements, elements):
        super()._on_replaced(idxs, replaced_elements, elements)
        self._add_listeners(elements)
        self._remove_listeners(replaced_elements)

    def _on_removed(self, idxs, elements):
        super()._on_removed(idxs, elements)
        self._remove_listeners(elements)

    def _on_model_about_to_be_reset(self):
        self._remove_listeners(self.signaling_list)

    def _on_model_reset(self):
        self._add_listeners(self.signaling_list)

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
            images.append(Image(freeimage.read(fpath_str), name=fpath_str, mask=self.layer_stack.imposed_image_mask, immediate_texture_upload=False))
        self.signaling_list[dst_row:dst_row] = images
        return True

class PageContentModel(PageContentModelDragDropBehavior, om.signaling_list.PropertyTableModel):
    PROPERTIES = (
        'name',
        )

    def __init__(self, layer_stack, parent=None):
        super().__init__(self.PROPERTIES, layer_stack.layers, parent)
        self.layer_stack = layer_stack