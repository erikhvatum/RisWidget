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

def with_multi_selection_deletion_shortcut(cls,
                                           desc='Delete selection',
                                           shortcut=Qt.Qt.Key_Delete,
                                           shortcut_context=Qt.Qt.WidgetShortcut,
                                           action_attr_name='delete_selection_action',
                                           make_handler_method=True,
                                           handler_method_name='_on_delete_selection_action_triggered',
                                           connect_action_triggered_to_handler_method=True):
    orig_init = cls.__init__
    def init(self, *va, **kw):
        orig_init(self, *va, **kw)
        action = Qt.QAction(self)
        action.setText(desc)
        action.setShortcut(shortcut)
        action.setShortcutContext(shortcut_context)
        self.addAction(action)
        setattr(self, action_attr_name, action)
        if connect_action_triggered_to_handler_method:
            action.triggered.connect(getattr(self, handler_method_name))
    init.__doc__ = orig_init.__doc__
    cls.__init__ = init
    if make_handler_method:
        def on_action_triggered(self):
            sm = self.selectionModel()
            m = self.model()
            if None in (m, sm):
                return
            midxs = sorted(sm.selectedRows(), key=lambda midx: midx.row())
            # "run" as in RLE as in consecutive indexes specified as range rather than individually
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
        setattr(cls, handler_method_name, on_action_triggered)
    return cls
