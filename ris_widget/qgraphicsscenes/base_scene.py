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

class BaseScene(Qt.QGraphicsScene):
    """BaseScene provides for creating and maintaining a ContextualInfoItem (or compatible).

    Although the Qt Graphics View Framework supports multiple views into a single scene, we don't
    have a need for this capability, and we do not go out of our way to make it work correctly (which
    would entail signficant additional code complexity).

    In the past, multiple views into a single scene were supported, and BaseScene had more complexity.
    All that remains is the call to create the contextual info item and a pair of member functions
    for setting and clearing it:

    The clear_contextual_info(self, requester) and update_contextual_info(self, text, requester)
    member functions ensure that a single pair of mouse-exited-so-clear-the-text and
    mouse-over-new-thing-so-display-some-other-text events are handled in order, even if the
    associated calls to clear_contextual_info and update_contextual_info occured out of order.
    The value of the "requester" argument is used to determine the correct order:
    a clear_contextual_info call or, equivalently, update_contextual_info(None...) with a
    non-None requester argument is ignored if the current contextual info text was
    set by a different non-None requester.

    For example, consider the following scenario, on J. Random Platform:

    The mouse pointer enters weather_widget from outside of the application window,
    causing the following call:
    s.update_contextual_info('St. Louis, 75F, Sunny', weather_widget)
    The contextual info text changes to "St. Louis, 75F, Sunny", and it is noted that
    weather_widget set this text.

    The mouse pointer leaves weather_widget, crossing into planetarium_widget.  On
    J. Random Platform, entry events always happen before exit events.  Who knows why.
    (J. Random Platform is win32.)  So,
    s.update_contextual_info('Polaris', planetarium_widget) is called, the contextual info
    text changes to 'Polaris', and it is noted that planetarium_widget set this text.

    Belatedly, s.clear_contextual_info(weather_widget) is called.  The current text was
    set by planetarium_widget, not weather_widget.  Why does weather_widget want to clear
    planetarium_widget's text?  Well, it doesn't; it wanted to clear its own text, but
    planetarium_widget intervened and replaced that text before weather_widget got
    around to clearing it.  Therefore, clear requester does not match current text requester,
    causing that clear call to be ignored.

    The mouse cursor leaves planetarium_widget, crossing out of the application window entirely.
    s.clear_contextual_info(planetarium_widget) is called.  planetarium_widget requested
    the current text, and so, the contextual info text is cleared."""

    def __init__(self, parent, ContextualInfoItemClass):
        super().__init__(parent)
        self._requester_of_current_nonempty_mouseover_info = None
        self.contextual_info_item = ContextualInfoItemClass()
        self.addItem(self.contextual_info_item)

    def clear_contextual_info(self, requester):
        self.update_contextual_info(None, requester)

    def update_contextual_info(self, text, requester):
        if text:
            self._requester_of_current_nonempty_mouseover_info = requester
            self.contextual_info_item.text = text
        else:
            if self._requester_of_current_nonempty_mouseover_info is None or self._requester_of_current_nonempty_mouseover_info is requester:
                self._requester_of_current_nonempty_mouseover_info = None
                self.contextual_info_item.text = None
