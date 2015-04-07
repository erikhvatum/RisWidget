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

from .shared_resources import GL, UNIQUE_QGRAPHICSITEM_TYPE
from .shader_view import ShaderView
from pathlib import Path
from PyQt5 import Qt
from string import Template

class ShaderScene(Qt.QGraphicsScene):
    """Although the Qt Graphics View Framework supports multiple views into a single scene, we don't
    have much need for this capability, and we do not go out of our way to make it work correctly.
    GraphicsItems that maintain view relative positions by responding to the
    ShaderView.scene_view_rect_changed signal will be positioned correctly only for the view that
    last emitted the signal.  So, if you make two ImageViews into the same ImageScene and pan one
    ImageView, you will see the contextual info text item remain fixed in the view being panned
    while appearing to move about in the stationary view."""

    # update_contextual_info_signal serves to relay contextual info plaintext/html change requests
    # from any items in the ShaderScene to any interested recipient.  By default, the only recipient
    # is self.contextual_info_text_item, which is pinned to the top left corner of
    # will typically update its mouseover info item in response.  The clear_contextual_info(self, requester)
    # and update_contextual_info(self, string, is_html, requester) member functions provide an interface that
    # relieves the need to ensure that a single pair of mouse-exited-so-clear-the-text and
    # mouse-over-new-thing-so-display-some-other-text events are handled in order, which should be
    # done if update_contextual_info_signal.emit(..) is called directly.
    #
    # If the second parameter is True, the first parameter is interpreted as html.  The first parameter is
    # treated as a plaintext string otherwise.
    update_contextual_info_signal = Qt.pyqtSignal(str, bool)
    # shader_scene_view_rect_changed is emitted by the ShaderView associated with this scene
    # immediately after the boundries of the scene region framed by the view rect change.  Item
    # view coordinates may be held constant by updating item scene position in response to this
    # signal (in the case of shader_scene.ContextualInfoTextItem, for example).
    shader_scene_view_rect_changed = Qt.pyqtSignal(ShaderView)

    def __init__(self, parent):
        super().__init__(parent)
        self.requester_of_current_nonempty_mouseover_info = None
        self.add_contextual_info_item()

    def add_contextual_info_item(self):
        self.contextual_info_text_item = ContextualInfoTextItem()
        self.addItem(self.contextual_info_text_item)
        self.update_contextual_info_signal.connect(self.contextual_info_text_item.on_update_contextual_info)
        self.shader_scene_view_rect_changed.connect(self.contextual_info_text_item.on_shader_view_scene_rect_changed)

    def clear_contextual_info(self, requester):
        self.update_contextual_info(None, False, requester)

    def update_contextual_info(self, string, is_html, requester):
        if string is None or len(string) == 0:
            if self.requester_of_current_nonempty_mouseover_info is None or self.requester_of_current_nonempty_mouseover_info is requester:
                self.requester_of_current_nonempty_mouseover_info = None
                self.update_contextual_info_signal.emit('', False)
        else:
            self.requester_of_current_nonempty_mouseover_info = requester
            self.update_contextual_info_signal.emit(string, is_html)

    @property
    def contextual_info_default_color(self):
        """(r,g,b,a) tuple, with elements in the range [0,255].  The alpha channel value (4th element of the 
        tuple) defaults to 255 and may be omitted when setting this property.  "default" in
        "contextual_info_default_color" refers to the fact that this color may be overridden by color information
        in an HTML ShaderScene.update_contextual_info_signal."""
        c = self.contextual_info_text_item.defaultTextColor()
        return c.red(), c.green(), c.blue(), c.alpha()

    @contextual_info_default_color.setter
    def contextual_info_default_color(self, rgb_a):
        rgb_a = tuple(map(int, rgb_a))
        if len(rgb_a) == 3:
            rgb_a = rgb_a + (255,)
        elif len(rgb_a) != 4:
            raise ValueError('Value supplied for contextual_info_default_color must be a 3 or 4 element iterable.')
        self.contextual_info_text_item.setDefaultTextColor(Qt.QColor(*rgb_a))

class ContextualInfoTextItem(Qt.QGraphicsTextItem):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()

    def __init__(self, parent_item=None):
        super().__init__(parent_item)
        self.setFlag(Qt.QGraphicsItem.ItemIgnoresTransformations)
        f = Qt.QFont('Courier', pointSize=14, weight=Qt.QFont.Bold)
        f.setKerning(False)
        f.setStyleHint(Qt.QFont.Monospace, Qt.QFont.OpenGLCompatible | Qt.QFont.PreferQuality)
        # Necessary to prevent context information from disappearing when mouse pointer passes over
        # context info text
        self.setAcceptHoverEvents(False)
        self.setAcceptedMouseButtons(Qt.Qt.NoButton)
        self.setFont(f)
        c = Qt.QColor(45,255,70,255)
        self.setDefaultTextColor(c)
        self.setZValue(10)

    def type(self):
        return ContextualInfoTextItem.QGRAPHICSITEM_TYPE

    def on_shader_view_scene_rect_changed(self, shader_view):
        """Maintain position at top left corner of shader_view."""
        topleft = Qt.QPoint()
        if shader_view.mapFromScene(self.pos()) != topleft:
            self.setPos(shader_view.mapToScene(topleft))

    def on_update_contextual_info(self, string, is_html):
        if is_html:
            self.setHtml(string)
        else:
            self.setPlainText(string)

class ShaderItem(Qt.QGraphicsObject):
    def __init__(self, parent_item=None):
        super().__init__(parent_item)
        self.image = None
        self._image_id = 0
        self.setAcceptHoverEvents(True)
        self.progs = {}

    def build_shader_prog(self, desc, vert_fn, frag_fn, **frag_template_mapping):
        source_dpath = Path(__file__).parent / 'shaders'
        prog = Qt.QOpenGLShaderProgram(self)

        if not prog.addShaderFromSourceFile(Qt.QOpenGLShader.Vertex, str(source_dpath / vert_fn)):
            raise RuntimeError('Failed to compile vertex shader "{}" for {} {} shader program.'.format(vert_fn, type(self).__name__, desc))

        if len(frag_template_mapping) == 0:
            if not prog.addShaderFromSourceFile(Qt.QOpenGLShader.Fragment, str(source_dpath / frag_fn)):
                raise RuntimeError('Failed to compile fragment shader "{}" for {} {} shader program.'.format(frag_fn, type(self).__name__, desc))
        else:
            with (source_dpath / frag_fn).open('r') as f:
                frag_template = Template(f.read())
            if not prog.addShaderFromSourceCode(Qt.QOpenGLShader.Fragment, frag_template.substitute(frag_template_mapping)):
                raise RuntimeError('Failed to compile fragment shader "{}" for {} {} shader program.'.format(frag_fn, type(self).__name__, desc))

        if not prog.link():
            raise RuntimeError('Failed to link {} {} shader program.'.format(type(self).__name__, desc))
        self.progs[desc] = prog
        return prog

    def on_image_changing(self, image):
        self.image = image
        self._image_id += 1
        self.update()

    def _normalize_min_max(self, min_max):
        r = self.image.range
        min_max -= r[0]
        min_max /= r[1] - r[0]

class ShaderTexture:
    """QOpenGLTexture does not support support GL_LUMINANCE*_EXT, etc, as specified by GL_EXT_texture_integer,
    which is required for integer textures in OpenGL 2.1 (QOpenGLTexture does support GL_RGB*U/I formats,
    but these were introduced with OpenGL 3.0 and should not be relied upon in 2.1 contexts).  So, in
    cases where GL_LUMINANCE*_EXT format textures may be required, we use ShaderTexture rather than
    QOpenGLTexture."""
    def __init__(self, target):
        self.texture_id = GL().glGenTextures(1)
        self.target = target

    def bind(self):
        GL().glBindTexture(self.target, self.texture_id)

    def release(self):
        GL().glBindTexture(self.target, 0)

    def destroy(self):
        GL().glDeleteTextures(1, (self.texture_id,))
        del self.texture_id

class ShaderQOpenGLTexture(Qt.QOpenGLTexture):
    """ShaderQOpenGLTexture replaces QOpenGLTexture's release function with one that does not require an
    argument, simplifying the implementation of ShaderItem.free_shader_view_resources."""
    def release(self):
        super().release(0)
