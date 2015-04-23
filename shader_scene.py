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

from .image import Image
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
        self.update_contextual_info_signal.connect(self.contextual_info_text_item.on_update_contextual_info)
        self.shader_scene_view_rect_changed.connect(self.contextual_info_text_item.on_shader_view_scene_rect_changed)

    def add_contextual_info_item(self):
        self.contextual_info_text_item = ContextualInfoTextItem()
        self.addItem(self.contextual_info_text_item)

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

#   def paint(self, painter, option, widget):
#       qpicture = Qt.QPicture()
#       super().paint(painter, option, widget)
#       painter.setBrush(Qt.Qt.transparent)
#       color = Qt.QColor(Qt.Qt.black)
#       color.setAlphaF(self.opacity())
#       painter.setPen(Qt.QPen(color))
#       painter.drawPath(self.shape())

    def on_shader_view_scene_rect_changed(self, shader_view):
        """Maintain position at top left corner of shader_view."""
        topleft = Qt.QPoint()
        if shader_view.mapFromScene(self.pos()) != topleft:
            self.setPos(shader_view.mapToScene(topleft))

    def on_update_contextual_info(self, string, is_html):
        if is_html:
#           print(string)
            self.setHtml(string)
        else:
            self.setPlainText(string)

class ItemWithImage(Qt.QGraphicsObject):
    """See image_scene.ImageItem.__doc__ for more information regarding the image_about_to_change, image_changing, and
    image_changed signals."""

    # Signal arguments: ItemWithImage_derivative_instance, old_image, new_image.  Either old_image or
    # new_image may be None, but not both - changing from having no image to having no image does not
    # represent a change.  Reassigning the same Image instance to ItemWithImage.image is interpreted
    # as indicating that the content of the buffer backing ItemWithImage.image.data has changed; in this
    # case, (old_image is new_image) == True.
    image_about_to_change = Qt.pyqtSignal(object, object, object)
    # Signal arguments: Same as image_about_to_change
    image_changing = Qt.pyqtSignal(object, object, object)
    # Signal arguments: ItemWithImage_derivative_instance, image
    image_changed = Qt.pyqtSignal(object, object)

    def __init__(self, parent_item=None):
        super().__init__(parent_item)
        self._image = None
        self._image_id = 0
        self._show_frame = False
        self._frame_color = Qt.QColor(0, 255, 255, 128)
        self.setAcceptHoverEvents(True)

    def type(self):
        raise NotImplementedError()

    @property
    def image_data(self):
        """image_data property:
        The input assigned to this property may be None, in which case the current image and histogram views are cleared,
        and otherwise must be convertable to a 2D or 3D numpy array of shape (w, h) or (w, h, c), respectively*.  2D input
        is interpreted as grayscale.  3D input, depending on the value of c, is iterpreted as grayscale & alpha (c of 2),
        red & blue & green (c of 3), or red & blue & green & alpha (c of 4).

        The following dtypes are directly supported (data of any other type is converted to 32-bit floating point,
        and an exception is thrown if conversion fails):
        numpy.uint8
        numpy.uint16
        numpy.float32

        Supplying a numpy array of one of the above types as input may avoid an intermediate copy step by allowing RisWidget
        to keep a reference to the supplied array, allowing its data to be accessed directly.


        * IE, the iterable assigned to the image property is interpreted as an iterable of columns (image left to right), each
        containing an iterable of rows (image top to bottom), each of which is either a grayscale intensity value or an
        iterable of color channel intensity values (gray & alpha, or red & green & blue, or red & green & blue & alpha)."""
        return None if self._image is None else self._image.data

    @image_data.setter
    def image_data(self, image_data):
        self.image = None if image_data is None else Image(image_data)

    @property
    def image_data_T(self):
        """image_data_T property:
        The input assigned to this property may be None, in which case the current image and histogram views are cleared,
        and otherwise must be convertable to a 2D or 3D numpy array of shape (h, w) or (h, w, c), respectively*.  2D input
        is interpreted as grayscale.  3D input, depending on the value of c, is iterpreted as grayscale & alpha (c of 2),
        red & blue & green (c of 3), or red & blue & green & alpha (c of 4).

        The following dtypes are directly supported (data of any other type is converted to 32-bit floating point,
        and an exception is thrown if conversion fails):
        numpy.uint8
        numpy.uint16
        numpy.float32

        Supplying a numpy array of one of the above types as input may avoid an intermediate copy step by allowing RisWidget
        to keep a reference to the supplied array, allowing its data to be accessed directly.


        * IE, the iterable assigned to the image property is interpreted as an iterable of columns (image left to right), each
        containing an iterable of rows (image top to bottom), each of which is either a grayscale intensity value or an
        iterable of color channel intensity values (gray & alpha, or red & green & blue, or red & green & blue & alpha)."""
        if self._image is not None:
            return self._image.data_T

    @image_data_T.setter
    def image_data_T(self, image_data_T):
        self.image = None if image_data_T is None else Image(image_data_T, shape_is_width_height=False)

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        if image is self._image:
            if image is not None:
                # The same image is being reassigned, presumably because its data has been modified
                self.image_about_to_change.emit(self, image, image)
                self._image.recompute_stats()
                self._image_id += 1
                self.image_changing.emit(self, image, image)
                self.image_changed.emit(self, image)
        else:
            if image is not None and not issubclass(type(image), Image):
                e = 'The value assigned to the image property must either be derived '
                e+= 'from ris_widget.image.Image or must be None.  Did you mean to assign '
                e+= 'to the image_data property?'
                raise ValueError(e)
            old_image = self._image
            self.image_about_to_change.emit(self, old_image, image)
            self._image = image
            self._image_id += 1
            self.image_changing.emit(self, old_image, image)
            self.image_changed.emit(self, image)

    def _normalize_min_max(self, min_max):
        image = self.image
        # OpenGL normalizes uint16 data uploaded to float32 texture for the full uint16 range.  We store
        # our unpacked 12-bit images in uint16 arrays.  Therefore, OpenGL will normalize by dividing by
        # 65535, and we must follow suit, even though no 12-bit image will have a component value larger
        # than 4095.
        r = (0, 65535) if image.is_twelve_bit else self.image.range
        min_max -= r[0]
        min_max /= r[1] - r[0]

    def paint_frame(self, qpainter):
        if self._show_frame:
            qpainter.setBrush(Qt.QBrush(Qt.Qt.transparent))
            pen = Qt.QPen(self._frame_color)
            pen.setWidth(2)
            pen.setCosmetic(True)
            pen.setStyle(Qt.Qt.DotLine)
            qpainter.setPen(pen)
            qpainter.drawRect(self.boundingRect())

    @property
    def show_frame(self):
        return self._show_frame

    @show_frame.setter
    def show_frame(self, show_frame):
        if show_frame != self.show_frame:
            self._show_frame = show_frame
            self.update()

    @property
    def frame_color(self):
        return self._frame_color

    @frame_color.setter
    def frame_color(self, frame_color):
        assert isinstance(frame_color, Qt.QColor)
        self._frame_color = frame_color
        self.update()

class ShaderItemWithImage(ItemWithImage):
    def __init__(self, parent_item=None):
        super().__init__(parent_item)
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
    argument."""
    def release(self):
        super().release(0)
