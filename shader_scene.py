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
from contextlib import ExitStack
import numpy
from pathlib import Path
from PyQt5 import Qt
from string import Template

class ShaderScene(Qt.QGraphicsScene):
    """Although the Qt Graphics View Framework supports multiple views into a single scene, we don't
    have a need for this capability, and we do not go out of our way to make it work correctly (which
    would entail signficant additional code complexity)."""

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

class ContextualInfoItem(Qt.QGraphicsObject):
    QGRAPHICSITEM_TYPE = UNIQUE_QGRAPHICSITEM_TYPE()

    text_changed = Qt.pyqtSignal()

    def __init__(self, parent_item=None):
        super().__init__(parent_item)
        self.setFlag(Qt.QGraphicsItem.ItemIgnoresTransformations)
        self._font = Qt.QFont('Courier', pointSize=30, weight=Qt.QFont.Bold)
        self._font.setKerning(False)
        self._font.setStyleHint(Qt.QFont.Monospace, Qt.QFont.OpenGLCompatible | Qt.QFont.PreferQuality)
        self._pen = Qt.QPen(Qt.QColor(Qt.Qt.black))
        self._pen.setWidth(1)
        self._pen.setCosmetic(True)
        self._brush = Qt.QBrush(Qt.QColor(45,255,70,255))
        self._text = None
        self._text_flags = Qt.Qt.AlignLeft | Qt.Qt.AlignTop | Qt.Qt.AlignAbsolute
        self._picture = None
        self._bounding_rect = None
        # Necessary to prevent context information from disappearing when mouse pointer passes over
        # context info text
        self.setAcceptHoverEvents(False)
        self.setAcceptedMouseButtons(Qt.Qt.NoButton)
        # Info text generally should appear over anything else rather than z-fighting
        self.setZValue(10)
        self.hide()

    def type(self):
        return ContextualInfoItem.QGRAPHICSITEM_TYPE

    def boundingRect(self):
        if self._text:
            self._update_picture()
            return self._bounding_rect
        else:
            return Qt.QRectF(0,0,1,1)

    def paint(self, painter, option, widget):
        self._update_picture()
        self._picture.play(painter)

    def _on_view_scene_region_changed(self, view):
        """Maintain position at top left corner of view."""
        topleft = Qt.QPoint()
        if view.mapFromScene(self.pos()) != topleft:
            self.setPos(view.mapToScene(topleft))

    def _update_picture(self):
        if self._picture is None:
            self._picture = Qt.QPicture()
            ppainter = Qt.QPainter()
            with ExitStack() as stack:
                ppainter.begin(self._picture)
                stack.callback(ppainter.end)
                # The Qt API calls required for formatting multiline text such that it can be rendered to
                # a path are private, as can be seen in the implementation of
                # QGraphicsSimpleTextItem::paint, pasted below.  (Specifically, QStackTextEngine is a private
                # component and thus not available through PyQt).
                #
                # void QGraphicsSimpleTextItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
                # {
                #     Q_UNUSED(widget);
                #     Q_D(QGraphicsSimpleTextItem);
                #
                #     painter->setFont(d->font);
                #
                #     QString tmp = d->text;
                #     tmp.replace(QLatin1Char('\n'), QChar::LineSeparator);
                #     QStackTextEngine engine(tmp, d->font);
                #     QTextLayout layout(&engine);
                #
                #     QPen p;
                #     p.setBrush(d->brush);
                #     painter->setPen(p);
                #     if (d->pen.style() == Qt::NoPen && d->brush.style() == Qt::SolidPattern) {
                #         painter->setBrush(Qt::NoBrush);
                #     } else {
                #         QTextLayout::FormatRange range;
                #         range.start = 0;
                #         range.length = layout.text().length();
                #         range.format.setTextOutline(d->pen);
                #         QList<QTextLayout::FormatRange> formats;
                #         formats.append(range);
                #         layout.setAdditionalFormats(formats);
                #     }
                #
                #     setupTextLayout(&layout);
                #     layout.draw(painter, QPointF(0, 0));
                #
                #     if (option->state & (QStyle::State_Selected | QStyle::State_HasFocus))
                #         qt_graphicsItem_highlightSelected(this, painter, option);
                # }
                #
                # We would just use QGraphicsSimpleTextItem directly, but it is not derived from QGraphicsObject, so
                # it lacks the QObject base class required for emitting signals.  It is not possible to add a QObject
                # base to a QGraphicsItem derivative in Python (it can be done in C++ - this is how QGraphicsObject
                # is implemented).
                #
                # Total lack of signal support is not acceptable; it would greatly complicate the task of
                # positioning a non-child item relative to a ContextualInfoItem.
                # 
                # However, it's pretty easy to use that very paint function to generate paint commands that
                # we cache in self._picture, so we do that.  For strings large enough that the relayout
                # performed on each refresh by QGraphicsSimpleTextItem exceeds the CPython interpreter overhead 
                # for initiating the QPicture replay, our paint function is faster.
                #
                # Additionally, QGraphicsTextItem is very featureful, has a QObject base, and would be the first
                # choice, but the one thing it can not do is outline text, so it's out.
                i = Qt.QGraphicsSimpleTextItem(self._text)
                i.setPen(Qt.Qt.NoPen if self._pen is None else self._pen)
                i.setBrush(Qt.Qt.NoBrush if self._brush is None else self._brush)
                i.setFont(self._font)
                i.paint(ppainter, Qt.QStyleOptionGraphicsItem(), None)
                self._bounding_rect = i.boundingRect()

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, v):
        if self._text != v:
            if v:
                self.prepareGeometryChange()
            self._text = v
            self._picture = None
            if self._text:
                self.show()
                self.update()
            else:
                self.hide()
            self.text_changed.emit()

    @property
    def font(self):
        return self._font

    @font.setter
    def font(self, v):
        assert isinstance(v, Qt.QFont)
        self._font = v
        self._picture = None
        self.update()

    @property
    def pen(self):
        """The pen used to draw text outline to provide contrast against any background.  If None,
        outline is not drawn."""
        return self._pen

    @pen.setter
    def pen(self, v):
        assert isinstance(v, Qt.QPen) or v is None
        self._pen = v
        self._picture = None
        self.update()

    @property
    def brush(self):
        """The brush used to fill text.  If None, text is not filled."""
        return self._brush

    @brush.setter
    def brush(self, v):
        assert isinstance(v, Qt.QBrush) or v is None
        self._brush = v
        self._picture = None
        self.update()

class ItemWithImage(Qt.QGraphicsObject):
    """When the image_about_to_change(ItemWithImage, old_image, new_image) signal is emitted, ItemWithImage.image
    is still old_image.  This is useful, for example, in the case of informing Qt of bounding rect change via
    QGraphicsItem.prepareGeometryChange(), at which time ItemWithImage.boundingRect() must still return the old rect
    value (which, itself, varies with ItemWithImage.image).

    When the image_changing(ItemWithImage, old_image, new_image) signal is emitted, ItemWithImage.image has already
    been updated to the value represented by new_image.  The scene rect, the region of the scene framed by the view (which
    will change if zoom-to-fit is enabled and the new image is of different size than the old image), and the min/max
    values of the associated histogram item (which may change if auto-min/max is enabled) have not yet been updated -
    these are updated in response to image_changing.

    When the image_changed(ItemWithImage, image) signal is emitted, all RisWidget internals that vary with image
    have been updated.  The image_changed signal is emitted to provide an easy, catch-all opportunity to update
    anything with a state that is a function of the state of multiple other things that depend on image.

    For example, a hypothetical string containing the region of the image framed by the view and the gamma slider
    min/max values depends on two things that are updated in response to ItemWithImage.image_changing.  So, if this
    string is to be updated immediately (ie without waiting for an event loop iteration by using a queued
    connection) upon image change, it can not be in response to only ItemWithImage.image_changing - the changes
    needed in order to correctly assemble the string may not have yet occurred.  It can not be in response to only
    ImageScene.shader_scene_view_rect_changed - min/max may not yet have been updated.  Likewise, it can not be
    in response to only min/max changing - scene view rect may not yet have been updated.  It would be necessary
    to wait for both the rect and min/max change signals, updating the string only when the second of the pair arrive,
    and doing so such that the string is never updated with stale data in the face of failure for one of the signals
    to arrive would require having and clearing reception flags for both at the end of the current event loop iteration.

    Connecting updating of the hypothetical string to ItemWithImage.image_changed avoids all of these issues."""

    # Signal arguments: ItemWithImage subclass instance, old_image, new_image.  Either old_image or
    # new_image may be None, but not both - changing from having no image to having no image does not
    # represent a change.  Reassigning the same Image instance to ItemWithImage.image is interpreted
    # as indicating that the content of the buffer backing ItemWithImage.image.data has changed; in this
    # case, (old_image is new_image) == True.
    image_about_to_change = Qt.pyqtSignal(object, object, object)
    # Signal arguments: Same as image_about_to_change
    image_changing = Qt.pyqtSignal(object, object, object)
    # Signal arguments: ItemWithImage subclass instance, image
    image_changed = Qt.pyqtSignal(object, object)

    def __init__(self, parent_item=None):
        super().__init__(parent_item)
        self._image = None
        self._image_id = 0
        self._show_frame = False
        self._frame_color = Qt.QColor(255, 0, 0, 128)
        self.setAcceptHoverEvents(True)

    def type(self):
        raise NotImplementedError()

    def boundingRect(self):
        return Qt.QRectF(0,0,1,1) if self._image is None else Qt.QRectF(Qt.QPointF(), Qt.QSizeF(self._image.size))

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
                # The same image is being reassigned, presumably because its data has been modified.  In case other aspects
                # of the image has changed in ways that would cause texture upload or stats recomputation to fail, subclasses
                # are given the opportunity to reject the refresh.
                self.validate_image(image)
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
            self.validate_image(image)
            old_image = self._image
            self.image_about_to_change.emit(self, old_image, image)
            self._image = image
            self._image_id += 1
            self.image_changing.emit(self, old_image, image)
            self.image_changed.emit(self, image)

    def validate_image(self, image):
        """validate_image is provided for subclasses to override.  validate_image is called by the image property setter before
        any side effects occur, providing an opportunity to raise an exception and abort assignment of the new value to the image
        property - without requiring wholesale replacement of the image setter and/or duplication of aspects such as the image
        subclass check."""
        pass

    def _renormalize_for_gl(self, v):
        """OpenGL normalizes uint16 data uploaded to float32 texture for the full uint16 range.  We store
        our unpacked 12-bit images in uint16 arrays.  Therefore, OpenGL will normalize by dividing by
        65535, even though no 12-bit image will have a component value larger than 4095.  We normalize
        rescaling min and max values for 12-bit images by dividing by 4095.  This function converts from
        our normalized representation to that required by GL, when these two methods differ."""
        image = self._image
        if image is not None and image.is_twelve_bit:
            v *= 4095/65535
        return v

    def _normalize_from_image_range(self, v):
        image = self._image
        if image is not None:
            r = image.range
            v -= r[0]
            v /= r[1] - r[0]
        return v

    def _denormalize_to_image_range(self, v):
        image = self._image
        if image is not None:
            r = image.range
            v *= r[1] - r[0]
            v += r[0]
            if image.dtype is not numpy.float32:
                v = int(v)
        return v

    def _paint_frame(self, qpainter):
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

class ShaderItemMixin:
    def __init__(self):
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

class ShaderItem(ShaderItemMixin, Qt.QGraphicsObject):
    def __init__(self, parent_item=None):
        Qt.QGraphicsObject.__init__(self, parent_item)
        ShaderItemMixin.__init__(self)
        self.setAcceptHoverEvents(True)

    def type(self):
        raise NotImplementedError()

class ShaderItemWithImage(ShaderItemMixin, ItemWithImage):
    def __init__(self, parent_item=None):
        ItemWithImage.__init__(self, parent_item)
        ShaderItemMixin.__init__(self)

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
