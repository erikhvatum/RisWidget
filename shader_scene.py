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
    FIXED_POSITION_IN_VIEW = Qt.QPoint(10, 5)

    text_changed = Qt.pyqtSignal()

    def __init__(self, parent_item=None):
        super().__init__(parent_item)
        self.setFlag(Qt.QGraphicsItem.ItemIgnoresTransformations)
        self._font = Qt.QFont('Courier', pointSize=16, weight=Qt.QFont.Bold)
        self._font.setKerning(False)
        self._font.setStyleHint(Qt.QFont.Monospace, Qt.QFont.OpenGLCompatible | Qt.QFont.PreferQuality)
        self._pen = Qt.QPen(Qt.QColor(Qt.Qt.black))
        self._pen.setWidth(2)
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

    def paint(self, qpainter, option, widget):
        self._update_picture()
        self._picture.play(qpainter)

    def return_to_fixed_position(self, view):
        """Maintain position self.FIXED_POSITION_IN_VIEW relative to view's top left corner."""
        topleft = self.FIXED_POSITION_IN_VIEW
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
                i.setFont(self._font)
                # Disabling brush/pen via setBrush/Pen(Qt.QBrush/Pen(Qt.Qt.NoBrush/Pen)) ought to be more intelligent
                # than disablind via setting to transparent color.  However, using NoBrush or NoPen here seems to
                # cause extreme painting slowdowns on OS X.
                transparent_color = Qt.QColor(Qt.Qt.transparent)
                if self._pen is None or self._brush is None:
                    i.setPen(Qt.QPen(transparent_color) if self._pen is None else self._pen)
                    i.setBrush(Qt.QBrush(transparent_color) if self._brush is None else self._brush)
                    i.paint(ppainter, Qt.QStyleOptionGraphicsItem(), None)
                else:
                    # To ensure that character outlines never obscure the entirety of character interior, outline
                    # is drawn first and interior second.  If both brush and pen are nonempty, Qt draws interior first
                    # and outline second.
                    i.setBrush(Qt.QBrush(transparent_color))
                    i.setPen(self._pen)
                    i.paint(ppainter, Qt.QStyleOptionGraphicsItem(), None)
                    i.setBrush(self._brush)
                    i.setPen(Qt.QPen(transparent_color))
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

    GAMMA_RANGE = (0.0625, 16.0)
    NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE = {
        numpy.bool8  : Qt.QOpenGLTexture.UInt8,
        numpy.uint8  : Qt.QOpenGLTexture.UInt8,
        numpy.uint16 : Qt.QOpenGLTexture.UInt16,
        numpy.float32: Qt.QOpenGLTexture.Float32}
    IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT = {
        'g'   : Qt.QOpenGLTexture.R32F,
        'ga'  : Qt.QOpenGLTexture.RG32F,
        'rgb' : Qt.QOpenGLTexture.RGB32F,
        'rgba': Qt.QOpenGLTexture.RGBA32F}
    IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT = {
        'g'   : Qt.QOpenGLTexture.Red,
        'ga'  : Qt.QOpenGLTexture.RG,
        'rgb' : Qt.QOpenGLTexture.RGB,
        'rgba': Qt.QOpenGLTexture.RGBA}
    IMAGE_TYPE_TO_GETCOLOR_EXPRESSION = {
        'g'   : 'vec4(s.r, s.r, s.r, 1.0f)',
        'ga'  : 'vec4(s.r, s.r, s.r, s.a)',
        'rgb' : 'vec4(s.r, s.g, s.b, 1.0f)',
        'rgba': 's'}

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
    trilinear_filtering_enabled_changed = Qt.pyqtSignal()
    auto_getcolor_expression_enabled_changed = Qt.pyqtSignal()
    getcolor_expression_changed = Qt.pyqtSignal()
    extra_transformation_expression_changed = Qt.pyqtSignal()
    min_changed = Qt.pyqtSignal()
    max_changed = Qt.pyqtSignal()
    gamma_changed = Qt.pyqtSignal()

    def __init__(self, parent_item=None):
        super().__init__(parent_item)
        self._image = None
        self._image_id = 0
        self._tex = None
        self._show_frame = False
        self._frame_color = Qt.QColor(255, 0, 0, 128)
        self._trilinear_filtering_enabled = True
        self._normalized_min = 0.0
        self._normalized_max = 1.0
        self._gamma = 1.0
        self.image_about_to_change.connect(self._on_image_about_to_change)
        self.image_changing.connect(self._on_image_changing)
        self.image_changed.connect(self.update)
        self._keep_auto_getcolor_expression_enabled_on_getcolor_expression_change = False
        self._auto_getcolor_expression_enabled = True
        self._getcolor_expression = None
        self._extra_transformation_expression = None
        self.auto_getcolor_expression_enabled_changed.connect(self._on_auto_getcolor_expression_enabled_changed)
        self.getcolor_expression_changed.connect(self._on_getcolor_expression_changed)
        self.auto_min_max_enabled_action = Qt.QAction('Auto Min/Max', self)
        self.auto_min_max_enabled_action.setCheckable(True)
        self.auto_min_max_enabled_action.setChecked(True)
        self.auto_min_max_enabled_action.toggled.connect(self._on_auto_min_max_enabled_action_toggled)
        self._keep_auto_min_max_on_min_max_value_change = False
        self.min_changed.connect(self._on_min_or_max_changed)
        self.max_changed.connect(self._on_min_or_max_changed)

    def type(self):
        raise NotImplementedError()

    def boundingRect(self):
        return Qt.QRectF(0,0,1,1) if self._image is None else Qt.QRectF(Qt.QPointF(), Qt.QSizeF(self._image.size))

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

    def _on_image_about_to_change(self, self_, old_image, new_image):
        if old_image is None or new_image is None or old_image.size != new_image.size:
            self.prepareGeometryChange()

    def _on_image_changing(self, self_, old_image, new_image):
        if self.auto_min_max_enabled:
            self.do_auto_min_max()
        if self.auto_getcolor_expression_enabled:
            self.do_auto_getcolor_expression()

    def _on_auto_getcolor_expression_enabled_changed(self):
        if self._auto_getcolor_expression_enabled:
            self.do_auto_getcolor_expression()

    def _on_getcolor_expression_changed(self):
        if self._auto_getcolor_expression_enabled and not self._keep_auto_getcolor_expression_enabled_on_getcolor_expression_change:
            self.auto_getcolor_expression_enabled = False

    def do_auto_getcolor_expression(self):
        self._keep_auto_getcolor_expression_enabled_on_getcolor_expression_change = True
        try:
            self.getcolor_expression = None if self._image is None else self.IMAGE_TYPE_TO_GETCOLOR_EXPRESSION[self._image.type]
        finally:
            self._keep_auto_getcolor_expression_enabled_on_getcolor_expression_change = False

    def _on_auto_min_max_enabled_action_toggled(self, v):
        if v:
            self.do_auto_min_max()

    def do_auto_min_max(self):
        image = self._image
        if image is not None:
            self._keep_auto_min_max_on_min_max_value_change = True
            try:
                mm = image.min_max
                if image.has_alpha_channel:
                    self.min = mm[:-1, 0].min()
                    self.max = mm[:-1, 1].max()
                else:
                    self.min, self.max = mm
            finally:
                self._keep_auto_min_max_on_min_max_value_change = False

    def _on_min_or_max_changed(self):
        if self.auto_min_max_enabled and not self._keep_auto_min_max_on_min_max_value_change:
            self.auto_min_max_enabled = False

    def _update_tex(self, estack, texture_unit=0):
        """Meant to be executed between a pair of QPainter.beginNativePainting() QPainter.endNativePainting() calls or,
        at the very least, when an OpenGL context is current, _update_tex does whatever is required for self._tex to
        represent self._image, including texture object creation and texture data uploading, and it leaves self._tex bound
        to the specified texture unit if possible.  Additionally, if self._image, is None and self._tex is not, self._tex
        is destroyed."""
        if self._image is None:
            if self._tex is not None:
                self._tex.destroy()
                self._tex = None
        else:
            tex = self._tex
            image = self._image
            desired_texture_format = self.IMAGE_TYPE_TO_QOGLTEX_TEX_FORMAT[image.type]
            desired_minification_filter = Qt.QOpenGLTexture.LinearMipMapLinear if self._trilinear_filtering_enabled else Qt.QOpenGLTexture.Linear
            if tex is not None:
                if image.size != Qt.QSize(tex.width(), tex.height()) or tex.format() != desired_texture_format or tex.minificationFilter() != desired_minification_filter:
                    tex.destroy()
                    tex = self._tex = None
            if tex is None:
                tex = Qt.QOpenGLTexture(Qt.QOpenGLTexture.Target2D)
                tex.setFormat(desired_texture_format)
                tex.setWrapMode(Qt.QOpenGLTexture.ClampToEdge)
                if self._trilinear_filtering_enabled:
                    tex.setMipLevels(6)
                    tex.setAutoMipMapGenerationEnabled(True)
                else:
                    tex.setMipLevels(1)
                    tex.setAutoMipMapGenerationEnabled(False)
                tex.setSize(image.size.width(), image.size.height(), 1)
                tex.allocateStorage()
                tex.setMinMagFilters(desired_minification_filter, Qt.QOpenGLTexture.Nearest)
                tex.image_id = -1
            tex.bind(texture_unit)
            estack.callback(lambda: tex.release(texture_unit))
            if tex.image_id != self._image_id:
#               import time
#               t0=time.time()
                pixel_transfer_opts = Qt.QOpenGLPixelTransferOptions()
                pixel_transfer_opts.setAlignment(1)
                tex.setData(self.IMAGE_TYPE_TO_QOGLTEX_SRC_PIX_FORMAT[image.type],
                            self.NUMPY_DTYPE_TO_QOGLTEX_PIXEL_TYPE[image.dtype],
                            image.data.ctypes.data,
                            pixel_transfer_opts)
#               t1=time.time()
#               print('tex.setData {}ms / {}fps'.format(1000*(t1-t0), 1/(t1-t0)))
                tex.image_id = self._image_id
                # self._tex is updated here and not before so that any failure preparing tex results in a retry the next time self._tex
                # is needed
                self._tex = tex

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

    @property
    def trilinear_filtering_enabled(self):
        """If set to True (the default), trilinear filtering is used for minification (zooming out).  This is
        somewhat higher quality than simple linear filtering, but it requires mipmap computation, which is too slow
        for 30fps display of 2560x2160 16bpp grayscale images on the ZPLAB acquisition computer.  Trilinear filtering
        as a minification filter tends to preserve some representation of small image details that simply disappear
        with linear filtering, so it is therefore desirable when frame rate is not of paramount importance.

        As compared to trilinear filtering, bilinear filtering would provide higher textured fragment fill rate - 
        which is not our bottleneck - at slightly lower quality while still requiring mipmap computation - which 
        is our bottleneck.  So, for the purposes of ZPLAB, trilinear and linear minification filters are the sensible
        choices, and this property selects between the two."""
        return self._trilinear_filtering_enabled

    @trilinear_filtering_enabled.setter
    def trilinear_filtering_enabled(self, trilinear_filtering_enabled):
        if trilinear_filtering_enabled != self._trilinear_filtering_enabled:
            self._trilinear_filtering_enabled = trilinear_filtering_enabled
            self.trilinear_filtering_enabled_changed.emit()

    @property
    def auto_getcolor_expression_enabled(self):
        """If set to True (the default), self.getcolor_expression is set automatically to a value appropriate for
        the type of the current image."""
        return self._auto_getcolor_expression_enabled

    @auto_getcolor_expression_enabled.setter
    def auto_getcolor_expression_enabled(self, v):
        if v != self._auto_getcolor_expression_enabled:
            self._auto_getcolor_expression_enabled = v
            self.auto_getcolor_expression_enabled_changed.emit()

    @property
    def getcolor_expression(self):
        """This property contains the GLSL 1.2 expression executed to transform fetched image texture (self._tex) data
        prior to applying min/max rescaling, gamma adjustment, and possibly composition with overlays.  In the case
        of a single channel image, this is particularly important: the fetched data will contain zeros for all color
        components except for red, making it necessary to use the red component value for green and blue as well in
        order for the image to be rendered in grayscale.  For images without an alpha channel, it is recommended that
        getcolor_expression should supply 1 for alpha component value.

        The four channel, raw texture fetch result is available in the variable "s", and the result of evaluating
        the getcolor_expression is used directly as input for the next step in the fragment shader (min/max scaling
        and gamma transform).  So, to display a single channel image as grayscale, getcolor_expression may be
        "vec4(s.r, s.r, s.r, 1.0f)".  To display a four channel RGBA image as is, getcolor_expression may be simply
        "s", or, equivalently, "vec4(s.r, s.g, s.b, s.a)"."""
        return self._getcolor_expression

    @getcolor_expression.setter
    def getcolor_expression(self, v):
        if v != self._getcolor_expression:
            self._getcolor_expression = v
            self.getcolor_expression_changed.emit()

    @property
    def extra_transformation_expression(self):
        """This property contains the optional GLSL 1.2 expression that transforms the output of min/max rescaling and
        gamma transformation (available in the rgb vector "sc").  For example, to invert after rescaling and gamma
        have been applied, extra_transformation_expression should be "vec3(1,1,1) - sc".

        If the extra_transformation_expression property is None, the extra transformation step is skipped and the output
        min/max rescaling and gamma are used without modification."""
        return self._extra_transformation_expression

    @extra_transformation_expression.setter
    def extra_transformation_expression(self, v):
        if self._extra_transformation_expression != v:
            self._extra_transformation_expression = v
            self.extra_transformation_expression_changed.emit()

    @property
    def auto_min_max_enabled(self):
        return self.auto_min_max_enabled_action.isChecked()

    @auto_min_max_enabled.setter
    def auto_min_max_enabled(self, v):
        self.auto_min_max_enabled_action.setChecked(v)

    @property
    def normalized_min(self):
        return self._normalized_min

    @normalized_min.setter
    def normalized_min(self, v):
        v = float(v)
        if self._normalized_min != v:
            self._normalized_min = v
            self.min_changed.emit()
            if self._normalized_min > self._normalized_max:
                self._normalized_max = v
                self.max_changed.emit()

    @normalized_min.deleter
    def normalized_min(self):
        self._normalized_min = 0.0
        self.min_changed.emit()

    @property
    def min(self):
        return self._denormalize_to_image_range(self._normalized_min)

    @min.setter
    def min(self, v):
        v = self._normalize_from_image_range(float(v))
        if not 0 <= v <= 1:
            raise ValueError('The value assigned to min must lie in the interval [{}, {}].'.format(
                self._denormalize_to_image_range(0.0), self._denormalize_to_image_range(1.0)))
        self.normalized_min = v

    @min.deleter
    def min(self):
        del self.normalized_min

    @property
    def normalized_max(self):
        return self._normalized_max

    @normalized_max.setter
    def normalized_max(self, v):
        v = float(v)
        if self._normalized_max != v:
            self._normalized_max = v
            self.max_changed.emit()
            if self._normalized_max < self._normalized_min:
                self._normalized_min = v
                self.min_changed.emit()

    @normalized_max.deleter
    def normalized_max(self):
        self._normalized_max = 1.0
        self.max_changed.emit()

    @property
    def max(self):
        return self._denormalize_to_image_range(self._normalized_max)

    @max.setter
    def max(self, v):
        v = self._normalize_from_image_range(float(v))
        if not 0 <= v <= 1:
            raise ValueError('The value assigned to {}.max must lie in the closed interval [{}, {}].'.format(
                type(self).__name__,
                self._denormalize_to_image_range(0.0), self._denormalize_to_image_range(1.0)))
        self.normalized_max = v

    @max.deleter
    def max(self):
        del self.normalized_max

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, v):
        v = float(v)
        if v != self._gamma:
            if not ItemWithImage.GAMMA_RANGE[0] <= v <= ItemWithImage.GAMMA_RANGE[1]:
                raise ValueError('The value assigned to {}.gamma must lie in the interval [{}, {}].'.format(type(self).__name__, *ItemWithImage.GAMMA_RANGE))
            self._gamma = v
            self.gamma_changed.emit()

    @gamma.deleter
    def gamma(self):
        self._gamma = 1.0
        self.gamma_changed.emit()

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
