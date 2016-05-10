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

try:
    from .image_impl.cpp_image import CppImage as Image
except ImportError:
    import warnings
    warnings.warn('warning: Failed to load _cpp_image binary module; using slow histogram and extrema computation methods.')
    from .image_impl.py_image import PyImage as Image

Image.__doc__ = \
"""An instance of the Image class is a wrapper around a Numpy ndarray representing a single image, optional mask data, plus some related attributes and
a .name.

If an ndarray of supported dtype, shape, and striding is supplied as the data argument to Image's constructor or set method,
a reference to that ndarray is kept rather than a copy of it.  In such cases, if the wrapped data is subsequently modified,
care must be taken to call the Image's .refresh method before querying, for example, its .histogram property as changes to the
data are not automatically detected.

The attributes maintained by an Image instance fall into the following categories:
    * Properties that represent aspects of ONLY .data: .dtype, .strides.  For non-floating-point Images, .range is also in this category.  The
    .data_changed signal is emitted when a new value is assigned to .data or the .refresh method is called with data_changed=True in order to indicate
    that the contents of .data have been modified in place.
    * Plain instance attributes updated by .refresh() and .set(..): .size, .is_grayscale, .num_channels, .has_alpha_channel, .is_twelve_bit.  Although
    nothing prevents assigning over these attributes, doing so is not advised.  The .data_changed signal is emitted to indicate that the value of any
    or all of these attributes has changed.
    * Properties computed from the combination of .data, .mask, and .imposed_float_range.  These include .histogram, .histogram_max_bin, .extremae.
    * .mask: None or a 2D bool or uint8 ndarray with neither dimension smaller than the corresponding dimension of .data.  If mask is not None, only
    image pixels with non-zero mask counterparts contribute to the histogram and extremae.  Mask pixels outside of the image have no impact.  If mask
    is None, all image pixels are included.  The .mask_changed signal is emitted to indicate that .mask has been replaced or that the contents of .mask
    have changed.
    * Properties with individual change signals: .name.  It is safe to assign None in addition anything else that str(..) accepts as its argument to
    .name.  When the value of .name or .imposed_float_range is modified, .name_changed is emitted.

Additionally, emission of .data_changed, .mask_changed, or .name_changed causes emission of .changed."""