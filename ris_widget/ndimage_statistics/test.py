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
import unittest
from . import ndimage_statistics

IMAGE_SHAPE = (2560, 2160)
IMAGE_COUNT_PER_FLAVOR = 10

print('Preparing to generate test data...')

class ImageFlavor:
    __slots__ = 'name', 'dtype', 'interval', 'images'
    def __init__(self, name, dtype, interval):
        self.name = name
        self.dtype = dtype
        self.interval = interval
        center = (interval[0] + interval[1]) / 2
        stddev = (interval[1] - interval[0]) / 4
        def make_example():
            example = numpy.random.normal(center, stddev, size=(IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
            if numpy.issubdtype(dtype, numpy.integer):
                example = numpy.round(example)
            example[example < interval[0]] = interval[0]
            example[example > interval[1]] = interval[1]
            return example.astype(dtype).T
        self.images = [ndimage_statistics.pool.submit(make_example) for i in range(IMAGE_COUNT_PER_FLAVOR)]

IMAGE_FLAVORS = [
    ImageFlavor('uint8', numpy.uint8, (0, 255)),
    ImageFlavor('uint12', numpy.uint16, (0, 4096)),
    ImageFlavor('uint16', numpy.uint16, (0, 65535)),
    ImageFlavor('float32', numpy.float32, (-5, 5))
]
FLAVOR_COUNT = len(IMAGE_FLAVORS)
IMAGE_COUNT = IMAGE_COUNT_PER_FLAVOR * FLAVOR_COUNT

MASKS_IMAGE_SHAPE = [
    ndimage_statistics.pool.submit(
        lambda: numpy.random.randint(0, 2, (IMAGE_SHAPE[1], IMAGE_SHAPE[0])).astype(numpy.bool).T
    ) for i in range(IMAGE_COUNT_PER_FLAVOR)
]
MASKS_RANDOM_SHAPE = [
    ndimage_statistics.pool.submit(
        lambda: numpy.random.randint(
            0, 2, (
                numpy.random.randint(1, IMAGE_SHAPE[1]*2+1),
                numpy.random.randint(1, IMAGE_SHAPE[0]*2+1)
            )
        ).astype(numpy.bool).T
    ) for i in range(IMAGE_COUNT_PER_FLAVOR)
]

print('Generating {} test image{} ({} each for {})...'.format(
    IMAGE_COUNT,
    's' if IMAGE_COUNT > 1 else '',
    IMAGE_COUNT_PER_FLAVOR,
    ', '.join(f.name for f in IMAGE_FLAVORS)
))

for imflav in IMAGE_FLAVORS:
    imflav.images = [example.result() for example in imflav.images]
del imflav

print('Generating {0} {1} test mask{2} and {0} random size test mask{2}...'.format(
    IMAGE_COUNT_PER_FLAVOR,
    '{}x{}'.format(*IMAGE_SHAPE),
    's' if IMAGE_COUNT_PER_FLAVOR > 1 else ''
))

MASKS_IMAGE_SHAPE = [m.result() for m in MASKS_IMAGE_SHAPE]
MASKS_RANDOM_SHAPE = [m.result() for m in MASKS_RANDOM_SHAPE]

# TEST_MASKS

# IM_TYPES = [
#     ImType('uint8', numpy.uint8, (0, 255)),
#     ImType('uint12', numpy.uint16, (0, 4095)),
#     ImType('uint16', numpy.uint16, (0, 65535)),
#     ImType('float32', numpy.float32, (-5, 5))
# ]
#
# IM_TYPES = {
#     'uint8' :
# }
# IM_DTYPES = [numpy.uint8, numpy.uint16, numpy.float32]
# IM_DTYPE_NAMES = {numpy.uint8: 'uint8', numpy.uint16: 'uint16', numpy.float32: 'float32'}
# IM_DTYPE_RANGES = {
#     numpy.uint8: numpy.array((0, 255), dtype=numpy.uint8),
#     numpy.uint16: numpy.array((0, 65535), dtype=numpy.uint16),
#     numpy.float32: numpy.array((-5, 5), dtype=numpy.float32)
# }
#
#
# def _genu8(self):
#     im = numpy.random.normal(127, 16, self.IM_SHAPE)
#     im[im > 255] = 255
#     im[im < 0] = 0
#     return im.astype(numpy.uint8)
#
# def _genu16(self):
#     im = numpy.random.normal(32767, 1024, self.IM_SHAPE)
#     im[im > 65535] = 65535
#     im[im < 0] = 0
#     return im.astype(numpy.uint16)
#
# def _genfloat32(self):
#     return numpy.random.normal(size=self.IM_SHAPE).astype(numpy.float32)
#
# im_gens = [self._genu8, self._genu16, self._genfloat32]
#
# imfs = {im_dtype: [ndimage_statistics.pool.submit(im_gen) for i in range(self.IM_COUNT)] for (im_dtype, im_gen) in zip(self.IM_DTYPES, im_gens)}
# IMS = {im_dtype: [fut.result().swapaxes(0, 1) for fut in futs] for (im_dtype, futs) in imfs.items()}
# print('Generating test masks...')
# SAMESHAPE_MASKS = [numpy.random.randint(0, 2, self.IM_SHAPE).astype(numpy.bool).T for i in range(self.IM_COUNT)]
# OFFSHAPE_MASKS = [numpy.random.randint(0, 2, (self.IM_SHAPE[0] - i, self.IM_SHAPE[1] - i)).astype(numpy.bool).T for i in range(self.IM_COUNT)]
# del im_gens
# del imfs
#
#
# class NDImageStatisticsTestCase(unittest.TestCase):
#
#
#
#
#     def setUp(self):
        

# TODO: Identify best or at least reasonably common method of sharing test function cores with benchmark functions.
# Promising: http://stackoverflow.com/questions/24150016/how-to-benchmark-unit-tests-in-python-without-adding-any-code
# However, not exactly what we want.  This gives the time required to run each test function once, including the overhead
# of verifying output, whereas we seek the mean wall clock time required to run the method that the test function
# tests a number of times.
# So, we want to refine the scheme seen in _test_and_benchmark such that test verifier and core function to be
# tested or benchmarked along with name and accepted or excluded input types.  With this, after moving the setUp()
# contents back to module scope, we can generate test methods for NDImageStatisticsTestCase and implement a simple
# benchmarker class.