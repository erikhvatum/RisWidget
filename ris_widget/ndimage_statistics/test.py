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

from collections import namedtuple
import numpy
import unittest
try:
    from . import _measures_fast
except ImportError:
    import sys
    print('\n\n*** The ris_widget.ndimage_statistics._ndimage_statistics binary module must be built in order to test ris_widget.ndimage_statistics. ***\n\n', file=sys.stderr)
    raise
from . import _measures_slow
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
    ImageFlavor('uint12', numpy.uint16, (0, 4095)),
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

class NDImageStatisticsTestCase(unittest.TestCase):
    pass

TargetFuncDesc = namedtuple('TargetFuncDesc', ('name', 'fast_func', 'slow_func', 'validator', 'accepted_dtypes', 'takes_is_12_bit_arg', 'masks', 'kwargs'))

TARGET_FUNC_DESCS = [
    TargetFuncDesc(
        name='min_max',
        fast_func=_measures_fast._min_max,
        slow_func=_measures_slow._min_max,
        validator=NDImageStatisticsTestCase.assertSequenceEqual,
        accepted_dtypes=(numpy.integer, numpy.floating),
        takes_is_12_bit_arg=False,
        masks=None,
        kwargs={}
    ),
    TargetFuncDesc(
        name='min_max__mask_of_same_shape_as_image',
        fast_func=_measures_fast._min_max,
        slow_func=_measures_slow._min_max,
        validator=NDImageStatisticsTestCase.assertSequenceEqual,
        accepted_dtypes=(numpy.integer, numpy.floating),
        takes_is_12_bit_arg=False,
        masks=MASKS_IMAGE_SHAPE,
        kwargs={}
    ),
    TargetFuncDesc(
        name='min_max__mask_of_random_shape',
        fast_func=_measures_fast._min_max,
        slow_func=_measures_slow._min_max,
        validator=NDImageStatisticsTestCase.assertSequenceEqual,
        accepted_dtypes=(numpy.integer, numpy.floating),
        takes_is_12_bit_arg=False,
        masks=MASKS_RANDOM_SHAPE,
        kwargs={}
    ),
]

# for desc in TARGET_FUNC_DESCS:
#     for flavor in IMAGE_FLAVORS:
#         if any(numpy.issubdtype(flavor.dtype, accepted_dtype) for accepted_dtype in desc.accepted_dtypes):
#             def test(self):
#                 if desc.

def run_test():
    pass

def run_benchmark():
    pass

# TEST_MASKS



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