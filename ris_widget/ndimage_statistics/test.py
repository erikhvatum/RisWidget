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

from pathlib import Path
import functools
import math
import numpy
import time
import unittest

from . import _ndimage_statistics

#import freeimage; im = freeimage.read(str(Path(__file__).parent.parent.parent / 'Opteron_6300_die_shot_16_core_mod.jpg'))
#im[0,0,0]=1123.456
#mask = (freeimage.read('/home/ehvatum/code_repositories/ris_widget/top_left_g.png') / 256).astype(numpy.uint8)
#im = numpy.array([[0,1,2],[3,4,5],[6,7,8]],dtype=numpy.uint8).swapaxes(0,1)
#mask = numpy.array([[0,0,0],[0,0,1],[0,0,0]], dtype=bool).swapaxes(0,1)
im = numpy.array(list(range(15*5*3))).astype(numpy.uint8).reshape(5,15,3).swapaxes(0,1)
mask = numpy.zeros((5,15),dtype=numpy.uint8).T
mask[1,0] = 1
mask[1,1] = 1
#im[im==0]=45
stats = _ndimage_statistics.NDImageStatistics(im, (0, 255), mask, False)
#stats = _ndimage_statistics.NDImageStatistics(im, (0, 255), False)
stats.launch_computation()
#del stats
print(stats.image_stats)

import sys
sys.exit(0)

import functools
import numpy
import time
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
IMAGE_COUNT_PER_FLAVOR = 1

print('Preparing to generate test data...')

class ImageFlavor:
    __slots__ = 'name', 'dtype', 'interval', 'images'
    def __init__(self, name, dtype, interval):
        self.name = name
        self.dtype = dtype
        self.interval = numpy.array(interval, dtype=dtype)
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
    imflav.images = [image.result() for image in imflav.images]
del imflav

print('Generating {0} {1} test mask{2} and {0} random size test mask{2}...'.format(
    IMAGE_COUNT_PER_FLAVOR,
    '{}x{}'.format(*IMAGE_SHAPE),
    's' if IMAGE_COUNT_PER_FLAVOR > 1 else ''
))

MASKS_IMAGE_SHAPE = [m.result() for m in MASKS_IMAGE_SHAPE]
MASKS_RANDOM_SHAPE = [m.result() for m in MASKS_RANDOM_SHAPE]

ROI_CENTER_AND_RADIUS = (IMAGE_SHAPE[0] / 2, IMAGE_SHAPE[1] / 2), IMAGE_SHAPE[0] / 2

class NDImageStatisticsTestCase(unittest.TestCase):
    def assert_ndarray_equal(self, ndarray_a, ndarray_b):
        self.assertTrue((ndarray_a == ndarray_b).all())

TARGET_FUNC_DESCS = [
    dict(
        name='min_max',
        fast_func=_measures_fast._min_max,
        slow_func=_measures_slow._min_max,
        validator=NDImageStatisticsTestCase.assert_ndarray_equal,
    ),
    dict(
        name='min_max__mask_of_same_shape_as_image',
        fast_func=_measures_fast._min_max,
        slow_func=_measures_slow._min_max,
        validator=NDImageStatisticsTestCase.assert_ndarray_equal,
        masks=MASKS_IMAGE_SHAPE,
    ),
    dict(
        name='min_max__mask_of_random_shape',
        fast_func=_measures_fast._min_max,
        slow_func=_measures_slow._min_max,
        validator=NDImageStatisticsTestCase.assert_ndarray_equal,
        masks=MASKS_RANDOM_SHAPE,
    ),
    dict(
        name='min_max__roi',
        fast_func=functools.partial(_measures_fast._min_max, roi_center_and_radius=ROI_CENTER_AND_RADIUS),
        #slow_func=functools.partial(_measures_slow._min_max, roi_center_and_radius=ROI_CENTER_AND_RADIUS),
    ),
    dict(
        name='histogram',
        fast_func=_measures_fast._histogram,
        slow_func=_measures_slow._histogram,
        takes_bin_count_and_range_arg=True,
    ),
    dict(
        name='histogram__mask_of_same_shape_as_image',
        fast_func=_measures_fast._histogram,
        slow_func=_measures_slow._histogram,
        masks=MASKS_IMAGE_SHAPE,
        takes_bin_count_and_range_arg=True,
    ),
    dict(
        name='histogram__mask_of_random_shape',
        fast_func=_measures_fast._histogram,
        slow_func=_measures_slow._histogram,
        masks=MASKS_RANDOM_SHAPE,
        takes_bin_count_and_range_arg=True,
    ),
    dict(
        name='histogram__roi',
        fast_func=functools.partial(_measures_fast._histogram, roi_center_and_radius=ROI_CENTER_AND_RADIUS),
        takes_bin_count_and_range_arg=True,
    ),
    dict(
        name="statistics",
        fast_func=_measures_fast._statistics,
        slow_func=_measures_slow._statistics,
        accepted_dtypes=(numpy.integer,),
        takes_is_12_bit_arg=True,
    ),
    dict(
        name="statistics__mask_of_same_shape_as_image",
        fast_func=_measures_fast._statistics,
        slow_func=_measures_slow._statistics,
        accepted_dtypes=(numpy.integer,),
        takes_is_12_bit_arg=True,
        masks=MASKS_IMAGE_SHAPE,
    ),
    dict(
        name="statistics__mask_of_random_shape",
        fast_func=_measures_fast._statistics,
        slow_func=_measures_slow._statistics,
        accepted_dtypes=(numpy.integer,),
        takes_is_12_bit_arg=True,
    ),
    dict(
        name="statistics__roi",
        fast_func=functools.partial(_measures_fast._statistics, roi_center_and_radius=ROI_CENTER_AND_RADIUS),
        slow_func=functools.partial(_measures_slow._statistics, roi_center_and_radius=ROI_CENTER_AND_RADIUS),
        accepted_dtypes=(numpy.integer,),
        takes_is_12_bit_arg=True,
    )
]

def _benchmark(desc, flavor, use_fast_version_else_slow=True):
    func = None
    if use_fast_version_else_slow:
        if 'fast_func' in desc:
            func = desc['fast_func']
        else:
            if 'slow_func' in desc:
                print('***Fast implementation not available for {}; using slow implementation.***'.format(desc['name']), end='  ')
                func = desc['slow_func']
    else:
        if 'slow_func' in desc:
            func = desc['slow_func']
        else:
            if 'fast_func' in desc:
                print('***Slow implementation not available for {}; using fast implementation.***'.format(desc['name']), end='  ')
                func = desc['fast_func']
    if func is None:
        print('***No implementation available for {}; skipping.***'.format(desc['name']))
        return
    if desc.get('takes_is_12_bit_arg'):
        func = functools.partial(func, is_twelve_bit = flavor.name=='uint12')
    if desc.get('takes_bin_count_and_range_arg'):
        func = functools.partial(func, bin_count=1024, range_=flavor.interval)
    calls = []
    for idx, image in enumerate(flavor.images):
        if desc.get('masks'):
            calls.append(functools.partial(
                func,
                im=image,
                mask=desc['masks'][idx] if len(desc['masks']) > 1 else desc['masks'][0]
            ))
        else:
            calls.append(functools.partial(func, im=image))
    t0 = time.time()
    for call in calls:
        call()
    t1 = time.time()
    print('{}_{}: {}'.format(desc['name'], flavor.name, 1000 * ((t1 - t0) / len(calls))))

benchmarks = []

def _test(self, desc, flavor):
    fast_func = desc['fast_func']
    slow_func = desc['slow_func']
    if desc.get('takes_is_12_bit_arg'):
        fast_func = functools.partial(fast_func, is_twelve_bit = flavor.name=='uint12')
        slow_func = functools.partial(slow_func, is_twelve_bit = flavor.name=='uint12')
    if desc.get('takes_bin_count_and_range_arg'):
        fast_func = functools.partial(fast_func, bin_count=1024, range_=flavor.interval)
        slow_func = functools.partial(slow_func, bin_count=1024, range_=flavor.interval)
    for idx, image in enumerate(flavor.images):
        if desc.get('masks'):
            fast_result = fast_func(im=image, mask=desc['masks'][idx] if len(desc['masks']) > 1 else desc['masks'][0])
            slow_result = slow_func(im=image, mask=desc['masks'][idx] if len(desc['masks']) > 1 else desc['masks'][0])
        else:
            fast_result = fast_func(im=image)
            slow_result = slow_func(im=image)
    if desc.get('validator_takes_desc_and_flavor_args'):
        desc['validator'](self, fast_result, slow_result, desc=desc, flavor=flavor)
    else:
        desc['validator'](self, fast_result, slow_result)

for desc in TARGET_FUNC_DESCS:
    for flavor in IMAGE_FLAVORS:
        if 'accepted_dtypes' not in desc or any(numpy.issubdtype(flavor.dtype, accepted_dtype) for accepted_dtype in desc['accepted_dtypes']):
            if 'slow_func' in desc and 'fast_func' in desc and 'validator' in desc:
                 setattr(NDImageStatisticsTestCase, 'test_{}_{}'.format(desc['name'], flavor.name), functools.partialmethod(_test, desc=desc, flavor=flavor))
            # def test(self):
            #     if desc.
            benchmarks.append(functools.partial(_benchmark, desc, flavor))

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1 or sys.argv[1] not in ('test', 'benchmark', 'test_and_benchmark'):
        print("***Defaulting to running tests.  Supply benchmark or test_and_benchmark as an argument to do otherwise.***")
        unittest.main(argv=sys.argv)
    elif sys.argv[1] == 'test':
        unittest.main(argv=sys.argv[0:1] + sys.argv[2:])
    elif sys.argv[1] == 'benchmark':
        print("All results in milliseconds.  Running benchmarks...")
        for benchmark in benchmarks:
            benchmark()
    elif sys.argv[1] == 'test_and_benchmark':
        unittest.main(argv=sys.argv[0:1] + sys.argv[2:])
        print("\nAll results in milliseconds.  Running benchmarks...")
        for benchmark in benchmarks:
            benchmark()