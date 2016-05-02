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

from . import ndimage_statistics
import numpy
import time

im_dtypes = [numpy.uint8, numpy.uint16, numpy.float32]
im_dtype_names = {numpy.uint8:'uint8', numpy.uint16:'uint16', numpy.float32:'float32'}
im_dtype_ranges = {
    numpy.uint8 : numpy.array((0,255),dtype=numpy.uint8),
    numpy.uint16 : numpy.array((0,65535), dtype=numpy.uint16),
    numpy.float32 : numpy.array((-5,5), dtype=numpy.float32)
}
im_shape = (2160, 2560)
def genu8():
    im = numpy.random.normal(127, 16, im_shape)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype(numpy.uint8)
def genu16():
    im = numpy.random.normal(32767, 1024, im_shape)
    im[im > 65535] = 65535
    im[im < 0] = 0
    return im.astype(numpy.uint16)
def genfloat32():
    return numpy.random.normal(size=im_shape).astype(numpy.float32)
im_gens = [genu8, genu16, genfloat32]
im_count = 20

print('Generating {} test image{}...'.format(im_count, 's' if im_count > 1 else ''))
imfs = {im_dtype : [ndimage_statistics.pool.submit(im_gen) for i in range(im_count)] for (im_dtype, im_gen) in zip(im_dtypes, im_gens)}
ims = {im_dtype : [fut.result().swapaxes(0,1) for fut in futs] for (im_dtype, futs) in imfs.items()}
print('Generating test masks...')
sameshape_masks = [numpy.random.randint(0, 2, im_shape).astype(numpy.bool).T for i in range(im_count)]
offshape_masks = [numpy.random.randint(0, 2, (im_shape[0]-1,im_shape[1]-1)).astype(numpy.bool).T for i in range(im_count)]
del imfs

class Inapplicable(Exception):
    pass

def run_mark(im_dtype, mark_name, mark, masks=None, threaded=False):
    t0 = time.time()
    try:
        if threaded:
            if masks is None:
                tasks = [ndimage_statistics.pool.submit(mark, im) for im in ims[im_dtype]]
            else:
                tasks = [ndimage_statistics.pool.submit(mark, im, mask=mask) for (im, mask) in zip(ims[im_dtype], masks)]
            [task.result() for task in tasks]
        else:
            if masks is None:
                for im in ims[im_dtype]:
                    mark(im)
            else:
                for im, mask in zip(ims[im_dtype], masks):
                    mark(im, mask=mask)
    except Inapplicable:
        return
    t1 = time.time()
    print('{}({}): {}'.format(mark_name, im_dtype_names[im_dtype], 1000*((t1 - t0)/len(ims[im_dtype]))))

def mark_ndimage_statistics_statistics(im, mask=None):
    if numpy.issubdtype(im.dtype, numpy.floating):
        raise Inapplicable
    ndimage_statistics.statistics(im, mask=mask)

def mark_ndimage_statistics_histogram(im, mask=None):
    ndimage_statistics.histogram(im, 1024, im_dtype_ranges[im.dtype.type], mask)

def benchmark():
    print("All results are mean of {} runs, in milliseconds.  Running benchmarks...".format(im_count))
    marks_and_names = [
        [ndimage_statistics.extremae, 'extremae'],
        [mark_ndimage_statistics_histogram, 'histogram'],
        [mark_ndimage_statistics_statistics, 'statistics']
    ]
    # Unmasked
    for mark, mark_name in marks_and_names:
        for im_dtype in im_dtypes:
            run_mark(im_dtype, mark_name, mark)
    # Masked
    for mark, mark_name in marks_and_names:
        for im_dtype in im_dtypes:
            run_mark(im_dtype, 'masked ' + mark_name, mark, masks=sameshape_masks)
    # Stretched mask
    for mark, mark_name in marks_and_names:
        for im_dtype in im_dtypes:
            run_mark(im_dtype, 'stretchedmasked ' + mark_name, mark, masks=offshape_masks)
    # SMP Unmasked
    for mark, mark_name in marks_and_names:
        for im_dtype in im_dtypes:
            run_mark(im_dtype, 'threaded ' + mark_name, mark, threaded=True)
    # SMP Masked
    for mark, mark_name in marks_and_names:
        for im_dtype in im_dtypes:
            run_mark(im_dtype, 'threaded masked ' + mark_name, mark, masks=sameshape_masks, threaded=True)
    # SMP Stretched mask
    for mark, mark_name in marks_and_names:
        for im_dtype in im_dtypes:
            run_mark(im_dtype, 'threaded stretchedmasked ' + mark_name, mark, masks=offshape_masks, threaded=True)

class TestFailed(Exception):
    pass

class TestFates:
    def __init__(self):
        self.skipped=0
        self.passed=0
        self.failed=0

def run_test(test_fates, im_dtype, test_name, test, masks=None, threaded=False):
    test_desc = '{}({})'.format(test_name, im_dtype_names[im_dtype])
    print('{}...'.format(test_desc))
    try:
        if masks is None:
            tasks = [ndimage_statistics.pool.submit(test, im) for im in ims[im_dtype]]
        else:
            tasks = [ndimage_statistics.pool.submit(test, im, mask=mask) for (im, mask) in zip(ims[im_dtype], masks)]
        [task.result() for task in tasks]
        test_fates.passed += 1
        print('\t... {} passed'.format(test_desc))
    except Inapplicable:
        test_fates.skipped += 1
        print('\t... {} skipped'.format(test_desc))
    except TestFailed:
        test_fates.failed += 1
        print('\t... {} failed'.format(test_desc))

def test_ndimage_statistics_statistics(im, mask=None):
    if numpy.issubdtype(im.dtype, numpy.floating):
        raise Inapplicable
    from ._measures_fast import _statistics as _fast_statistics
    from ._measures_slow import _statistics as _slow_statistics
    fast = _fast_statistics(im, False, mask)
    slow = _slow_statistics(im, False, mask)
    passed = True
    if not (fast.histogram == slow.histogram).all():
        print("  failure: fast.histogram != slow.histogram")
        passed = False
    if fast.min_max_intensity[0] != slow.min_max_intensity[0]:
        print("  failure: fast.min != slow.min ({} vs {})".format(fast.min_max_intensity[0], slow.min_max_intensity[0]))
        passed = False
    if fast.min_max_intensity[1] != slow.min_max_intensity[1]:
        print("  failure: fast.max != slow.max ({} vs {})".format(fast.min_max_intensity[1], slow.min_max_intensity[1]))
        passed = False
    if mask is None:
        pix_count = im.shape[0] * im.shape[1]
    else:
        pix_count = mask.sum()
    if mask is None or im.shape == mask.shape:
        if fast.histogram.sum() != pix_count:
            print("  failure: fast.histogram.sum() != pix_count ({} vs {})".format(fast.histogram.sum(), pix_count))
            passed = False
        if slow.histogram.sum() != pix_count:
            print("  failure: slow.histogram.sum() != pix_count ({} vs {})".format(slow.histogram.sum(), pix_count))
            passed = False
        if fast.max_bin != slow.max_bin:
            print("  failure: fast.max_bin != slow.max_bin ({} vs {})".format(fast.max_bin, slow.max_bin))
            passed = False
    if not passed:
        raise TestFailed

def test():
    test_fates = TestFates()
    print('Running tests...')
    if ndimage_statistics.USING_FAST_MEASURES:
        tests_and_names = [
            [test_ndimage_statistics_statistics, '_ndimage_statistics']
        ]
        # Unmasked
        for test, test_name in tests_and_names:
            for im_dtype in im_dtypes:
                run_test(test_fates, im_dtype, test_name, test)
        # Masked
        for test, test_name in tests_and_names:
            for im_dtype in im_dtypes:
                run_test(test_fates, im_dtype, 'masked ' + test_name, test, masks=sameshape_masks)
        # Stretched mask
        for test, test_name in tests_and_names:
            for im_dtype in im_dtypes:
                run_test(test_fates, im_dtype, 'stretchedmasked ' + test_name, test, masks=offshape_masks)
        print('passed: {}\tfailed: {}\tskipped: {}'.format(test_fates.passed, test_fates.failed, test_fates.skipped))
        return test_fates.failed == 0
    else:
        print('Running tests requires optional binary module built by setup.py.')
        return False

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        print("Defaulting to running tests.  Supply benchmark or test_and_benchmark as an argument to do otherwise.")
        if not test():
            sys.exit(-1)
    elif sys.argv[1] == 'test':
        if not test():
            sys.exit(-1)
    elif sys.argv[1] == 'benchmark':
        benchmark()
    elif sys.argv[1] == 'test_and_benchmark':
        if not test():
            sys.exit(-1)
        print('')
        benchmark()