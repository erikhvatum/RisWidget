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

class NDImageStatisticsTestCase(unittest.TestCase):
    IM_DTYPES = [numpy.uint8, numpy.uint16, numpy.float32]
    IM_DTYPE_NAMES = {numpy.uint8: 'uint8', numpy.uint16: 'uint16', numpy.float32: 'float32'}
    IM_DTYPE_RANGES = {
        numpy.uint8: numpy.array((0, 255), dtype=numpy.uint8),
        numpy.uint16: numpy.array((0, 65535), dtype=numpy.uint16),
        numpy.float32: numpy.array((-5, 5), dtype=numpy.float32)
    }
    IM_SHAPE = (2160, 2560)
    IM_COUNT = 3

    def _genu8(self):
        im = numpy.random.normal(127, 16, self.IM_SHAPE)
        im[im > 255] = 255
        im[im < 0] = 0
        return im.astype(numpy.uint8)

    def _genu16(self):
        im = numpy.random.normal(32767, 1024, self.IM_SHAPE)
        im[im > 65535] = 65535
        im[im < 0] = 0
        return im.astype(numpy.uint16)

    def _genfloat32(self):
        return numpy.random.normal(size=self.IM_SHAPE).astype(numpy.float32)

    def setUp(self):
        im_gens = [self._genu8, self._genu16, self._genfloat32]
        print('Generating {} test image{}...'.format(self.IM_COUNT, 's' if self.IM_COUNT > 1 else ''))
        imfs = {im_dtype: [ndimage_statistics.pool.submit(im_gen) for i in range(self.IM_COUNT)] for (im_dtype, im_gen) in zip(self.IM_DTYPES, im_gens)}
        self.ims = {im_dtype: [fut.result().swapaxes(0, 1) for fut in futs] for (im_dtype, futs) in imfs.items()}
        print('Generating test masks...')
        self.sameshape_masks = [numpy.random.randint(0, 2, self.IM_SHAPE).astype(numpy.bool).T for i in range(self.IM_COUNT)]
        self.offshape_masks = [numpy.random.randint(0, 2, (self.IM_SHAPE[0] - i, self.IM_SHAPE[1] - i)).astype(numpy.bool).T for i in range(self.IM_COUNT)]

# TODO: Identify best or at least reasonably common method of sharing test function cores with benchmark functions.
# Promising: http://stackoverflow.com/questions/24150016/how-to-benchmark-unit-tests-in-python-without-adding-any-code
# However, not exactly what we want.  This gives the time required to run each test function once, including the overhead
# of verifying output, whereas we seek the mean wall clock time required to run the method that the test function
# tests a number of times.
# So, we want to refine the scheme seen in _test_and_benchmark such that test verifier and core function to be
# tested or benchmarked along with name and accepted or excluded input types.  With this, after moving the setUp()
# contents back to module scope, we can generate test methods for NDImageStatisticsTestCase and implement a simple
# benchmarker class.