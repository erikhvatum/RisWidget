#!/usr/bin/env python3

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

from distutils.core import setup
from distutils.extension import Extension
import numpy
import os
import subprocess
import sys

cpp_source = 'cython/_ndimage_statistics_impl.cpp'
cython_source = 'cython/_ndimage_statistics.pyx'
cythoned_source = 'cython/_ndimage_statistics.cpp'
cython_source_deps = ['cython/_ndimage_statistics_impl.h']

include_dirs = [numpy.get_include()]

extra_compile_args = []
extra_link_args = []
define_macros = []

if sys.platform != 'win32':
    extra_compile_args.extend(('-O3', '-march=native'))

try:
    from Cython.Distutils import build_ext

    ext_modules = [Extension('_ndimage_statistics',
                             sources = [cython_source, cpp_source],
                             include_dirs = include_dirs,
                             define_macros = define_macros,
                             language = 'c++',
                             depends = cython_source_deps,
                             extra_compile_args = extra_compile_args,
                             extra_link_args = extra_link_args
                             )]

    setup(name = 'ris_widget',
          cmdclass = {'build_ext' : build_ext},
          ext_modules = ext_modules)
except ImportError:
    print('Cython does not appear to be installed.  Attempting to use pre-made cpp file...')

    ext_modules = [Extension('_ndimage_statistics',
                             sources = [cythoned_source, cpp_source],
                             include_dirs = include_dirs,
                             define_macros = define_macros,
                             language = 'c++',
                             depends = cython_source_deps,
                             extra_compile_args = extra_compile_args,
                             extra_link_args = extra_link_args
                             )]

    setup(name = 'ris_widget',
          ext_modules = ext_modules)
