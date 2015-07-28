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
import numpy
import os
from pathlib import Path
import subprocess
import sys

cpp_source = 'ris_widget/ndimage_statistics/_ndimage_statistics_impl.cpp'
cython_source = 'ris_widget/ndimage_statistics/_ndimage_statistics.pyx'
cythoned_source = 'ris_widget/ndimage_statistics/_ndimage_statistics.cpp'
cython_source_deps = ['ris_widget/ndimage_statistics/_ndimage_statistics_impl.h']

include_dirs = [numpy.get_include()]

extra_compile_args = []
extra_link_args = []
define_macros = []

if sys.platform != 'win32':
    extra_compile_args.append('-std=c++11')

common_setup_args = {
    'classifiers' : [
        'Environment :: MacOS X',
        'Environment :: Win32 (MS Windows)',
        'Environment :: X11 Applications :: Qt',
        'Intended Audience :: Developers',
        'Intended Audience :: Education'
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: C++',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Multimedia :: Graphics :: Viewers',
        'Topic :: Multimedia :: Video :: Display',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Widget Sets'],
    'package_data' : {
        'ris_widget' : [
            'shaders/histogram_widget_fragment_shader_g.glsl',
            'shaders/histogram_widget_fragment_shader_rgb.glsl',
            'shaders/histogram_widget_vertex_shader.glsl',
            'shaders/image_widget_fragment_shader_template.glsl',
            'shaders/image_widget_vertex_shader.glsl']},
    'description' : 'ris_widget (rapid image streaming widget) package',
    'name' : 'ris_widget',
    'packages' : ['ris_widget'],
    'version' : '1.1'
    }

try:
    from Cython.Distutils import build_ext
    from Cython.Distutils.extension import Extension

    class build_ext_forced_rebuild(build_ext):
        def __init__(self, *va, **ka):
            super().__init__(*va, **ka)
            self.force = True

    ext_modules = [Extension('ris_widget.ndimage_statistics._ndimage_statistics',
                             sources = [cython_source, cpp_source],
                             include_dirs = include_dirs,
                             define_macros = define_macros,
                             language = 'c++',
                             depends = cython_source_deps,
                             extra_compile_args = extra_compile_args,
                             extra_link_args = extra_link_args,
                             cython_directives={'language_level' : 3}
                             )]

    setup(cmdclass = {'build_ext' : build_ext_forced_rebuild},
          ext_modules = ext_modules,
          **common_setup_args)
except ImportError:
    print('Cython does not appear to be installed.  Attempting to use pre-made cpp file...')
    from distutils.extension import Extension

    ext_modules = [Extension('ris_widget.ndimage_statistics._ndimage_statistics',
                             sources = [cythoned_source, cpp_source],
                             include_dirs = include_dirs,
                             define_macros = define_macros,
                             language = 'c++',
                             depends = cython_source_deps,
                             extra_compile_args = extra_compile_args,
                             extra_link_args = extra_link_args
                             )]

    setup(ext_modules = ext_modules,
          **common_setup_args)
