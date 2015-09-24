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
# Authors: Erik Hvatum <ice.rikh@gmail.com>, Zach Pincus

import distutils.core
import numpy
import pathlib
import sys

extra_compile_args = []
if sys.platform != 'win32':
    extra_compile_args.append('-std=c++11')

try:
    from Cython.Build import cythonize
    ext_processor = cythonize
except ImportError:
    def uncythonize(extensions, **_ignore):
        for extension in extensions:
            sources = []
            for src in map(pathlib.Path, extension.sources):
                if src.suffix == '.pyx':
                    if extension.language == 'c++':
                        ext = '.cpp'
                    else:
                        ext = '.c'
                    src = src.with_suffix(ext)
                sources.append(str(src))
            extension.sources[:] = sources
        return extensions
    ext_processor = uncythonize

_ndimage_statistics = distutils.core.Extension(
    'ris_widget.ndimage_statistics._ndimage_statistics',
    language = 'c++',
    sources = [
        'ris_widget/ndimage_statistics/_ndimage_statistics.pyx',
        'ris_widget/ndimage_statistics/_ndimage_statistics_impl.cpp'],
    depends = ['ris_widget/ndimage_statistics/_ndimage_statistics_impl.h'],
    extra_compile_args = extra_compile_args,
    include_dirs = [numpy.get_include()])

distutils.core.setup(
    classifiers = [
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
    package_data = {
        'ris_widget' : [
            'shaders/histogram_widget_fragment_shader_g.glsl',
            'shaders/histogram_widget_fragment_shader_rgb.glsl',
            'shaders/histogram_widget_vertex_shader.glsl',
            'shaders/image_widget_fragment_shader_template.glsl',
            'shaders/image_widget_vertex_shader.glsl']},
    ext_modules = ext_processor([_ndimage_statistics]),
    description = 'ris_widget (rapid image streaming widget) package',
    name = 'ris_widget',
    packages = [
        'ris_widget',
        'ris_widget.ndimage_statistics',
        'ris_widget.om',
        'ris_widget.om.signaling_list',
        'ris_widget.qdelegates',
        'ris_widget.qgraphicsitems',
        'ris_widget.qgraphicsviews',
        'ris_widget.qwidgets'],
    version = '1.2')
