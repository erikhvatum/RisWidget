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

import numpy
import setuptools
import setuptools.command
import setuptools.command.build_ext
import sys

# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')


class BuildExt(setuptools.command.build_ext.build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        cflags = self.c_opts.get(ct, [])
        ldflags = []
        if ct == 'unix':
            cflags.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                cflags.append('-fvisibility=hidden')
            if has_flag(self.compiler, '-mtune=native'):
                cflags.append('-mtune=native')
        for ext in self.extensions:
            ext.extra_compile_args = cflags
            ext.extra_link_args = ldflags
            setuptools.command.build_ext.build_ext.build_extensions(self)

ext_modules = [
    setuptools.Extension(
        'ris_widget.ndimage_statistics._ndimage_statistics',
        language='c++',
        sources=[
            'ris_widget/ndimage_statistics/_ndimage_statistics.cpp',
            'ris_widget/ndimage_statistics/NDImageStatistics.cpp',
            'ris_widget/ndimage_statistics/Luts.cpp'
        ],
        depends=[
            'ris_widget/ndimage_statistics/NDImageStatistics.h',
            'ris_widget/ndimage_statistics/NDImageStatistics_impl.h',
            'ris_widget/ndimage_statistics/Luts.h'
        ],
        include_dirs=[
            numpy.get_include(),
            'pybind11/include'
        ]
    ),
]

setuptools.setup(
    requires = [
        'numpy',
        'OpenGL',
        'PyQt5'
    ],
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
        'Topic :: Software Development :: Widget Sets'
    ],
    package_data = {
        'ris_widget' : [
            'icons/checked_box_icon.svg',
            'icons/disabled_checked_box_icon.svg',
            'icons/disabled_pseudo_checked_box_icon.svg',
            'icons/disabled_unchecked_box_icon.svg',
            'icons/disabled_wrong_type_checked_box_icon.svg',
            'icons/image_icon.svg',
            'icons/layer_icon.svg',
            'icons/layer_stack_icon.svg',
            'icons/pseudo_checked_box_icon.svg',
            'icons/unchecked_box_icon.svg',
            'icons/wrong_type_checked_box_icon.svg',
            'shaders/histogram_item_fragment_shader.glsl',
            'shaders/layer_stack_item_fragment_shader_template.glsl',
            'shaders/planar_quad_vertex_shader.glsl'
        ]
    },
    ext_modules = ext_modules,
    description = 'ris_widget (rapid image streaming widget) package',
    name = 'ris_widget',
    packages = [
        'ris_widget',
        'ris_widget.ndimage_statistics',
        'ris_widget.om',
        'ris_widget.om.signaling_list',
        'ris_widget.qdelegates',
        'ris_widget.qgraphicsitems',
        'ris_widget.qgraphicsscenes',
        'ris_widget.qgraphicsviews',
        'ris_widget.qwidgets'
    ],
    cmdclass={'build_ext': BuildExt},
    version = '1.5')
