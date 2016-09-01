# RisWidget


RisWidget is a PyQt5 widget for viewing and manipulating scientific images and composite images ("image stacks"), offering
an extensible and intuitive GUI and Python API convenient for use as a stand-alone application and in IPython for everyday
and specialized microscopy tasks.

## Required
* Python 3.5 or later (Python 3.4 compatibility is not regularly verified)
* Numpy
* PyQt 5.5 or later
* PyOpenGL
* A compatible OS such as Linux, OS X ([macOS](https://en.wikipedia.org/wiki/MacOS_Sierra)), Windows, or FreeBSD

## Recommended
* If you are installing RisWidget from source, a C++ compiler with C++11 support compatible with your Python installation is
recommended (eg, GCC with its G++ component, clang, Visual Studio 2015, the Intel C++ Compiler). Without this, histogram
computation and image loading are far slower.
* [freeimage-py](https://github.com/zpincus/freeimage-py) enables support for drag-and-drop and multithreaded image loading.
* IPython (see section *Using RisWidget in IPython*)

## Installation
RisWidget may be installed from source in the conventional manner:
```sh
$ git clone https://github.com/erikhvatum/RisWidget.git
$ cd RisWidget
$ python3 setup.py install
```

## Using RisWidget in IPython


## Annotator

```python

from ..qwidgets.flipbook_page_annotator import _BaseField, FlipbookPageAnnotator
import numpy
rw = RisWidget()
xr = numpy.linspace(0, 2*numpy.pi, 65536, True)
xg = xr + 2*numpy.pi/3
xb = xr + 4*numpy.pi/3
im = (((numpy.dstack(list(map(numpy.sin, (xr, xg, xb)))) + 1) / 2) * 65535).astype(numpy.uint16)
rw.flipbook_pages.append(im.swapaxes(0,1).reshape(256,256,3))
fpa = FlipbookPageAnnotator(
    rw.flipbook,
    'annotation',
    (
        ('foo', str, 'default_text'),
        ('bar', int, -11, -20, 35),
        ('baz', float, -1.1, -1000, 1101.111),
        ('choice', tuple, 'za', list('aaaa basd casder eadf ZZza aasdfer lo ad bas za e12 1'.split())),
        ('toggle', bool, False),
    )
)
fpa.show()
```