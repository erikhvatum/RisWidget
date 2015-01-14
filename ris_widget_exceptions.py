# The MIT License (MIT)
#
# Copyright (c) 2014 WUSTL ZPLAB
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

from ris_widget.ris_exceptions import RisException

class RisWidgetException(RisException):
    pass

class ShaderException(RisWidgetException):
    pass

class ShaderCompilationException(ShaderException):
    pass

class ShaderValidationException(ShaderException):
    pass

class ShaderLinkingException(ShaderException):
    pass

class ShaderBindingException(ShaderException):
    pass

class BufferException(RisWidgetException):
    pass

# TODO: find a way to detect this error condition such that there would be occassion to throw this exception
#class BufferAllocationException(BufferException):
#   pass

class BufferCreationException(BufferException):
    pass

class BufferBindingException(BufferException):
    pass

