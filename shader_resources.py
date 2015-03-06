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

from PyQt5 import Qt

# GL is set to instance of QOpenGLFunctions_2_1 (or QOpenGLFunctions_2_0 for old PyQt5 versions) during
# creation of first OpenGL widget.  QOpenGLFunctions_VER are obese namespaces containing all OpenGL
# plain procedural C functions and #define values valid for the associated OpenGL version, in Python
# wrappers.  It doesn't matter which context birthed the QOpenGLFunctions_VER object, only that
# an OpenGL context is current for the thread executing a QOpenGLFunctions_VER method.
GL = None

def init_GL():
    if GL is None:
        context = Qt.QOpenGLContext.currentContext()
        if context is None:
            raise RuntimeError('No OpenGL context is current for this thread.')
        vp = Qt.QOpenGLVersionProfile()
        vp.setProfile(Qt.QSurfaceFormat.CompatibilityProfile)
        try:
            vp.setVersion(2, 1)
            GL = context.versionFunctions(vp)
        except AttributeError:
            vp.setVersion(2, 0)
            GL = context.versionFunctions(vp)
        if not GL:
            raise RuntimeError('Failed to retrieve OpenGL wrapper namespace.')
        if not GL.initializeOpenGLFunctions():
            raise RuntimeError('Failed to initialize OpenGL wrapper namespace.')

# Minimal OpenGL entity wrappers for cases where the QOpenGL* version is not ideal for our
# needs or does not exist.  For example, QOpenGLTexture does not support support GL_LUMINANCE32UI_EXT
# as specified by GL_EXT_texture_integer, which is required for integer textures in OpenGL 2.1
# (QOpenGLTexture does support GL_RGB*U/I formats, but these were introduced with OpenGL 3.0
# and should not be relied upon in 2.1 contexts).

class ShaderTexture:
    def __init__(self, target):
        self.texture_id = GL.glGenTextures(1)
        self.target = target

    def bind(self):
        GL.glBindTexture(self.target, self.texture_id)

    def release(self):
        GL.glBindTexture(self.target, 0)

    def free(self):
        GL.glDeleteTextures(1, self.texture_id)
        del self.texture_id
