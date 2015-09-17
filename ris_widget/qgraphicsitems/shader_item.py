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

from pathlib import Path
from PyQt5 import Qt
from string import Template
from ..shared_resources import QGL, NoGLContextIsCurrentError

class ShaderItemMixin:
    def __init__(self):
        self.progs = {}

    def build_shader_prog(self, desc, vert_fn, frag_fn, **frag_template_mapping):
        source_dpath = Path(__file__).parent.parent / 'shaders'
        prog = Qt.QOpenGLShaderProgram(self)

        if not prog.addShaderFromSourceFile(Qt.QOpenGLShader.Vertex, str(source_dpath / vert_fn)):
            raise RuntimeError('Failed to compile vertex shader "{}" for {} {} shader program.'.format(vert_fn, type(self).__name__, desc))

        if len(frag_template_mapping) == 0:
            if not prog.addShaderFromSourceFile(Qt.QOpenGLShader.Fragment, str(source_dpath / frag_fn)):
                raise RuntimeError('Failed to compile fragment shader "{}" for {} {} shader program.'.format(frag_fn, type(self).__name__, desc))
        else:
            with (source_dpath / frag_fn).open('r') as f:
                frag_template = Template(f.read())
            s=frag_template.substitute(frag_template_mapping)
#           print(s)
            if not prog.addShaderFromSourceCode(Qt.QOpenGLShader.Fragment, s):
                raise RuntimeError('Failed to compile fragment shader "{}" for {} {} shader program.'.format(frag_fn, type(self).__name__, desc))

        if not prog.link():
            raise RuntimeError('Failed to link {} {} shader program.'.format(type(self).__name__, desc))
        self.progs[desc] = prog
        return prog

    def set_blend(self, estack):
        GL = QGL()
        if not GL.glIsEnabled(GL.GL_BLEND):
            GL.glEnable(GL.GL_BLEND)
            estack.callback(lambda: GL.glDisable(GL.GL_BLEND))
        bfs = GL.glGetIntegerv(GL.GL_BLEND_SRC), GL.glGetIntegerv(GL.GL_BLEND_DST)
        if bfs != (GL.GL_SRC_ALPHA, GL.GL_DST_ALPHA):
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_DST_ALPHA)
            estack.callback(lambda: GL.glBlendFunc(*bfs))
        be = GL.glGetIntegerv(GL.GL_BLEND_EQUATION)
        if be != GL.GL_FUNC_ADD:
            GL.glBlendEquation(GL.GL_FUNC_ADD)
            estack.callback(lambda: GL.glBlendEquation(be))

class ShaderItem(ShaderItemMixin, Qt.QGraphicsObject):
    def __init__(self, parent_item=None):
        Qt.QGraphicsObject.__init__(self, parent_item)
        ShaderItemMixin.__init__(self)
        self.setAcceptHoverEvents(True)

    def type(self):
        raise NotImplementedError()

class ShaderTexture:
    """QOpenGLTexture does not support support GL_LUMINANCE*_EXT, etc, as specified by GL_EXT_texture_integer,
    which is required for integer textures in OpenGL 2.1 (QOpenGLTexture does support GL_RGB*U/I formats,
    but these were introduced with OpenGL 3.0 and should not be relied upon in 2.1 contexts).  So, in
    cases where GL_LUMINANCE*_EXT format textures may be required, we use ShaderTexture rather than
    QOpenGLTexture."""
    def __init__(self, target):
        self.texture_id = QGL().glGenTextures(1)
        self.target = target

    def __del__(self):
        self.destroy()

    def bind(self):
        QGL().glBindTexture(self.target, self.texture_id)

    def release(self):
        QGL().glBindTexture(self.target, 0)

    def destroy(self):
        if hasattr(self, 'texture_id'):
            try:
                QGL().glDeleteTextures(1, (self.texture_id,))
                del self.texture_id
            except NoGLContextIsCurrentError:
                pass
