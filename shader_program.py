# The MIT License (MIT)
#
# Copyright (c) 2014 Erik Hvatum
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

from OpenGL import GL
import OpenGL.GL.shaders as GLS
import os

from ris_widget.ris_widget_exceptions import *

class ShaderProgram:
    def __init__(self, context, sourceFileNamesTypesAndSubroutines, uniformNames=None, attributeNames=None):
        def listsrcs():
            srcs = ''
            first = True
            for sfnats in sourceFileNamesTypesAndSubroutines:
                if first:
                    first = False
                else:
                    srcs += ', '
                srcs += '"'
                srcs += sfnats[0]
                srcs += '"'
            return srcs

        self.context = context
        self.prog = GLS.glCreateProgram()
        shaders = []

        ## Compile shaders

        for sfnats in sourceFileNamesTypesAndSubroutines:
            shader = GLS.glCreateShader(sfnats[1])
            GLS.glShaderSource(shader, self._loadSource(sfnats[0]))
            shaders.append(shader)
            GLS.glCompileShader(shader)
            if GLS.glGetShaderiv(shader, GLS.GL_COMPILE_STATUS) == GL.GL_FALSE:
                log = GLS.glGetShaderInfoLog(shader).decode('utf-8')
                for shader in shaders:
                    GLS.glDeleteShader(shader)
                GL.glDeleteProgram(self.prog); self.prog = None
                raise ShaderCompilationException('Compilation of shader "{}" failed: {}'.format(sfnats[0], log))
            GLS.glAttachShader(self.prog, shader)

        ## Link shaders

        failed = None
        GLS.glLinkProgram(self.prog)
        if GLS.glGetProgramiv(self.prog, GLS.GL_VALIDATE_STATUS) == GL.GL_FALSE:
            failed = 'Validation'
        else:
            GLS.glValidateProgram(self.prog)
            if GLS.glGetProgramiv(self.prog, GLS.GL_LINK_STATUS) == GL.GL_FALSE:
                failed = 'Linking'

        for shader in shaders:
            GLS.glDeleteShader(shader)

        if failed is not None:
            log = GLS.glGetProgramInfoLog(self.prog).decode('utf-8')
            GL.glDeleteProgram(self.prog); self.prog = None
            raise ShaderValidationException('{} of shader program with source files {} failed: {}'.format(failed, listsrcs(), log))

        ## Make member variables storing location indexes of:

        # subroutines
        for sfnats in sourceFileNamesTypesAndSubroutines:
            if len(sfnats) >= 2:
                subroutineShaderType = sfnats[1]
                for subroutineTypeAndFuncs in sfnats[2:]:
                    v = GL.glGetSubroutineUniformLocation(self.prog, subroutineShaderType, subroutineTypeAndFuncs[0].encode('utf-8'))
                    if v < 0:
                        raise ShaderException('Failed to get uniform for location of subroutine "{}" in "{}".'.format(subroutineTypeAndFuncs[0], sfnats[0]))
                    self._addAttr(subroutineTypeAndFuncs[0], v)
                    for subroutineFunc in subroutineTypeAndFuncs[1:]:
                        v = GL.glGetSubroutineIndex(self.prog, subroutineShaderType, subroutineFunc.encode('utf-8'))
                        if v < 0:
                            raise ShaderException('Failed to get subroutine index of function "{}" in "{}".'.format(subroutineFunc, sfnats[0]))
                        self._addAttr(subroutineFunc, v)

        # uniforms
        for uniformName in uniformNames:
            if type(uniformName) == tuple or type(uniformName) == list:
                nameInProg, nameInClass = uniformName
            else:
                nameInProg = nameInClass = uniformName
            v = GL.glGetUniformLocation(self.prog, nameInProg.encode('utf-8'))
            if v < 0:
                raise ShaderException('Failed to get index of uniform "{}" in "{}".'.format(nameInProg, sfnats[0]))
            self._addAttr(nameInClass, v)

        # attributes
        for attributeName in attributeNames:
            v = GLS.glGetAttribLocation(self.prog, attributeName)
            if v < 0:
                raise ShaderException('Failed to get index of attribute "{}" in "{}".'.format(attributeName, sfnats[0]))
            self._addAttr(attributeName, v)

    def __del__(self):
        if self.prog is not None:
            self.context.makeCurrent()
            GL.glDeleteProgram(self.prog)

    def _loadSource(self, sourceFileName):
        with open(os.path.join(os.path.dirname(__file__), sourceFileName), 'r') as src:
            return src.read()

    def _addAttr(self, namePrefix, value, nameSuffix='Loc'):
        name = namePrefix + nameSuffix
        if name in self.__dict__:
            raise ShaderException('Duplicate shader program attribute name: ' + name)
        self.__setattr__(name, value)

    def use(self):
        GLS.glUseProgram(self.prog)

    def unuse(self):
        GLS.glUseProgram(0)

    def __enter__(self):
        self.use()

    def __exit__(self, blah_a, blah_b, blah_c):
        self.unuse()
