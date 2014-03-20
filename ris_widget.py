import ctypes
import numpy
from OpenGL import GL
from OpenGL.arrays import vbo
import OpenGL.GL.shaders
import os
from PyQt5 import QtCore, QtGui, QtWidgets, QtOpenGL
import sip

from ris_widget_exceptions import *

class RisWidget(QtOpenGL.QGLWidget):
    '''RisWidget stands for Rapid Image Stream Widget.  If tearing is visible, try enabling vsync, and failing that, providing True
    for enableSwapInterval1_ argument.'''
    def __init__(self, parent_ = None, enableSwapInterval1_ = False):
        super().__init__(RisWidget._makeGlFormat(enableSwapInterval1_), parent_)
        self.enableSwapInterval1 = enableSwapInterval1_

    @staticmethod
    def _makeGlFormat(enableSwapInterval1_):
        glFormat = QtOpenGL.QGLFormat()
        # Our weakest target platform is Macmini6,1 having Intel HD 4000 graphics supporting up to OpenGL 4.1 on OS X
        glFormat.setVersion(1, 2)
        # We avoid relying on depcrecated fixed-function pipeline functionality; any attempt to use legacy OpenGL calls
        # should fail.
#       glFormat.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        # It's highly likely that enabling swap interval 1 will not ameliorate tearing: any display supporting GL 4.1
        # supports double buffering, and tearing should not be visible with double buffering.  Therefore, the tearing
        # is caused by vsync being off or some fundamental brain damage in your out-of-date X11 display server; further
        # breaking things with swap interval won't help.  But, perhaps you can manage to convince yourself that it's
        # tearing less, and by the simple expedient of displaying less, it will be.
        glFormat.setSwapInterval(enableSwapInterval1_)
        # Want hardware rendering (should be enabled by default, but this can't hurt)
        glFormat.setDirectRendering(True)
        # Likewise, double buffering should be enabled by default
        glFormat.setDoubleBuffer(True)
        return glFormat

    def initializeGL(self):
        self.shaderProgram = QtGui.QOpenGLShaderProgram(self)

        if not self.shaderProgram.addShaderFromSourceFile(QtGui.QOpenGLShader.Vertex, os.path.join(os.path.dirname(__file__), 'panel.glslv')):
            raise ShaderCompilationException(self.shaderProgram.log())

        if not self.shaderProgram.addShaderFromSourceFile(QtGui.QOpenGLShader.Fragment, os.path.join(os.path.dirname(__file__), 'image.glslf')):
            raise ShaderCompilationException(self.shaderProgram.log())

        if not self.shaderProgram.link():
            raise ShaderLinkingException(self.shaderProgram.log())

        # Note: self.shaderProgram.bind() is equivalent to GL.glUseProgram(self.shaderProgram.programId())
        if not self.shaderProgram.bind():
            raise ShaderBindingException(self.shaderProgram.log())

        quad = numpy.array([
            # Vertex positions
            -1.0, -1.0, 0.0, 1.0,
            1.0, -1.0, 0.0, 1.0,
            1.0,  1.0, 0.0, 1.0,
#           -1.0,  1.0, 0.0, 1.0,
            -1.0,  1.0, 0.0, 1.0], numpy.float32)
            # Texture coordinates
#           0.0, 0.0,
#           1.0, 0.0,
#           1.0, 0.0,
#           0.0, 0.0], numpy.float32)

        # Like other QtGui::QOpenGL.. primitives, QOpenGLBuffer's constructor does not assume that it is called
        # from within a valid GL context and therefore does not complete all requisite setup.  NB:
        # QtGui.QOpenGLBuffer.VertexBuffer represents GL_ARRAY_BUFFER.
#       self.quadShaderBuff = QtGui.QOpenGLBuffer(QtGui.QOpenGLBuffer.VertexBuffer)
#       if not self.quadShaderBuff.create():
#           raise BufferCreationException(self.shaderProgram.log())
#       if not self.quadShaderBuff.bind():
#           raise BufferBindingException(self.shaderProgram.log())

#       self.quadShaderBuff.setUsagePattern(QtGui.QOpenGLBuffer.StaticDraw)
#       self.quadShaderBuff.allocate(sip.voidptr(quad.data), quad.nbytes)

#       self.vbo = vbo.VBO(numpy.array([ [ 0, 1, 0 ], [ -1,-1, 0 ], [ 1,-1, 0 ], [ 2,-1, 0 ], [ 4,-1, 0 ], [ 4, 1, 0 ], [ 2,-1, 0 ], [ 4, 1, 0 ], [ 2, 1, 0 ] ], numpy.float32))
        self.vbo = vbo.VBO(quad)

        # quad is deleted to make clear that quad's data has been copied to the GPU
        del quad

#       GL.glVertexAttribPointer(0, 4, GL.GL_FLOAT, False, 0, None)
#       GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, False, 0, ctypes.c_void_p(4*4))

    def paintGL(self):
#       GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        if not self.shaderProgram.bind():
            raise ShaderBindingException(self.shaderProgram.log())
        try:
            self.vbo.bind()
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glVertexPointerf(self.vbo)
#           self.quadShaderBuff
            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)
        finally:
            self.vbo.unbind()
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            self.shaderProgram.release()
