import ctypes as ct
import numpy as np
from OpenGL import GL
from PyQt5 import QtCore, QtGui, QtWidgets, QtOpenGL, uic

class RisWidget(QtOpenGL.QGLWidget):
    '''RisWidget stands for Rapid Image Stream Widget'''
    
