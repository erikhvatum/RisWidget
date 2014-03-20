class RisWidgetException(BaseException):
    def __init__(self, description_):
        self.description = description_

class ShaderException(RisWidgetException):
    pass

class ShaderCompilationException(ShaderException):
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

