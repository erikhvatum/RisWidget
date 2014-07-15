// The MIT License (MIT)
//
// Copyright (c) 2014 Erik Hvatum
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "Common.h"

#if !defined(__APPLE__) && !defined(__MACOSX) && !defined(_WIN32)
 #ifdef __cplusplus
  extern "C"
  {
 #endif

 #include <GL/glx.h>
// #undef Bool
// #undef Status
// #undef CursorShape
// #undef Unsorted
// #undef None
// #undef KeyPress
// #undef Type
// #undef KeyRelease
// #undef FocusIn
// #undef FocusOut
// #undef FontChange
// #undef Expose
 #include <CL/cl.h>
 #include <CL/cl_gl.h>

 #include "GlxFunctions.h"

void implantClContextGlxSharingProperties(cl_context_properties* properties, unsigned int propertiesSize)
{
    properties[propertiesSize - 5] = CL_GL_CONTEXT_KHR;
    properties[propertiesSize - 4] = (cl_context_properties)glXGetCurrentContext();
    properties[propertiesSize - 3] = CL_GLX_DISPLAY_KHR;
    properties[propertiesSize - 2] = (cl_context_properties)glXGetCurrentDisplay();
    // Last element is 0 and serves as a terminator
}

 #ifdef __cplusplus
  }
 #endif
#endif
