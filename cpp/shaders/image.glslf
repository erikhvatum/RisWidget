#version 410 core
// #extension GL_ARB_separate_shader_objects : enable

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


// layout (origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord;
uniform sampler2D tex;
// 2D homogeneous transformation matrix for transforming gl_FragCoord viewport coordinates into image texture
// coordinates
uniform mat3 fragToTex;

layout (location = 0) out vec4 fsColor;

void main()
{
    // Note that gl_FragCoord is a homogeneous 3D coordinate.  Its scaling term (.w) is used as the scaling term for its
    // X, Y 2D homogeneous coordinate (.z).
    vec3 texCoordh = fragToTex * vec3(gl_FragCoord.x, gl_FragCoord.y, gl_FragCoord.w);
    // Compute 2D coordinate from homogeneous 2D coordinate
    vec2 texCoord = texCoordh.xy / texCoordh.z;

    // Render nothing if fragment coordinate is outside of texture.  This should only happen for a display row or column
    // at the edge of the quad.
//  if(texCoord.x < 0 || texCoord.x >= 1)
//  {
//      discard;
//  }
//  if(texCoord.y < 0 || texCoord.y >= 1)
//  {
//      discard;
//  }

    float v;
    if(texCoord.x < 0.5)
    {
        if(texCoord.y < 0.5)
        {
            v = fract(textureSize(tex, 0).x * texCoord.x);
        }
        else
        {
            v = fract(textureSize(tex, 0).y * texCoord.y);
        }
    }
    else
    {
//      texCoord = (floor(textureSize(tex, 0) * texCoord) + 0.5) / textureSize(tex, 0);
        v = texture(tex, texCoord).r;
    }
    fsColor = vec4(v, v, v, 1);
}
