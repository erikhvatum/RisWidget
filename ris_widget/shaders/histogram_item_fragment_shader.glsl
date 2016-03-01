#version 120
#line 3
#extension GL_EXT_gpu_shader4 : require
// The MIT License (MIT)
// 
// Copyright (c) 2014 WUSTL ZPLAB
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
// 
// Authors: Erik Hvatum <ice.rikh@gmail.com> 

uniform usampler1D tex;
uniform vec2 inv_view_size;
uniform float x_offset;
uniform float x_factor;
uniform float inv_max_transformed_bin_val;
uniform float gamma_gamma;
uniform float opacity;

void main()
{
    float bin_value = float(texture1D(tex, gl_FragCoord.x * x_factor * inv_view_size.x + x_offset).r);
    float bin_height = pow(bin_value, gamma_gamma) * inv_max_transformed_bin_val;
    float intensity = 1.0f - clamp(floor((gl_FragCoord.y * inv_view_size.y) / bin_height), 0, 1);

    gl_FragColor = vec4(intensity, intensity, intensity, intensity * opacity);
}
