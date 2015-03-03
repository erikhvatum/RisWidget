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
uniform float inv_max_transformed_bin_val;
uniform float gamma;
uniform float gamma_gamma;

void main()
{
    vec2 unit_coord = gl_FragCoord.xy * inv_view_size;

    float bin_value = float(texture1D(tex, unit_coord.x).r);
    float bin_height = pow(bin_value, gamma_gamma) * inv_max_transformed_bin_val;
    float intensity = 1.0f - clamp(floor(unit_coord.y / bin_height), 0, 1);
    vec4 outcolor = vec4(intensity, intensity , intensity, intensity);

    float plot_y = floor(pow(unit_coord.x, gamma) / inv_view_size.y);
    if(plot_y == floor(gl_FragCoord.y))
    {
        if(intensity > 0)
        {
            outcolor.rb = vec2(0.5,0.5);
        }
        else
        {
            outcolor.g = 1;
        }
    }

    gl_FragColor = outcolor;
}
