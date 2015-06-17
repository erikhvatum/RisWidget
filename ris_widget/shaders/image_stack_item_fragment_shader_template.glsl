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

uniform float image_stack_item_opacity;
uniform float viewport_height;
$uniforms

vec3 min_max_gamma_transform(vec3 cc, float rescale_min, float rescale_range, float gamma)
{
    return pow(clamp((cc.rgb - rescale_min) / rescale_range, 0, 1), vec3(gamma, gamma, gamma));
}

vec2 transform_frag_to_tex(mat3 frag_to_tex_mat)
{
    vec3 tex_coord_h = frag_to_tex_mat * vec3(gl_FragCoord.x, viewport_height - gl_FragCoord.y, gl_FragCoord.w);
    return tex_coord_h.xy / tex_coord_h.z;
}

void main()
{
    vec2 tex_coord;
    vec4 s;
    float sa, da;
    vec3 sc, dc;
    vec3 sca, dca;
    int i;
    float isa, ida, osa, oda, sada;

$main
    gl_FragColor = vec4(dca / da, da * image_stack_item_opacity);
}
