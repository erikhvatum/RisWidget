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

uniform sampler2D tex;
uniform float tex_global_alpha;
uniform float item_opacity;
// 2D homogeneous transformation matrix for transforming gl_FragCoord viewport coordinates into image texture
// coordinates
uniform mat3 frag_to_tex;
uniform float rescale_min;
uniform float rescale_range;
uniform float gamma;
uniform float viewport_height;
$overlay_uniforms

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
    vec2 tex_coord = transform_frag_to_tex(frag_to_tex);

    // Render nothing if fragment coordinate is outside of texture. This should only happen for a display row or column
    // at the edge of the quad.
    if(tex_coord.x < 0 || tex_coord.x >= 1)
    {
        discard;
    }
    if(tex_coord.y < 0 || tex_coord.y >= 1)
    {
        discard;
    }

    vec4 s = texture2D(tex, tex_coord);
    s = ${getcolor_expression};
    float sa = clamp(s.a, 0, 1) * tex_global_alpha;
    vec3 sc = min_max_gamma_transform(s.rgb, rescale_min, rescale_range, gamma);
    ${extra_transformation_expression};
    vec3 sca = sc * sa;
    float da = sa;
    vec3 dca = sca;

    int i;
    float isa, ida, osa, oda, sada;

$do_overlay_blending

    gl_FragColor = vec4(dca / da, da * item_opacity);
}
