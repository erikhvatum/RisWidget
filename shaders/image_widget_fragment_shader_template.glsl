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
$overlay_samplers
// 2D homogeneous transformation matrix for transforming gl_FragCoord viewport coordinates into image texture
// coordinates
uniform mat3 frag_to_tex;
$overlay_frag_to_texs
uniform $vcomponents_t vcomponent_rescale_mins;
uniform $vcomponents_t vcomponent_rescale_ranges;
uniform $vcomponents_t gammas;
uniform float viewport_height;

$vcomponents_t extract_vcomponents(vec4 tcomponents)
{
    return $extract_vcomponents;
}

$vcomponents_t transform_vcomponents($vcomponents_t vcomponents)
{
    return gammas == $vcomponents_ones_vector ? clamp((vcomponents - vcomponent_rescale_mins) / vcomponent_rescale_ranges, 0, 1) :
                                                pow(clamp((vcomponents - vcomponent_rescale_mins) / vcomponent_rescale_ranges, 0, 1), gammas);
}

vec4 combine_vt_components($vcomponents_t vcomponents, vec4 tcomponents)
{
    return $combine_vt_components;
}

vec2 transform_frag_to_tex(mat3 frag_to_tex_max)
{
    vec3 tex_coord_h = frag_to_tex * vec3(gl_FragCoord.x, viewport_height - gl_FragCoord.y, gl_FragCoord.w);
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

    vec4 tcomponents = texture2D(tex, tex_coord);
    $vcomponents_t vcomponents = extract_vcomponents(tcomponents);
    $vcomponents_t transformed_vcomponents = transform_vcomponents(vcomponents);
    vec4 t_transformed = combine_vt_components(transformed_vcomponents, tcomponents);

    $do_overlay_blending

    gl_FragColor = ${gl_FragColor};
}
