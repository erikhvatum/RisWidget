#version 410 core

layout (location = 0) in vec4 vert_pos;
layout (location = 1) in vec2 tex_coord;

out vec2 vs_tex_coord;

void main()
{
    gl_Position = vert_pos;
    vs_tex_coord = tex_coord;
}
