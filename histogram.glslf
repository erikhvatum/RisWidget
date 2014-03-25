#version 430 core

uniform usampler2D tex;

// From vertex shader
in vec2 vsTexCoord;

layout (location = 0) out vec4 fsColor;

void main()
{
    fsColor = vec4(1, 1, 1, 1);
}
