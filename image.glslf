#version 410 core

uniform sampler2D tex;

// From vertex shader
in vec2 vsTexCoord;

layout (location = 0) out vec4 fsColor;

void main()
{
    fsColor = texture(tex, vsTexCoord);
//  fsColor = vec4(0, 1, 0, 1);
}
