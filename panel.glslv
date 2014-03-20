#version 410 core

uniform mat4 projectionModelViewMatrix;

layout (location = 0) in vec2 vertPos;
layout (location = 1) in vec2 texCoord;

out vec2 vsTexCoord;

void main()
{
    gl_Position = projectionModelViewMatrix * vec4(vertPos, 0.5, 1.0);
    vsTexCoord = texCoord;
}
