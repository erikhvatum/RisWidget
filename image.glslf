#version 120

//uniform sampler2D tex;

// From vertex shader
//in vec2 vs_tex_coord;

// layout (location = 0) out vec4 fs_color;

void main()
{
//  fs_color = texture(tex, vs_tex_coord);
    gl_FragColor = vec4(0, 1, 0, 1);
}
