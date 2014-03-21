#version 410 core

struct GammaTransformParams
{
    bool isEnabled;
    float minVal;
    float maxVal;
    float gammaVal;
};
uniform GammaTransformParams gtp = GammaTransformParams(false, 0.0, 65535.0, 1.0);
uniform usampler2D tex;

// From vertex shader
in vec2 vsTexCoord;

layout (location = 0) out vec4 fsColor;

void main()
{
    if(gtp.isEnabled)
    {
//      fsColor = vec4(0, 1, 0, 1);
        float intensity = texture(tex, vsTexCoord).r;
        if(intensity <= gtp.minVal)
        {
            fsColor = vec4(0, 0, 1, 1);
        }
        else if(intensity >= gtp.maxVal)
        {
            fsColor = vec4(1, 0, 0, 1);
        }
        else
        {
            float scaled = pow(clamp((intensity - gtp.minVal) / (gtp.maxVal - gtp.minVal), 0.0, 1.0), gtp.gammaVal);
//          float scaled = pow((intensity - gtp.minVal) / (gtp.maxVal - gtp.minVal), gtp.gammaVal);
            fsColor = vec4(scaled, scaled, scaled, 1.0);
        }
    }
    else
    {
        fsColor = vec4(vec3(texture(tex, vsTexCoord).rrr) / 65535.0, 1);
    }
}
