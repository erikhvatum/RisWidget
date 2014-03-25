#version 430 core

uniform mat4 projectionModelViewMatrix;
uniform uint binCount;
uniform float binScale;
layout (binding = 0, r32ui) uniform coherent uimage2DArray histograms;

in float binIndex;

void main()
{
    ivec3 histogramsSize = imageSize(histograms);
    uint binValue = 0;
    for(ivec3 p = ivec3(0, 0, 0); p[0] < histogramsSize[0]; ++p[0])
    {
        for(p[1] = 0; p[1] < histogramsSize[1]; ++p[1])
        {
            p[2] = int(binIndex);
            binValue += imageLoad(histograms, p).r;
        }
    }
    gl_Position = projectionModelViewMatrix * vec4((float(binIndex) / float(binCount) - 0.5) * 2,
                                                   (log(float(binValue)) / binScale - 0.5) * 2, 0.4, 1.0);
}
