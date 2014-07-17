// The MIT License (MIT)
//
// Copyright (c) 2014 Erik Hvatum
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

#pragma once

#include "Common.h"
#include "GlslProg.h"

class ImageDrawProg
  : public GlslProg
{
    Q_OBJECT;

public:
    enum Locations : int
    {
        VertCoordLoc = 0
    };

    explicit ImageDrawProg(QObject* parent);
    virtual ~ImageDrawProg();

    void init(QOpenGLFunctions_4_1_Core* glfs) override;

    QPointer<QOpenGLVertexArrayObject> m_quadVao;
    QOpenGLBuffer m_quadVaoBuff;
    const int m_pmvLoc;
    const int m_fragToTexLoc;
    const int m_gtpMinLoc;
    const int m_gtpMaxLoc;
    const int m_gtpRangeLoc;
    const int m_gtpGammaLoc;
    const GLint m_drawImageLoc;
    const GLuint m_drawImagePassthroughIdx;
    const GLuint m_drawImageGammaIdx;

    void setGtpRange(const GLfloat& gtpMin, const GLfloat& gtpMax);
    void setGtpGamma(const GLfloat& gtpGamma);
    void setDrawImageSubroutineIdx(const GLuint& drawImageSubroutineIdx);

private:
    GLfloat m_gtpMin;
    GLfloat m_gtpMax;
    GLfloat m_gtpRange;
    GLfloat m_gtpGamma;
    GLuint m_drawImageSubroutineIdx;
};
