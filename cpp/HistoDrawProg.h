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

class HistoDrawProg
  : public GlslProg
{
    Q_OBJECT;

public:
    enum Locations : int
    {
        VertCoordLoc = 0
    };

    explicit HistoDrawProg(QObject* parent);
    virtual ~HistoDrawProg();

    void init(QOpenGLFunctions_4_1_Core* glfs) override;

    QPointer<QOpenGLVertexArrayObject> m_binVao;
    std::unique_ptr<QOpenGLBuffer> m_binVaoBuff;
    const int m_pmvLoc;
    const int m_binCountLoc;
    const int m_binScaleLoc;
    const int m_gammaGammaValLoc;

    void setBinCount(const GLuint& binCount);
    void setBinScale(const GLfloat& binScale);
    void setGammaGammaVal(const GLfloat& gammaGammaVal);
    void setPmv(const glm::mat4& pmv);

private:
    GLuint m_binCount;
    GLfloat m_binScale;
    GLfloat m_gammaGammaVal;
    glm::mat4 m_pmv;
};
