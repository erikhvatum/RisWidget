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
        BinIndexLoc = 0
    };

    explicit HistoDrawProg(QObject* parent);
    virtual ~HistoDrawProg();

    void init(QOpenGLFunctions_4_1_Core* glfs) override;

    const int m_pmvLoc;
    const int m_binCountLoc;
    const int m_binScaleLoc;
    const int m_gtpGammaGammaLoc;

    void setPmv(const glm::mat4& pmv);
    void setBinCount(const GLuint& binCount);
    void setBinScale(const GLfloat& binScale);
    void setGtpGammaGamma(const GLfloat& gtpGammaGamma);

    // Note: m_binCount must be current before this function is called; do setBinCount(..) before getBinVao()
    std::shared_ptr<QOpenGLVertexArrayObject::Binder> getBinVao();

private:
    QPointer<QOpenGLVertexArrayObject> m_binVao;
    QOpenGLBuffer m_binVaoBuff;
    GLuint m_binVaoSize;

    glm::mat4 m_pmv;
    GLuint m_binCount;
    GLfloat m_binScale;
    GLfloat m_gtpGammaGamma;
};
