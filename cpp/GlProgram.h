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

class GlProgram
{
public:
    explicit GlProgram(const std::string& name_);
    // The destructor calls glDeleteProgram, so a copy constructor would have to do a deep copy or keep a reference
    // count for m_id.  Both are problematic in terms of unintended side effects, and copying a fully linked shader
    // program is not something routinely required, so copying is forbidden in order that any attempt to do so,
    // intentional or unintentional, will hopefully result in this very comment being read.
    GlProgram(const GlProgram&) = delete;
    GlProgram& operator = (const GlProgram&) = delete;

    const GLuint& id() const;
    operator GLuint () const;
    const std::string& name() const;

    void build(QOpenGLFunctions_4_3_Core* glfs);
    virtual void del();

protected:
    GLuint m_id;
    std::string m_name;
    // A copy of m_view->m_glfs
    QOpenGLFunctions_4_3_Core* m_glfs;

    virtual void getSources(std::vector<QString>& sourceFileNames) = 0;
    // GlProgram provides a no-op implementation rather than making this pure virtual as a convenience for derived
    // classes that do not require the postBuild hook.  Additionally, this means that there is no need for postBuild
    // overrides in derived classes to call GlProgram::postBuild().
    virtual void postBuild();

    GLint getUniLoc(const char* uniName);
    GLint getSubUniLoc(const GLenum& type, const char* subUniName);
    GLuint getSubIdx(const GLenum& type, const char* subName);
    GLint getAttrLoc(const char* attrName);
};

class HistoCalcProg
  : public GlProgram
{
public:
    explicit HistoCalcProg(const std::string& name_);

    GLint binCountLoc;
    GLint invocationRegionSizeLoc;
    const GLint imageLoc;
    const GLint blocksLoc;
    const GLuint wgCountPerAxis;
    // This value must match local_size_x and local_size_y in histogramCalc.glslc
    const GLuint liCountPerAxis;

protected:
    virtual void getSources(std::vector<QString>& sourceFileNames);
    virtual void postBuild();
};

class HistoConsolidateProg
  : public GlProgram
{
public:
    explicit HistoConsolidateProg(const std::string& name_);

    GLint binCountLoc;
    GLint invocationBinCountLoc;
    const GLint blocksLoc;
    const GLint histogramLoc;
    const GLint extremaLoc;
    // This value must match local_size_x in histogramConsolidate.glslc
    const GLuint liCount;
    GLuint extremaBuff;
    std::uint32_t extrema[2];

protected:
    virtual void getSources(std::vector<QString> &sourceFileNames);
    virtual void postBuild();
};

class ImageDrawProg
  : public GlProgram
{
public:
    explicit ImageDrawProg(const std::string& name_);

    GLint panelColorerLoc;
    GLuint imagePanelGammaTransformColorerIdx;
    GLuint imagePanelGammaTransformColorerHighlightIdx;
    GLuint imagePanelPassthroughColorerIdx;
    GLuint imagePanelPassthroughColorerHighlightIdx;

    bool gtpEnabled;
    GLushort gtpMin;
    GLushort gtpMax;
    GLfloat gtpGamma;
    GLint gtpMinLoc;
    GLint gtpMaxLoc;
    GLint gtpGammaLoc;
    GLint projectionModelViewMatrixLoc;

    GLuint quadVaoBuff;
    GLuint quadVao;

    GLint vertPosLoc;
    GLint texCoordLoc;

    GLint highlightCoordsLoc;
    GLuint highlightCoordsBuff;
    glm::vec2 wantedHighlightCoord;
    glm::vec2 actualHighlightCoord;

protected:
    virtual void getSources(std::vector<QString> &sourceFileNames);
    virtual void postBuild();
};

class HistoDrawProg
  : public GlProgram
{
public:
    explicit HistoDrawProg(const std::string& name_);

    GLint projectionModelViewMatrixLoc;
    GLint binCountLoc;
    GLint binScaleLoc;
    GLfloat gammaGamma;
    GLint gammaGammaLoc;
    const GLint histogramLoc;

    GLuint pointVaoBuff;
    GLuint pointVao;

    GLint binIndexLoc;

protected:
    virtual void getSources(std::vector<QString> &sourceFileNames);
    virtual void postBuild();
};
