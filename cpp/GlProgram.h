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

class View;

class GlProgram
{
public:
    explicit GlProgram(const std::string& name_);
    virtual ~GlProgram();
    // The destructor calls glDeleteProgram, so a copy constructor would have to do a deep copy or keep a reference
    // count for m_id.  Both are problematic in terms of unintended side effects, and copying a fully linked shader
    // program is not something routinely required, so copying is forbidden in order that any attempt to do so,
    // intentional or unintentional, will hopefully result in this very comment being read.
    GlProgram(const GlProgram&) = delete;
    GlProgram& operator = (const GlProgram&) = delete;

    const GLuint& id() const;
    operator GLuint () const;
    View* view();
    void setView(View* view);
    const std::string& name() const;

    void build();
    void del();

protected:
    GLuint m_id{std::numeric_limits<GLuint>::max()};
    View* m_view;
    std::string m_name;
    // A copy of m_view->m_glfs
    QOpenGLFunctions_4_3_Core* m_glfs{nullptr};

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

    GLint binCountLoc{std::numeric_limits<GLint>::min()};
    GLint invocationRegionSizeLoc{std::numeric_limits<GLint>::min()};
    const GLint imageLoc{0};
    const GLint blocksLoc{1};
    const GLuint wgCountPerAxis{8};
    // This value must match local_size_x and local_size_y in histogramCalc.glslc
    const GLuint liCountPerAxis{4};

protected:
    virtual void getSources(std::vector<QString>& sourceFileNames);
    virtual void postBuild();
};

class HistoConsolidateProg
  : public GlProgram
{
public:
    explicit HistoConsolidateProg(const std::string& name_);

    GLint binCountLoc{std::numeric_limits<GLint>::min()};
    GLint invocationBinCountLoc{std::numeric_limits<GLint>::min()};
    const GLint blocksLoc{0};
    const GLint histogramLoc{1};
    const GLint extremaLoc{0};
    // This value must match local_size_x in histogramConsolidate.glslc
    const GLuint liCount{16};
    GLuint extremaBuff{std::numeric_limits<GLuint>::max()};
    std::uint32_t extrema[2]{std::numeric_limits<std::uint32_t>::max(),
                             std::numeric_limits<std::uint32_t>::min()};

protected:
    virtual void getSources(std::vector<QString> &sourceFileNames);
    virtual void postBuild();
};

class ImageDrawProg
  : public GlProgram
{
public:
    explicit ImageDrawProg(const std::string& name_);

    GLint panelColorerLoc{std::numeric_limits<GLint>::min()};
    GLuint imagePanelGammaTransformColorerIdx{std::numeric_limits<GLuint>::max()};
    GLuint imagePanelPassthroughColorerIdx{std::numeric_limits<GLuint>::max()};

    bool gtpEnabled{false};
    GLint gtpMinLoc{std::numeric_limits<GLint>::min()};
    GLint gtpMaxLoc{std::numeric_limits<GLint>::min()};
    GLint gtpGammaLoc{std::numeric_limits<GLint>::min()};
    GLint projectionModelViewMatrixLoc{std::numeric_limits<GLint>::min()};

    GLuint quadVaoBuff{std::numeric_limits<GLuint>::max()};
    GLuint quadVao{std::numeric_limits<GLuint>::max()};

    GLint vertPosLoc{std::numeric_limits<GLint>::min()};
    GLint texCoordLoc{std::numeric_limits<GLint>::min()};

protected:
    virtual void getSources(std::vector<QString> &sourceFileNames);
    virtual void postBuild();
};

class HistoDrawProg
  : public GlProgram
{
public:
    explicit HistoDrawProg(const std::string& name_);

    GLint projectionModelViewMatrixLoc{std::numeric_limits<GLint>::min()};
    GLint binCountLoc{std::numeric_limits<GLint>::min()};
    GLint binScaleLoc{std::numeric_limits<GLint>::min()};
    GLint histogramLoc{std::numeric_limits<GLint>::min()};

    GLuint pointVaoBuff{std::numeric_limits<GLuint>::max()};
    GLuint pointVao{std::numeric_limits<GLuint>::max()};

    GLint binIndexLoc{std::numeric_limits<GLint>::min()};

protected:
    virtual void getSources(std::vector<QString> &sourceFileNames);
    virtual void postBuild();
};
