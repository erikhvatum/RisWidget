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
    GlProgram();
    virtual ~GlProgram();
    GlProgram(const GlProgram&) = delete;
    GlProgram& operator = (const GlProgram&) = delete;

    const GLuint& id() const;
    operator GLuint () const;
    QGLContext* context();
    void setContext(QGLContext* context_);

    void build();
    void del();

#ifdef Q_OS_WIN
    PFNGLCREATEPROGRAMPROC    glCreateProgram{nullptr};
    PFNGLCREATESHADERPROC     glCreateShader{nullptr};
    PFNGLSHADERSOURCEPROC     glShaderSource{nullptr};
    PFNGLCOMPILESHADERPROC    glCompileShader{nullptr};
    PFNGLGETSHADERIVPROC      glGetShaderiv{nullptr};
    PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog{nullptr};
    PFNGLATTACHSHADERPROC     glAttachShader{nullptr};
    PFNGLLINKPROGRAMPROC      glLinkProgram{nullptr};
    PFNGLGETPROGRAMIVPROC     glGetProgramiv{nullptr};
    PFNGLVALIDATEPROGRAMPROC  glValidateProgram{nullptr};
    PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog{nullptr};
    PFNGLDELETESHADERPROC     glDeleteShader{nullptr};
    PFNGLDETACHSHADERPROC     glDetachShader{nullptr};
    PFNGLDELETEPROGRAMPROC    glDeleteProgram{nullptr};
#endif

protected:
    GLuint m_id{std::numeric_limits<GLuint>::max()};
    QGLContext* m_context{nullptr};

    virtual void getSources(std::vector<QString>& sourceFileNames) = 0;
};

class HistoCalcProg
  : public GlProgram
{
public:
    using GlProgram::GlProgram;

protected:
    virtual void getSources(std::vector<QString>& sourceFileNames);
};

class HistoConsolidateProg
  : public GlProgram
{
public:
    using GlProgram::GlProgram;

protected:
    virtual void getSources(std::vector<QString> &sourceFileNames);
};

class ImageDrawProg
  : public GlProgram
{
public:
    using GlProgram::GlProgram;

protected:
    virtual void getSources(std::vector<QString> &sourceFileNames);
};

class HistoDrawProg
  : public GlProgram
{
public:
    using GlProgram::GlProgram;

protected:
    virtual void getSources(std::vector<QString> &sourceFileNames);
};
