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

#include "Common.h"
#include "GlProgram.h"

GlProgram::GlProgram(const std::string& name_)
  : m_name(name_)
{
    if(m_name.empty())
    {
        m_name = "(unnamed)";
    }
}

GlProgram::~GlProgram()
{
    if(m_context)
    {
        m_context->makeCurrent();
        del();
    }
}

const GLuint& GlProgram::id() const
{
    return m_id;
}

GlProgram::operator GLuint () const
{
    return m_id;
}

QGLContext* GlProgram::context()
{
    return m_context;
}

void GlProgram::setContext(QGLContext* context_)
{
    m_context = context_;
#ifdef Q_OS_WIN
    PFNGLCREATEPROGRAMPROC    glCreateProgram   = reinterpret_cast<PFNGLCREATEPROGRAMPROC>(m_context->getProcAddress("glCreateProgram"));
    PFNGLCREATESHADERPROC     glCreateShader    = reinterpret_cast<PFNGLCREATESHADERPROC>(m_context->getProcAddress("glCreateShader"));
    PFNGLSHADERSOURCEPROC     glShaderSource    = reinterpret_cast<PFNGLSHADERSOURCEPROC>(m_context->getProcAddress("glShaderSource"));
    PFNGLCOMPILESHADERPROC    glCompileShader   = reinterpret_cast<PFNGLCOMPILESHADERPROC>(m_context->getProcAddress("glCompileShader"));
    PFNGLGETSHADERIVPROC      glGetShaderiv     = reinterpret_cast<PFNGLGETSHADERIVPROC>(m_context->getProcAddress("glGetShaderiv"));
    PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog= reinterpret_cast<PFNGLGETSHADERINFOLOGPROC>(m_context->getProcAddress("glGetShaderInfoLog"));
    PFNGLATTACHSHADERPROC     glAttachShader    = reinterpret_cast<PFNGLATTACHSHADERPROC>(m_context->getProcAddress("glAttachShader"));
    PFNGLLINKPROGRAMPROC      glLinkProgram     = reinterpret_cast<PFNGLLINKPROGRAMPROC>(m_context->getProcAddress("glLinkProgram"));
    PFNGLGETPROGRAMIVPROC     glGetProgramiv    = reinterpret_cast<PFNGLGETPROGRAMIVPROC>(m_context->getProcAddress("glGetProgramiv"));
    PFNGLVALIDATEPROGRAMPROC  glValidateProgram = reinterpret_cast<PFNGLVALIDATEPROGRAMPROC>(m_context->getProcAddress("glValidateProgram"));
    PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog = reinterpret_cast<PFNGLGETPROGRAMINFOLOGPROC>(m_context->getProcAddress("glGetProgramInfoLog"));
    PFNGLDELETESHADERPROC     glDeleteShader    = reinterpret_cast<PFNGLDELETESHADERPROC>(m_context->getProcAddress("glDeleteShader"));
    PFNGLDETACHSHADERPROC     glDetachShader    = reinterpret_cast<PFNGLDETACHSHADERPROC>(m_context->getProcAddress("glDetachShader"));
    PFNGLDELETEPROGRAMPROC    glDeleteProgram   = reinterpret_cast<PFNGLDELETEPROGRAMPROC>(m_context->getProcAddress("glDeleteProgram"));
#endif
}

const std::string& GlProgram::name() const
{
    return m_name;
}

void GlProgram::build()
{
    del();

    std::vector<QString> sourceFileNames;
    getSources(sourceFileNames);
    GLenum type;
    m_id = glCreateProgram();
    std::list<GLuint> shaders;

    try
    {
        QString src;
        GLuint shader;
        GLint param;

        /* Compile shaders */

        for(const QString& sfn : sourceFileNames)
        {
            if(sfn.endsWith(".glslc", Qt::CaseInsensitive))
            {
                type = GL_COMPUTE_SHADER;
            }
            else if(sfn.endsWith(".glslf", Qt::CaseInsensitive))
            {
                type = GL_FRAGMENT_SHADER;
            }
            else if(sfn.endsWith(".glslv", Qt::CaseInsensitive))
            {
                type = GL_VERTEX_SHADER;
            }
            else if(sfn.endsWith(".glslg", Qt::CaseInsensitive))
            {
                type = GL_GEOMETRY_SHADER;
            }
            else if(sfn.endsWith(".glsltc", Qt::CaseInsensitive))
            {
                type = GL_TESS_CONTROL_SHADER;
            }
            else if(sfn.endsWith(".glslte", Qt::CaseInsensitive))
            {
                type = GL_TESS_EVALUATION_SHADER;
            }
            else
            {
                throw RisWidgetException(std::string("GlProgram::build(): Shader source file \"") +
                                         sfn.toStdString() + "\" does not end with a recognized extension.");
            }

            QFile sf(sfn);
            if(!sf.open(QIODevice::ReadOnly | QIODevice::Text))
            {
                throw RisWidgetException(std::string("GlProgram::build(): Failed to open shader source file \"") +
                                         sfn.toStdString() + "\".");
            }
            QByteArray s{sf.readAll()};
            if(s.isEmpty())
            {
                throw RisWidgetException(std::string("GlProgram::build(): Failed to read any data from source file \"") +
                                         sfn.toStdString() + "\".  Is it a zero byte file?  If so, it probably shouldn't be.");
            }

            shader = glCreateShader(type);
            const char* sp{s.constData()};
            GLint sl{static_cast<GLint>(s.size())};
            glShaderSource(shader, 1, &sp, &sl);
            glCompileShader(shader);
            glGetShaderiv(shader, GL_COMPILE_STATUS, &param);
            if(param == GL_FALSE)
            {
                glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &param);
                if(param > 0)
                {
                    std::unique_ptr<char[]> err(new char[param]);
                    glGetShaderInfoLog(shader, param, nullptr, err.get());
                    throw RisWidgetException(std::string("GlProgram::build(): Compilation of shader \"") + sfn.toStdString() +
                                             "\" failed:\n" + err.get());
                }
                else
                {
                    throw RisWidgetException(std::string("GlProgram::build(): Compilation of shader \"") + sfn.toStdString() +
                                             std::string("\" failed: (GL shader info log is empty)."));
                }
            }
            glAttachShader(m_id, shader);
        }

        /* Link shaders and validate resulting program */

        auto listsrcs = [&](){
            bool first{true};
            std::string ret;
            for(const QString& sfn : sourceFileNames)
            {
                if(first)
                {
                    first = false;
                }
                else
                {
                    ret += ", ";
                }
                ret += '"';
                ret += sfn.toStdString();
                ret += '"';
            }
            return ret;
        };

        std::string failed;

        glLinkProgram(m_id);
        glGetProgramiv(m_id, GL_LINK_STATUS, &param);
        if(param == GL_FALSE)
        {
            failed = "Linking";
        }
        else
        {
            glValidateProgram(m_id);
            glGetProgramiv(m_id, GL_VALIDATE_STATUS, &param);
            if(param == GL_FALSE)
            {
                failed = "Validation";
            }
        }

        if(!failed.empty())
        {
            glGetProgramiv(m_id, GL_INFO_LOG_LENGTH, &param);
            if(param > 0)
            {
                std::unique_ptr<char[]> err(new char[param]);
                glGetProgramInfoLog(m_id, param, nullptr, err.get());
                throw RisWidgetException(std::string("GlProgram::build(): ") + failed + " of shader program \"" + m_name +
                                         "\" with source files " + listsrcs() + " failed: " + err.get());
            }
            else
            {
                throw RisWidgetException(std::string("GlProgram::build(): ") + failed + " of shader program \"" + m_name +
                                         "\" with source files " + listsrcs() + " failed: (GL program info log is empty)");
            }
        }
    }
    catch(RisWidgetException& e)
    {
        for(const GLuint& shader : shaders)
        {
            glDetachShader(m_id, shader);
            glDeleteShader(shader);
        }
        throw e;
    }

    // Once linked into a program, the shader component "object files" (for lack of a better term) are no longer needed
    for(const GLuint& shader : shaders)
    {
        glDeleteShader(shader);
    }

    postBuild();
}

void GlProgram::postBuild()
{
}

void GlProgram::del()
{
    if(m_id != std::numeric_limits<GLuint>::max())
    {
        glDeleteProgram(m_id);
        m_id = std::numeric_limits<GLuint>::max();
    }
}

GLint GlProgram::getUniLoc(const char* uniName)
{
    GLint ret{glGetUniformLocation(m_id, uniName)};
    if(ret < 0)
    {
        throw RisWidgetException(std::string("GlProgram: Failed to get location of uniform \"") + uniName + "\" in \"" + m_name + "\".");
    }
    return ret;
}

GLint GlProgram::getSubUniLoc(const GLenum& type, const char* subUniName)
{
    GLint ret{glGetSubroutineUniformLocation(m_id, type, subUniName)};
    if(ret < 0)
    {
        throw RisWidgetException(std::string("GlProgram: Failed to get location of subroutine uniform \"") + subUniName + "\" in \"" +
                                 m_name + "\".");
    }
    return ret;
}

GLuint GlProgram::getSubIdx(const GLenum& type, const char* subName)
{
    GLuint ret{glGetSubroutineIndex(m_id, type, subName)};
    if(ret == GL_INVALID_INDEX)
    {
        throw RisWidgetException(std::string("GlProgram: Failed to get location of subroutine index \"") + subName + "\" in \"" +
                                 m_name + "\".");
    }
    return ret;
}

GLint GlProgram::getAttrLoc(const char* attrName)
{
    GLint ret{glGetAttribLocation(m_id, attrName)};
    if(ret < 0)
    {
        throw RisWidgetException(std::string("GlProgram: Failed to get location of attribute \"") + attrName + "\" in \"" +
                                 m_name + "\".");
    }
    return ret;
}

void HistoCalcProg::getSources(std::vector<QString>& sourceFileNames)
{
    // Note that a colon prepended to a filename opened by a Qt object refers to a path in the Qt resource bundle built
    // into a program/library's binary
    sourceFileNames.emplace_back(":/shaders/histogramCalc.glslc");
}

void HistoCalcProg::postBuild()
{
    binCountLoc = getUniLoc("binCount");
    invocationRegionSizeLoc = getUniLoc("invocationRegionSize");
}

void HistoConsolidateProg::getSources(std::vector<QString> &sourceFileNames)
{
    sourceFileNames.emplace_back(":/shaders/histogramConsolidate.glslc");
}

void HistoConsolidateProg::postBuild()
{
    binCountLoc = getUniLoc("binCount");
    invocationBinCountLoc = getUniLoc("invocationBinCount");
}

void ImageDrawProg::getSources(std::vector<QString> &sourceFileNames)
{
    sourceFileNames.emplace_back(":/shaders/image.glslv");
    sourceFileNames.emplace_back(":/shaders/image.glslf");
}

void ImageDrawProg::postBuild()
{
    panelColorerLoc = getSubUniLoc(GL_FRAGMENT_SHADER, "panelColorer");
    imagePanelGammaTransformColorerIdx = getSubIdx(GL_FRAGMENT_SHADER, "imagePanelGammaTransformColorer");
    imagePanelPassthroughColorerIdx = getSubIdx(GL_FRAGMENT_SHADER, "imagePanelPassthroughColorer");

    gtpMinLoc = getUniLoc("gtp.minVal");
    gtpMaxLoc = getUniLoc("gtp.maxVal");
    gtpGammaLoc = getUniLoc("gtp.gammaVal");
    projectionModelViewMatrixLoc = getUniLoc("projectionModelViewMatrix");

    vertPosLoc = getAttrLoc("vertPos");
    texCoordLoc = getAttrLoc("texCoord");
}

void HistoDrawProg::getSources(std::vector<QString> &sourceFileNames)
{
    sourceFileNames.emplace_back(":/shaders/histogram.glslv");
    sourceFileNames.emplace_back(":/shaders/histogram.glslf");
}

void HistoDrawProg::postBuild()
{
    binCountLoc = getUniLoc("binCount");
    binScaleLoc = getUniLoc("binScale");

    binIndexLoc = getAttrLoc("binIndex");
}

