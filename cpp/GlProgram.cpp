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
#include "View.h"

GlProgram::GlProgram(const std::string& name_)
  : m_id(std::numeric_limits<GLuint>::max()),
    m_name(name_),
    m_glfs(nullptr)
{
    if(m_name.empty())
    {
        m_name = "(unnamed)";
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

const std::string& GlProgram::name() const
{
    return m_name;
}

void GlProgram::build(QOpenGLFunctions_4_3_Core* glfs)
{
    del();

    std::vector<QString> sourceFileNames;
    getSources(sourceFileNames);
    GLenum type;
    m_glfs = glfs;
    m_id = m_glfs->glCreateProgram();
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

            shader = m_glfs->glCreateShader(type);
            const char* sp{s.constData()};
            GLint sl{static_cast<GLint>(s.size())};
            m_glfs->glShaderSource(shader, 1, &sp, &sl);
            m_glfs->glCompileShader(shader);
            m_glfs->glGetShaderiv(shader, GL_COMPILE_STATUS, &param);
            if(param == GL_FALSE)
            {
                m_glfs->glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &param);
                if(param > 0)
                {
                    std::unique_ptr<char[]> err(new char[param]);
                    m_glfs->glGetShaderInfoLog(shader, param, nullptr, err.get());
                    throw RisWidgetException(std::string("GlProgram::build(): Compilation of shader \"") + sfn.toStdString() +
                                             "\" failed:\n" + err.get());
                }
                else
                {
                    throw RisWidgetException(std::string("GlProgram::build(): Compilation of shader \"") + sfn.toStdString() +
                                             std::string("\" failed: (GL shader info log is empty)."));
                }
            }
            m_glfs->glAttachShader(m_id, shader);
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

        m_glfs->glLinkProgram(m_id);
        m_glfs->glGetProgramiv(m_id, GL_LINK_STATUS, &param);
        if(param == GL_FALSE)
        {
            failed = "Linking";
        }
        else
        {
            m_glfs->glValidateProgram(m_id);
            m_glfs->glGetProgramiv(m_id, GL_VALIDATE_STATUS, &param);
            if(param == GL_FALSE)
            {
                failed = "Validation";
            }
        }

        if(!failed.empty())
        {
            m_glfs->glGetProgramiv(m_id, GL_INFO_LOG_LENGTH, &param);
            if(param > 0)
            {
                std::unique_ptr<char[]> err(new char[param]);
                m_glfs->glGetProgramInfoLog(m_id, param, nullptr, err.get());
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
            m_glfs->glDetachShader(m_id, shader);
            m_glfs->glDeleteShader(shader);
        }
        throw e;
    }

    // Once linked into a program, the shader component "object files" (for lack of a better term) are no longer needed
    for(const GLuint& shader : shaders)
    {
        m_glfs->glDeleteShader(shader);
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
        m_glfs->glDeleteProgram(m_id);
        m_id = std::numeric_limits<GLuint>::max();
        m_glfs = nullptr;
    }
}

GLint GlProgram::getUniLoc(const char* uniName)
{
    GLint ret{m_glfs->glGetUniformLocation(m_id, uniName)};
    if(ret < 0)
    {
        throw RisWidgetException(std::string("GlProgram: Failed to get location of uniform \"") + uniName + "\" in \"" + m_name + "\".");
    }
    return ret;
}

GLint GlProgram::getSubUniLoc(const GLenum& type, const char* subUniName)
{
    GLint ret{m_glfs->glGetSubroutineUniformLocation(m_id, type, subUniName)};
    if(ret < 0)
    {
        throw RisWidgetException(std::string("GlProgram: Failed to get location of subroutine uniform \"") + subUniName + "\" in \"" +
                                 m_name + "\".");
    }
    return ret;
}

GLuint GlProgram::getSubIdx(const GLenum& type, const char* subName)
{
    GLuint ret{m_glfs->glGetSubroutineIndex(m_id, type, subName)};
    if(ret == GL_INVALID_INDEX)
    {
        throw RisWidgetException(std::string("GlProgram: Failed to get location of subroutine index \"") + subName + "\" in \"" +
                                 m_name + "\".");
    }
    return ret;
}

GLint GlProgram::getAttrLoc(const char* attrName)
{
    GLint ret{m_glfs->glGetAttribLocation(m_id, attrName)};
    if(ret < 0)
    {
        throw RisWidgetException(std::string("GlProgram: Failed to get location of attribute \"") + attrName + "\" in \"" +
                                 m_name + "\".");
    }
    return ret;
}

HistoCalcProg::HistoCalcProg(const std::string& name_)
  : GlProgram(name_),
    binCountLoc(std::numeric_limits<GLint>::min()),
    invocationRegionSizeLoc(std::numeric_limits<GLint>::min()),
    imageLoc(0),
    blocksLoc(1),
    wgCountPerAxis(8),
    liCountPerAxis(4)
{
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

HistoConsolidateProg::HistoConsolidateProg(const std::string& name_)
  : GlProgram(name_),
    binCountLoc(std::numeric_limits<GLint>::min()),
    invocationBinCountLoc(std::numeric_limits<GLint>::min()),
    blocksLoc(0),
    histogramLoc(1),
    extremaLoc(0),
    liCount(16),
    extremaBuff(std::numeric_limits<GLuint>::max()),
    extrema{std::numeric_limits<std::uint32_t>::max(),
            std::numeric_limits<std::uint32_t>::min()}
{
}

void HistoConsolidateProg::getSources(std::vector<QString> &sourceFileNames)
{
    sourceFileNames.emplace_back(":/shaders/histogramConsolidate.glslc");
}

void HistoConsolidateProg::postBuild()
{
    binCountLoc = getUniLoc("binCount");
    invocationBinCountLoc = getUniLoc("invocationBinCount");

    m_glfs->glUseProgram(m_id);
    m_glfs->glGenBuffers(1, &extremaBuff);
    m_glfs->glBindBuffer(GL_SHADER_STORAGE_BUFFER, extremaBuff);
    m_glfs->glBufferData(GL_SHADER_STORAGE_BUFFER, 2 * 4, nullptr, GL_DYNAMIC_COPY);
}

ImageDrawProg::ImageDrawProg(const std::string& name_)
  : GlProgram(name_),
    panelColorerLoc(std::numeric_limits<GLint>::min()),
    imagePanelGammaTransformColorerIdx(std::numeric_limits<GLuint>::max()),
    imagePanelGammaTransformColorerHighlightIdx(std::numeric_limits<GLuint>::max()),
    imagePanelPassthroughColorerIdx(std::numeric_limits<GLuint>::max()),
    imagePanelPassthroughColorerHighlightIdx(std::numeric_limits<GLuint>::max()),
    gtpEnabled(true),
    gtpMin(0),
    gtpMax(65535),
    gtpGamma(1.0f),
    gtpMinLoc(std::numeric_limits<GLint>::min()),
    gtpMaxLoc(std::numeric_limits<GLint>::min()),
    gtpGammaLoc(std::numeric_limits<GLint>::min()),
    projectionModelViewMatrixLoc(std::numeric_limits<GLint>::min()),
    quadVaoBuff(std::numeric_limits<GLuint>::max()),
    quadVao(std::numeric_limits<GLuint>::max()),
    vertPosLoc(std::numeric_limits<GLint>::min()),
    texCoordLoc(std::numeric_limits<GLint>::min()),
    highlightCoordsLoc(0),
    highlightCoordsBuff(std::numeric_limits<GLuint>::max()),
    wantedHighlightCoord(std::numeric_limits<GLfloat>::lowest(), std::numeric_limits<GLfloat>::lowest()),
    actualHighlightCoord(std::numeric_limits<GLfloat>::lowest(), std::numeric_limits<GLfloat>::lowest())
{
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
    imagePanelGammaTransformColorerHighlightIdx = getSubIdx(GL_FRAGMENT_SHADER, "imagePanelGammaTransformColorerHighlight");
    imagePanelPassthroughColorerIdx = getSubIdx(GL_FRAGMENT_SHADER, "imagePanelPassthroughColorer");
    imagePanelPassthroughColorerHighlightIdx = getSubIdx(GL_FRAGMENT_SHADER, "imagePanelPassthroughColorerHighlight");

    gtpMinLoc = getUniLoc("gtp.minVal");
    gtpMaxLoc = getUniLoc("gtp.maxVal");
    gtpGammaLoc = getUniLoc("gtp.gammaVal");
    projectionModelViewMatrixLoc = getUniLoc("projectionModelViewMatrix");

    vertPosLoc = getAttrLoc("vertPos");
    texCoordLoc = getAttrLoc("texCoord");

    float quad[] = {
        // Vertex positions
        1.0f, -1.0f,
        -1.0f, -1.0f,
        -1.0f, 1.0f,
        1.0f, 1.0f,
       // Texture coordinates
        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f
//      1.0f, 1.0f,
//      0.0f, 1.0f,
//      0.0f, 0.0f,
//      1.0f, 0.0f
    };

    m_glfs->glUseProgram(m_id);
    m_glfs->glGenVertexArrays(1, &quadVao);
    m_glfs->glBindVertexArray(quadVao);
    m_glfs->glGenBuffers(1, &quadVaoBuff);
    m_glfs->glBindBuffer(GL_ARRAY_BUFFER, quadVaoBuff);
    m_glfs->glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    
    m_glfs->glEnableVertexAttribArray(vertPosLoc);
    m_glfs->glVertexAttribPointer(vertPosLoc, 2, GL_FLOAT, false, 0, nullptr);

    m_glfs->glEnableVertexAttribArray(texCoordLoc);
    m_glfs->glVertexAttribPointer(texCoordLoc, 2, GL_FLOAT, false, 0, reinterpret_cast<void*>(2 * 4 * 4));

    m_glfs->glGenBuffers(1, &highlightCoordsBuff);
    m_glfs->glBindBuffer(GL_SHADER_STORAGE_BUFFER, highlightCoordsBuff);
    m_glfs->glBufferData(GL_SHADER_STORAGE_BUFFER, 2 * 2 * 4, nullptr, GL_DYNAMIC_COPY);
}

HistoDrawProg::HistoDrawProg(const std::string& name_)
  : GlProgram(name_),
    projectionModelViewMatrixLoc(std::numeric_limits<GLint>::min()),
    binCountLoc(std::numeric_limits<GLint>::min()),
    binScaleLoc(std::numeric_limits<GLint>::min()),
    gammaGamma(1.0f),
    gammaGammaLoc(std::numeric_limits<GLint>::min()),
    histogramLoc(0),
    pointVaoBuff(std::numeric_limits<GLuint>::max()),
    pointVao(std::numeric_limits<GLuint>::max()),
    binIndexLoc(std::numeric_limits<GLint>::min())
{
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
    projectionModelViewMatrixLoc = getUniLoc("projectionModelViewMatrix");
    gammaGammaLoc = getUniLoc("gammaGammaVal");

    binIndexLoc = getAttrLoc("binIndex");
}

