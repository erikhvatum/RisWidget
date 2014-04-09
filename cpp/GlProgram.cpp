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

GlProgram::GlProgram()
{
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
                    throw RisWidgetException(std::string("GlProgram::build(): Compilation of shader \"") +
                                             sfn.toStdString() + std::string("\" failed:\n") + err.get());
                }
            }
        }
    }
    catch(RisWidgetException& e)
    {
        for(const GLuint& shader : shaders)
        {
            glDetachShader(m_id, shader);
        }
        throw e;
    }

    for(const GLuint& shader : shaders)
    {
        glDeleteShader(shader);
    }
}

void GlProgram::del()
{
    if(m_id != std::numeric_limits<GLuint>::max())
    {
        glDeleteProgram(m_id);
        m_id = std::numeric_limits<GLuint>::max();
    }
}

void HistoCalcProg::getSources(std::vector<QString>& sourceFileNames)
{
    sourceFileNames.emplace_back(":/shaders/histogramCalc.glslc");
}
