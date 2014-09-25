#include "Common.h"
#include "GlslProg.h"

GlslProg::GlslProg(QObject *parent)
  : QOpenGLShaderProgram(parent),
    m_glfs(nullptr)
{
}

GlslProg::~GlslProg()
{
}

void GlslProg::init(QOpenGLFunctions_4_1_Core* glfs)
{
    m_glfs = glfs;
}

void GlslProg::addShader(const QString& fileName, const QOpenGLShader::ShaderType& type)
{
    if(!addShaderFromSourceFile(type, fileName))
    {
        throw RisWidgetException(std::string("GlslProg::addShader(..): Failed to compile shader \"") +
                                 fileName.toStdString() + "\".");
    }
}
