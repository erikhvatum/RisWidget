#include "common.h"
#include "GlslProg.h"

GlslProg::GlslProg(QObject *parent)
  : QOpenGLShaderProgram(parent)
{
}

GlslProg::~GlslProg()
{
}

void GlslProg::init(QOpenGLFunctions_3_3_Core*)
{
}

void GlslProg::addShader(const QString& fileName, const QOpenGLShader::ShaderType& type)
{
    if(!addShaderFromSourceFile(type, fileName))
    {
        throw RisWidgetException(std::string("GlslProg::addShader(..): Failed to compile shader \"") +
                                 fileName.toStdString() + "\".");
    }
}
