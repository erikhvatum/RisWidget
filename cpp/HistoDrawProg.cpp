#include "Common.h"
#include "HistoDrawProg.h"

HistoDrawProg::HistoDrawProg(QObject* parent)
  : GlslProg(parent),
    m_binVao(new QOpenGLVertexArrayObject(this)),
    m_binVaoBuff(new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer)),
    m_pmvLoc(std::numeric_limits<int>::min()),
    m_pmv(1.0f),
    m_binCountLoc(std::numeric_limits<int>::min()),
    m_binCount(std::numeric_limits<GLuint>::max()),
    m_binScaleLoc(std::numeric_limits<int>::min()),
    m_binScale(std::numeric_limits<GLfloat>::max()),
    m_gammaGammaValLoc(std::numeric_limits<int>::min()),
    m_gammaGammaVal(std::numeric_limits<GLfloat>::max())
{
    // Note Qt interprets a path beginning with a colon as a Qt resource bundle identifier.  Such a path refers to an
    // object integrated into this application's binary.
    addShader(":/gpu/histogram.glslv", QOpenGLShader::Vertex);
    addShader(":/gpu/histogram.glslf", QOpenGLShader::Fragment);
}

HistoDrawProg::~HistoDrawProg()
{
}

void HistoDrawProg::init(QOpenGLFunctions_4_1_Core* glfs)
{
    m_binVaoBuff->setUsagePattern(QOpenGLBuffer::StaticDraw);

    const_cast<int&>(m_pmvLoc) = uniformLocation("projectionModelViewMatrix");
    const_cast<int&>(m_binCountLoc) = uniformLocation("binCount");
    const_cast<int&>(m_binScaleLoc) = uniformLocation("binScale");
    const_cast<int&>(m_gammaGammaValLoc) = uniformLocation("gammaGammaVal");
}

void setBinCount(const GLuint& binCount)
{
    if(binCount != m_binCount)
    {

    }
}

void setBinScale(const GLfloat& binScale)
{
}

void setGammaGammaVal(const GLfloat& gammaGammaVal)
{
}

void setPmv(const glm::mat4& pmv)
{
}
