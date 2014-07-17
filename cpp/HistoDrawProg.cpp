#include "Common.h"
#include "HistoDrawProg.h"

HistoDrawProg::HistoDrawProg(QObject* parent)
  : GlslProg(parent),
    m_pmvLoc(std::numeric_limits<int>::min()),
    m_binCountLoc(std::numeric_limits<int>::min()),
    m_binScaleLoc(std::numeric_limits<int>::min()),
    m_gammaGammaValLoc(std::numeric_limits<int>::min()),
    m_binVao(new QOpenGLVertexArrayObject(this)),
    m_binVaoBuff(QOpenGLBuffer::VertexBuffer),
    m_binVaoSize(std::numeric_limits<GLuint>::max()),
    m_pmv(1.0f),
    m_binCount(std::numeric_limits<GLuint>::max()),
    m_binScale(NAN),
    m_gammaGammaVal(NAN)
{
    // Cause any comparison to fail so that the first setPmv(..) call will assign new value to m_pmv
    m_pmv[0][0] = NAN;
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
    GlslProg::init(glfs);

    const_cast<int&>(m_pmvLoc) = uniformLocation("projectionModelViewMatrix");
    const_cast<int&>(m_binCountLoc) = uniformLocation("binCount");
    const_cast<int&>(m_binScaleLoc) = uniformLocation("binScale");
    const_cast<int&>(m_gammaGammaValLoc) = uniformLocation("gammaGammaVal");

    // For now, the histogram is the only content of the histogram view, and so the histogram occupies the entire view
    // and is not scaled.  Thus, its transformation matrix is the identity matrix, and only needs to be set once, upon
    // initialization, rather than checked/updated-if-different each time Renderer::execHistoDraw() is called.
    setPmv(glm::mat4(1.0f));
}

void HistoDrawProg::setPmv(const glm::mat4& pmv)
{
    if(pmv != m_pmv)
    {
        m_glfs->glUniformMatrix4fv(m_pmvLoc, 1, GL_FALSE, glm::value_ptr(pmv));
        m_pmv = pmv;
    }
}

void HistoDrawProg::setBinCount(const GLuint& binCount)
{
    if(binCount != m_binCount)
    {
        setUniformValue(m_binCountLoc, binCount);
        m_binCount = binCount;
    }
}

void HistoDrawProg::setBinScale(const GLfloat& binScale)
{
    if(binScale != m_binScale)
    {
        setUniformValue(m_binScaleLoc, binScale);
        m_binScale = binScale;
    }
}

void HistoDrawProg::setGammaGammaVal(const GLfloat& gammaGammaVal)
{
    if(gammaGammaVal != m_gammaGammaVal)
    {
        setUniformValue(m_gammaGammaValLoc, gammaGammaVal);
        m_gammaGammaVal = gammaGammaVal;
    }
}

std::shared_ptr<QOpenGLVertexArrayObject::Binder> HistoDrawProg::getBinVao()
{
    if(m_binCount <= 0)
    {
        throw RisWidgetException("HistoDrawProg::getBinVao(): Can not make zero length vertex array object.");
    }
    std::shared_ptr<QOpenGLVertexArrayObject::Binder> ret(new QOpenGLVertexArrayObject::Binder(m_binVao));
    if(m_binVaoSize != m_binCount)
    {
        if(m_binVaoBuff.isCreated())
        {
            m_binVaoBuff.bind();
        }
        else
        {
            m_binVaoBuff.create();
            m_binVaoBuff.bind();
            m_binVaoBuff.setUsagePattern(QOpenGLBuffer::StaticDraw);
        }
        m_binVaoBuff.allocate(m_binCount * sizeof(GLfloat));
        GLfloat* b{reinterpret_cast<GLfloat*>(m_binVaoBuff.map(QOpenGLBuffer::WriteOnly))};
        if(b == nullptr)
        {
            m_binVaoBuff.release();
            throw RisWidgetException("HistoDrawProg::getBinVao(): Failed to map vertex array object buffer.");
        }
        GLfloat* be{b + m_binCount};
        GLfloat binIndex{0};
        for(;;)
        {
            *b = binIndex;
            ++b;
            if(b == be) break;
            ++binIndex;
        }
        m_binVaoBuff.unmap();
        m_glfs->glEnableVertexAttribArray(BinIndexLoc);
        m_glfs->glVertexAttribPointer(BinIndexLoc, 1, GL_FLOAT, GL_FALSE, 0, nullptr);
        m_binVaoBuff.release();
        m_binVaoSize = m_binCount;
    }
    return ret;
}
