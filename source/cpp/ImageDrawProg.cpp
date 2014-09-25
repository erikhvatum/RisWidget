#include "Common.h"
#include "ImageDrawProg.h"

ImageDrawProg::ImageDrawProg(QObject* parent)
  : GlslProg(parent),
    m_quadVao(new QOpenGLVertexArrayObject(this)),
    m_quadVaoBuff(QOpenGLBuffer::VertexBuffer),
    m_pmvLoc(std::numeric_limits<int>::min()),
    m_fragToTexLoc(std::numeric_limits<int>::min()),
    m_gtpMinLoc(std::numeric_limits<int>::min()),
    m_gtpMaxLoc(std::numeric_limits<int>::min()),
    m_gtpRangeLoc(std::numeric_limits<int>::min()),
    m_gtpGammaLoc(std::numeric_limits<int>::min()),
    m_drawImageLoc(std::numeric_limits<int>::min()),
    m_drawImagePassthroughIdx(std::numeric_limits<GLuint>::max()),
    m_drawImageGammaIdx(std::numeric_limits<GLuint>::max()),
    m_gtpMin(NAN),
    m_gtpMax(NAN),
    m_gtpRange(NAN),
    m_gtpGamma(NAN),
    m_drawImageSubroutineIdx(std::numeric_limits<GLuint>::max())
{
    // Cause any comparison to fail so that the first setPmv(..) call will assign new value to m_pmv
    m_pmv[0][0] = NAN;
    m_fragToTex[0][0] = NAN;
    // Note Qt interprets a path beginning with a colon as a Qt resource bundle identifier.  Such a path refers to an
    // object integrated into this application's binary.
    addShader(":/gpu/image.glslv", QOpenGLShader::Vertex);
    addShader(":/gpu/image.glslf", QOpenGLShader::Fragment);
}

ImageDrawProg::~ImageDrawProg()
{
}

void ImageDrawProg::init(QOpenGLFunctions_4_1_Core* glfs)
{
    GlslProg::init(glfs);

    if(!m_quadVao->create())
    {
        throw RisWidgetException("ImageDrawProg::ImageDrawProg(..): Failed to create m_quadVao.");
    }
    QOpenGLVertexArrayObject::Binder quadVaoBinder(m_quadVao);

    float quad[] = {
        // Vertex positions
        1.1f, -1.1f,
        -1.1f, -1.1f,
        -1.1f, 1.1f,
        1.1f, 1.1f
    };

    m_quadVaoBuff.create();
    m_quadVaoBuff.bind();
    m_quadVaoBuff.setUsagePattern(QOpenGLBuffer::StaticDraw);
    m_quadVaoBuff.allocate(reinterpret_cast<void*>(quad), sizeof(quad));

    glfs->glEnableVertexAttribArray(VertCoordLoc);
    glfs->glVertexAttribPointer(VertCoordLoc, 2, GL_FLOAT, false, 0, nullptr);

    const_cast<int&>(m_pmvLoc) = uniformLocation("projectionModelViewMatrix");
    const_cast<int&>(m_fragToTexLoc) = uniformLocation("fragToTex");
    const_cast<int&>(m_gtpMinLoc) = uniformLocation("gtpMin");
    const_cast<int&>(m_gtpMaxLoc) = uniformLocation("gtpMax");
    const_cast<int&>(m_gtpRangeLoc) = uniformLocation("gtpRange");
    const_cast<int&>(m_gtpGammaLoc) = uniformLocation("gtpGamma");
    const_cast<GLint&>(m_drawImageLoc) =
        m_glfs->glGetSubroutineUniformLocation(programId(), GL_FRAGMENT_SHADER, "drawImage");
    const_cast<GLuint&>(m_drawImagePassthroughIdx) =
        m_glfs->glGetSubroutineIndex(programId(), GL_FRAGMENT_SHADER, "drawImage_passthrough");
    const_cast<GLuint&>(m_drawImageGammaIdx) =
        m_glfs->glGetSubroutineIndex(programId(), GL_FRAGMENT_SHADER, "drawImage_gamma");
}

void ImageDrawProg::setPmv(const glm::mat4& pmv)
{
    if(pmv != m_pmv)
    {
        m_glfs->glUniformMatrix4fv(m_pmvLoc, 1, GL_FALSE, glm::value_ptr(pmv));
        m_pmv = pmv;
    }
}

void ImageDrawProg::setFragToTex(const glm::mat3& fragToTex)
{
    if(fragToTex != m_fragToTex)
    {
        m_glfs->glUniformMatrix3fv(m_fragToTexLoc, 1, GL_FALSE, glm::value_ptr(fragToTex));
        m_fragToTex = fragToTex;
    }
}

const glm::mat3& ImageDrawProg::getFragToTex() const
{
    return m_fragToTex;
}

void ImageDrawProg::setGtpRange(const GLfloat& gtpMin, const GLfloat& gtpMax)
{
    bool changed{false};
    if(gtpMin != m_gtpMin)
    {
        setUniformValue(m_gtpMinLoc, gtpMin);
        m_gtpMin = gtpMin;
        changed = true;
    }
    if(gtpMax != m_gtpMax)
    {
        setUniformValue(m_gtpMaxLoc, gtpMax);
        m_gtpMax = gtpMax;
        changed = true;
    }
    if(changed)
    {
        float gtpRange{m_gtpMax - m_gtpMin};
        if(gtpRange != m_gtpRange)
        {
            setUniformValue(m_gtpRangeLoc, gtpRange);
            m_gtpRange = gtpRange;
        }
    }
}

void ImageDrawProg::setGtpGamma(const GLfloat& gtpGamma)
{
    if(gtpGamma != m_gtpGamma)
    {
        setUniformValue(m_gtpGammaLoc, gtpGamma);
        m_gtpGamma = gtpGamma;
    }
}

void ImageDrawProg::setDrawImageSubroutineIdx(const GLuint& drawImageSubroutineIdx)
{
    // Subroutine uniform values do not persist* through shader program unbind/rebind, so it is prudent to call
    // glUniformSubroutinesuiv even if its target value has not changed.  That is why this function does not make the
    // update contigent on whether m_drawImageSubroutineIdx != drawImageSubroutineIdx, which it would otherwise do, and
    // which would be consistent with the other setXxxx member functions.  Because all subroutines are set in one shot
    // by glUniformSubroutinesuiv, if this shader is extended to use more than one subroutine uniform, more
    // infrastructure will be required: useProgram must be overridden to update all subroutine uniforms, and subroutine
    // uniform setting functions (such as this one) must subroutine index values at specific positions in a subroutine
    // index array that is fed to the glUniformSubroutinesuiv call added to useProgram.
    // 
    // * From the OpenGL 4.1 spec: "When UseProgram is called, the subroutine uniforms for all shader stages are reset
    //   to arbitrarily chosen default functions with compatible subroutine types. When UseShaderProgramEXT is called,
    //   the subroutine uniforms for the shader stage specified by are reset to arbitrarily chosen default functions
    //   with compatible subroutine types."

    m_glfs->glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &drawImageSubroutineIdx);
    m_drawImageSubroutineIdx = drawImageSubroutineIdx;
}
