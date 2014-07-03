#include "Common.h"
#include "ImageDrawProg.h"

ImageDrawProg::ImageDrawProg(QObject* parent)
  : GlslProg(parent),
    m_quadVao(new QOpenGLVertexArrayObject(this)),
    m_quadVaoBuff(QOpenGLBuffer::VertexBuffer),
    m_pmvLoc(std::numeric_limits<int>::min()),
    m_fragToTexLoc(std::numeric_limits<int>::min())
{
    addShader(":/shaders/image.glslv", QOpenGLShader::Vertex);
    addShader(":/shaders/image.glslf", QOpenGLShader::Fragment);
}

ImageDrawProg::~ImageDrawProg()
{
}

void ImageDrawProg::init(QOpenGLFunctions_4_1_Core* glfs)
{
    if(!m_quadVao->create())
    {
        throw RisWidgetException("ImageDrawProg::ImageDrawProg(..): Failed to create m_quadVao.");
    }
    QOpenGLVertexArrayObject::Binder quadVaoBinder(m_quadVao);

    float quad[] = {
        // Vertex positions
        1.0f, -1.0f,
        -1.0f, -1.0f,
        -1.0f, 1.0f,
        1.0f, 1.0f
    };

    m_quadVaoBuff.create();
    m_quadVaoBuff.bind();
    m_quadVaoBuff.setUsagePattern(QOpenGLBuffer::StaticDraw);
    m_quadVaoBuff.allocate(reinterpret_cast<void*>(quad), sizeof(quad));

    glfs->glEnableVertexAttribArray(VertCoordLoc);
    glfs->glVertexAttribPointer(VertCoordLoc, 2, GL_FLOAT, false, 0, nullptr);

    const_cast<int&>(m_pmvLoc) = uniformLocation("projectionModelViewMatrix");
    const_cast<int&>(m_fragToTexLoc) = uniformLocation("fragToTex");
}
