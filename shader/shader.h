#pragma once

#include <tuple>
#include <utils/openglutils.h>
#include <GLFW/glfw3.h>


struct TextureQuad {
    GLuint vao;
    GLuint vbo;
    GLuint vboTexture;
    GLuint ebo;
};

inline GLuint createShader()
{
    const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        void main()
        {
            gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
            TexCoord = aTexCoord;
        }
    )";

    const char* fragmentShaderSource = R"(
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D ourTexture;
        void main() {
            vec4 texColor = texture(ourTexture, TexCoord);
            FragColor = texColor;
    }
)";

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glDetachShader(shaderProgram, vertexShader);
    glDetachShader(shaderProgram, fragmentShader);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    GET_GL_ERROR("Error in createShader()");

    return shaderProgram;
}

inline TextureQuad setupTextureDisplayQuad(float aspectRatio)
{
    TextureQuad quad;

    glGenVertexArrays(1, &quad.vao);
    glGenBuffers(1, &quad.vbo);
    glGenBuffers(1, &quad.vboTexture);
    glGenBuffers(1, &quad.ebo);

    struct float2 { float x; float y; };

    const float minX = -aspectRatio;
    const float minY = -1.0f;
    const float maxX = aspectRatio;
    const float maxY = 1.0f;

    float2 vertices[4];
    vertices[0] = {maxX, maxY};
    vertices[1] = {maxX, minY};
    vertices[2] = {minX, minY};
    vertices[3] = {minX, maxY};

    float2 textCoords[4];
    textCoords[0] = {1.0f, 1.0f};
    textCoords[1] = {1.0f, 0.0f};
    textCoords[2] = {0.0f, 0.0f};
    textCoords[3] = {0.0f, 1.0f};

    
    
    const unsigned indices[] = {
        0, 1, 3,
        1, 2, 3
    };

    glBindVertexArray(quad.vao);
        // Element buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quad.ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        // Vertex positions
        glBindBuffer(GL_ARRAY_BUFFER, quad.vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float2), 0);
        glEnableVertexAttribArray(0);

        // Texture coordinates
        glBindBuffer(GL_ARRAY_BUFFER, quad.vboTexture);
        glBufferData(GL_ARRAY_BUFFER, sizeof(textCoords), textCoords, GL_STATIC_DRAW);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float2), 0);
        glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return quad;
}

inline std::tuple<GLuint, GLuint> createTexture(int width, int height, int channels = 4)
{
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // set up the texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, 
        GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    //pixel buffer object
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * channels, nullptr, GL_DYNAMIC_DRAW);
    
    GLint bsize;
    glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bsize);

    if ((GLuint)bsize != (channels * sizeof(unsigned char) * width * height)) 
    {
        printf("Buffer object (%d) has incorrect size (%d).\n",
                (unsigned)pbo, (unsigned)bsize);
        exit(EXIT_FAILURE);
    }
    
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    return {texture, pbo};
}

