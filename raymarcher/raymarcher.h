#include <memory>
#include <cuda_runtime.h>

#include "../shader/shader.h"
#include "../utils/rgba.cuh"
#include "scene.h"
#include "../window/window.h"
#include "hit.h"

#include <cuda_gl_interop.h>


class Raymarcher {

    public:
        Raymarcher(std::unique_ptr<Window> w);
        ~Raymarcher();
        void run(const Scene& scene);
        void render(const Scene& scene);

        void setDrawCallback(std::function<void()> callbackFunction)
        {
            onDraw = callbackFunction;
        }
        void setShader(GLuint shader)
        {
            shaderProgram = shader;    
        }

        void setQuad(TextureQuad& q) {
            quad = q;
        }

        void setTexture (GLuint t) {
            texture = t;
        }

        void setPbo(GLuint p) {
            pbo = p;
            //last arg says that we intend on overwriting the contexts of the pbo
            cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsMapFlagsWriteDiscard);
            if (err != cudaSuccess) {
                std::cerr << "Failed to register PBO with CUDA: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("CUDA-OpenGL interop registration failed");
            }
        }


    private:
        std::unique_ptr<Window> window;
        std::function<void()> onDraw;
        GLuint texture;
        GLuint shaderProgram;
        GLuint pbo;
        TextureQuad quad;
        cudaGraphicsResource_t cudaPboResource;


        RGBA marchRay(const Scene& scene, const RGBA originalColor, const Eigen::Vector4f& p, const Eigen::Vector4f& d);
        Hit getClosestHit(const Scene& scene, const Eigen::Vector4f& pos);
        float getShapeDistance(const RenderShapeData& shapeData, const Eigen::Vector4f& pos);



};