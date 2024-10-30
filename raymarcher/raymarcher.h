#include <memory>
#include <cuda_runtime.h>

#include "../shader/shader.h"
#include "../utils/rgba.cuh"
#include "scene.h"
#include "../window/window.h"
#include "../kernel/cudautils.cuh"
#include "hit.h"

#include <cuda_gl_interop.h>


class Raymarcher {

    public:
        Raymarcher(std::unique_ptr<Window> w, const Scene& s);
        ~Raymarcher();
        void run();
        void render();

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
            cudaError_t err = gpuErrorCheck( cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsMapFlagsWriteDiscard) );
         
        }


    private:
        const Scene scene;
        std::unique_ptr<Window> window;
        std::function<void()> onDraw;
        GLuint texture;
        GLuint shaderProgram;
        GLuint pbo;
        TextureQuad quad;

        //device resources
        cudaGraphicsResource_t cudaPboResource;
        RGBA* deviceImageData;
        GPUShape* deviceShapes;
        mat4* deviceInverseViewMat;
        int* deviceWidth;
        int* deviceHeight;
        int* deviceNumPrimitives;
        float* deviceViewPlaneWidth;
        float* deviceViewPlaneHeight;


        RGBA marchRay(const Scene& scene, const RGBA originalColor, const Eigen::Vector4f& p, const Eigen::Vector4f& d);
        Hit getClosestHit(const Scene& scene, const Eigen::Vector4f& pos);
        float getShapeDistance(const RenderShapeData& shapeData, const Eigen::Vector4f& pos);



};