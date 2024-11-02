#include <memory>
#include <cuda_runtime.h>

#include "../shader/shader.h"
#include "../utils/rgba.cuh"
#include "scene.h"
#include "../window/window.h"
#include "../kernel/cudautils.cuh"
#include "../kernel/renderdata.cuh"
#include "../kernel/shape.cuh"

#include <cuda_gl_interop.h>


class Raymarcher {

    public:
        Raymarcher(std::unique_ptr<Window> w, const Scene& s, GLuint p);
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
        GPURenderData* deviceRenderData;
        GPUSceneLightData* deviceLights;
        GPURenderShapeData* deviceShapes;
        mat4* deviceInverseViewMat;
        int* deviceWidth;
        int* deviceHeight;
        float* deviceViewPlaneWidth;
        float* deviceViewPlaneHeight;


        void allocateDeviceRenderData();
       

};