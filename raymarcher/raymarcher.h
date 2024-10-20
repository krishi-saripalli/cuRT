#include "../shader/shader.h"
#include "../utils/rgba.cuh"
#include "scene.h"
#include "../window/window.h"
#include "hit.h"
#include <memory>

class Raymarcher {

    public:
        Raymarcher(std::unique_ptr<Window> w);
        void run();
        void render(const Scene& scene, RGBA *imageData);

        void setDrawCallback(std::function<void()> callbackFunction)
        {
            onDraw = callbackFunction;
        }
        void setShader(GLuint shader)
        {
            shaderProgram = shader;    
        }

        void setTextureQuad(TextureQuad& q) {
            quad = q;
        }


    private:
        std::unique_ptr<Window> window;
        std::function<void()> onDraw;
        GLuint shaderProgram;
        TextureQuad quad;

        RGBA marchRay(const Scene& scene, const RGBA originalColor, const Eigen::Vector4f& p, const Eigen::Vector4f& d);
        Hit getClosestHit(const Scene& scene, const Eigen::Vector4f& pos);
        float getShapeDistance(const RenderShapeData& shapeData, const Eigen::Vector4f& pos);



};