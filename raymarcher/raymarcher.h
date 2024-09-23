#include "../utils/rgba.h"
#include "scene.h"
#include "../window/window.h"
#include "hit.h"
#include <memory>

class Raymarcher {

    public:
        Raymarcher(std::unique_ptr<Window> w);
        void run();
        void render(const Scene& scene, RGBA *imageData);


    private:
        std::unique_ptr<Window> window;

        RGBA marchRay(const Scene& scene, const RGBA originalColor, const Eigen::Vector4f& p, const Eigen::Vector4f& d);
        Hit getClosestHit(const Scene& scene, const Eigen::Vector4f& pos);
        float getShapeDistance(const RenderShapeData& shapeData, const Eigen::Vector4f& pos);



};