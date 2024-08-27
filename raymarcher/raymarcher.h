#include "../utils/rgba.h"
#include "scene.h"
#include "../window/mainwindow.h"
#include "hit.h"


class Raymarcher {

    public:

    void render(const Scene& scene, MainWindow& window, RGBA *imageData);


    private:

    RGBA marchRay(const Scene& scene, const RGBA originalColor, Eigen::Vector4f p, Eigen::Vector4f d);
    Hit getClosestHit(const Scene& scene, const Eigen::Vector4f pos);
    float getShapeDistance(const RenderShapeData& shapeData, const Eigen::Vector4f pos);



};