#include "scene.h"
#include "utils/sceneparser.h"

Scene::Scene(int width, int height, const RenderData &renderData) {
    c_width = width;
    c_height = height;
    metaData = renderData;
    cameraData = renderData.cameraData;
    globalData = renderData.globalData;
    
}

const int& Scene::width() const {
    return c_width;

}

const int& Scene::height() const {
    return c_height;

}

const SceneGlobalData& Scene::getGlobalData() const {
    return globalData;

}

const Camera Scene::getCamera() const {
    
    return Camera{cameraData};

}