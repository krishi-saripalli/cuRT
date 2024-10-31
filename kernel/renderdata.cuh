#ifndef RENDER_DATA_CUH
#define RENDER_DATA_CUH
#define EIGEN_NO_CUDA

#include "vec3.cuh"
#include "vec4.cuh"
#include "shape.cuh"
#include "../utils/scenedata.h"

struct GPUSceneGlobalData {
    float ka; 
    float kd;
    float ks;
    float kt;

    GPUSceneGlobalData() : ka(0), kd(0), ks(0), kt(0) {}
    explicit GPUSceneGlobalData(const SceneGlobalData& data) : ka(data.ka), kd(data.kd), ks(data.ks), kt(data.kt) {}
};

struct GPUSceneCameraData {
    vec4 pos;
    vec4 look;
    vec4 up;
    float heightAngle;
    float aperture;
    float focalLength;

    // Add default constructor
    GPUSceneCameraData() : 
        pos(), look(), up(),
        heightAngle(0),
        aperture(0),
        focalLength(0) 
    {}
};

enum class GPULightType {
    LIGHT_POINT,
    LIGHT_DIRECTIONAL,
    LIGHT_SPOT,
};

struct GPUSceneLightData {
    int id;
    GPULightType type;
    vec4 color;
    vec3 function;
    vec4 pos;
    vec4 dir;
    float penumbra;
    float angle;

    GPUSceneLightData() :
        id(0),
        type(GPULightType::LIGHT_POINT),
        color(), function(), pos(), dir(),
        penumbra(0),
        angle(0)
    {}
};

struct GPURenderData {
    GPUSceneGlobalData globalData;
    GPUSceneCameraData cameraData;
    GPUSceneLightData* lights;
    GPURenderShapeData* shapes;
    int numLights;
    int numShapes;

    GPURenderData() :
        globalData(), 
        cameraData(), 
        lights(nullptr),
        shapes(nullptr),
        numLights(0),
        numShapes(0)
    {}
};

#endif