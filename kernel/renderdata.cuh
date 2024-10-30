#ifndef RENDER_DATA_CUH
#define RENDER_DATA_CUH

struct GPURenderData {
    GPUSceneGlobalData globalData;
    GPUSceneCameraData cameraData;

    GPUSceneLightData* lights;
    GPURenderShapeData* shapes;

    int numLights;
    int numShapes;
};

struct GPUSceneGlobalData {
    float ka; 
    float kd;
    float ks;
    float kt;
};

struct GPUSceneCameraData {
    vec4 pos;
    vec4 look;
    vec4 up;

    float heightAngle;

    float aperture;
    float focalLength;

}

struct GPUSceneLightData {
    int id;
    LightType type;

    vec4 color;
    vec3 function; // Attenuation function

    vec4 pos; // Position with CTM applied (Not applicable to directional lights)
    vec4 dir; // Direction with CTM applied (Not applicable to point lights)

    float penumbra; // Only applicable to spot lights, in RADIANS
    float angle;    // Only applicable to spot lights, in RADIANS

}

enum class GPULightType {
    LIGHT_POINT,
    LIGHT_DIRECTIONAL,
    LIGHT_SPOT,
};


#endif