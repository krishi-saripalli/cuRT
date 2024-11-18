#include "hit.cuh"
#ifndef LIGHT_CUH
#define LIGHT_CUH

__device__ illumination(const GPURenderData& renderData, Hit& intersection) {

    for (int i = 0; i < renderData.numLights; ++i) {
        GPUSceneLightData light = renderData.lights[i];

        switch (light.type) {
            case GPULightType::LIGHT_DIRECTIONAL:
                
            
                break;
            case GPULightType::LIGHT_POINT:
                break;
            case GPULightType::LIGHT_SPOT
                break;
            default:
                break;
        }

    }
}

__device__ directionalIllumination(GPUSceneLightData& light, Hit& intersection) {
    
}

#endif