
#ifndef LIGHT_CUH
#define LIGHT_CUH
#include "hit.cuh"
#include "renderdata.cuh"

__device__ vec4 illumination(const GPURenderData& renderData, const Hit& hit);


__device__ vec4 directionalIllumination(const GPURenderData& renderData, const Hit& hit, const GPUSceneLightData& light);


#endif