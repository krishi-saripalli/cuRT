#ifndef RENDER_CUH
#define RENDER_CUH

#include <cuda_runtime.h>
#include <float.h>
#include "cudautils.cuh"
#include "vec3.cuh"
#include "vec4.cuh"
#include "mat4.cuh"
#include "shape.cuh"
#include "renderdata.cuh"
#include "hit.cuh"
#include "../utils/rgba.cuh"


 __global__ void renderKernel(
    RGBA* imageData,
    const GPURenderData* renderData,
    const mat4* inverseViewMatrix,
    const int* width,
    const int* height,
    const float* viewPlaneWidth,
    const float* viewPlaneHeight
);


__device__ Hit getClosestHit(const GPURenderShapeData* shapes, const int numShapes, const vec4& worldPos, const vec4& direction); 

__device__ RGBA marchRay(const GPURenderData& renderData, const RGBA& originalColor, const vec4& p, const vec4& d);

__device__ vec3 getNormal(const GPURenderShapeData* shape, const vec3& p); 


#endif