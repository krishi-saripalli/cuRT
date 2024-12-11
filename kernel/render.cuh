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

struct Ray {
    vec4 origin;    
    vec4 direction;
};

__global__ void generateRaysKernel(
    Ray* rays,
    const vec4* cameraPos,     
    const vec4* cameraLook,    
    const vec4* cameraUp,      
    const int* width,
    const int* height,
    const float* viewPlaneWidth,
    const float* viewPlaneHeight
);

 __global__ void renderKernel(
    RGBA* imageData,
    const GPURenderData* renderData,
    const Ray* rays,
    const int* width,
    const int* height
);

__global__ void updateCameraKernel(
    vec4* devicePos,  
    vec4* deviceLook, 
    vec4* deviceUp,   
    mat4* deviceInverseViewMat,
    const float zoomAmount
);


__device__ Hit getClosestHit(const GPURenderShapeData* shapes, const int numShapes, const vec4& worldPos, const vec4& direction); 

__device__ RGBA marchRay(const GPURenderData& renderData, const RGBA& originalColor, const vec4& p, const vec4& d);

__device__ vec3 getNormal(const GPURenderShapeData* shape, const vec3& p); 


#endif