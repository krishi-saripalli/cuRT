#ifndef RENDER_CUH
#define RENDER_CUH

#include <cuda_runtime.h>
#include "cudautils.cuh"
#include "vec3.cuh"
#include "vec4.cuh"
#include "mat4.cuh"
#include "shape.cuh"
#include "../utils/rgba.cuh"


 __global__ void renderKernel(
    RGBA* imageData,
    const GPUShape* shapes,
    const mat4* inverseViewMatrix,
    const int* width,
    const int* height,
    const int* numShapes,
    const float* viewPlaneWidth,
    const float* viewPlaneHeight
);


#endif