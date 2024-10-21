#ifndef GPU_HIT_CUH
#define GPU_HIT_CUH

#include <cuda_runtime.h>
#include "vec3.cuh" 
#include "mat4.cuh"  


struct GPUHit {
    vec3 normal;
    vec3 location;
    int shapeId;
    float distance;

    __host__ __device__ GPUHit(vec3 _normal, vec3 _location, int _shapeId, float _distance)
        : normal(_normal), location(_location), shapeId(_shapeId), distance(_distance) {}
};

#endif