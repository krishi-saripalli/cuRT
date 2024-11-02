#ifndef HIT_CUH
#define HIT_CUH

#include <cuda_runtime.h>
#include "vec3.cuh" 
#include "vec4.cuh" 
#include "mat4.cuh"  


struct Hit {
    const GPURenderShapeData* shape;
    vec4 normal;
    vec4 intersection;
    float distance;

    __host__ __device__ Hit() 
        : shape(nullptr), normal(vec4(0.f,0.f,0.f,0.f)), intersection(vec4(0.f,0.f,0.f,0.f)), distance(FLT_MAX) {}


    __host__ __device__ Hit(GPURenderShapeData* _shape, vec4 _normal, vec4 _intersection, float _distance)
        : shape(_shape), normal(_normal), intersection(_intersection), distance(_distance) {}
};


#endif