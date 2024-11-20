#ifndef HIT_CUH
#define HIT_CUH

#include <cuda_runtime.h>
#include <float.h>
#include "vec3.cuh" 
#include "vec4.cuh" 
#include "mat4.cuh" 
#include "shape.cuh" 


struct Hit {
    const GPURenderShapeData* shape;
    vec4 normal;
    vec4 intersection;
    vec4 direction;
    float distance;

    __host__ __device__ Hit() 
        : shape(nullptr), normal(vec4(0.f,0.f,0.f,0.f)), intersection(vec4(0.f,0.f,0.f,0.f)), direction(vec4(0.f,0.f,0.f,0.f)), distance(FLT_MAX) {}


    __host__ __device__ Hit(GPURenderShapeData* _shape, vec4 _normal, vec4 _intersection, vec4 _direction, float _distance)
        : shape(_shape), normal(_normal), intersection(_intersection), direction(_direction), distance(_distance) {}
};


#endif