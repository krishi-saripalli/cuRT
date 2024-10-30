#ifndef HIT_CUH
#define HIT_CUH

#include <cuda_runtime.h>
#include "vec3.cuh" 
#include "mat4.cuh"  


struct Hit {
    GPURenderShapeData* shape;
    vec3 normal;
    vec3 intersection;
    float distance;

    __host__ __device__ Hit(GPURenderData* _shape, vec3 _normal, vec3 _location, float _distance)
        : shape(_shape), normal(_normal), location(_location), shapeId(_shapeId), distance(_distance) {}
};

#endif