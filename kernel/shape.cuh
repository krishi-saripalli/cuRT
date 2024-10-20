#ifndef GPU_SHAPE_CUH
#define GPU_SHAPE_CUH

#include "vec4.cuh" 
#include "mat4.cuh"  

enum class GPUPrimitiveType {
    PRIMITIVE_CUBE,
    PRIMITIVE_CONE,
    PRIMITIVE_CYLINDER,
    PRIMITIVE_SPHERE,
    PRIMITIVE_MESH
};

struct GPUShape {
    GPUPrimitiveType type;
    mat4 inverseCtm;
    int id; //index in original RenderShapeData vector

    __host__ __device__ GPUShape(GPUPrimitiveType _type, const mat4& _inverseCtm)
        : type(_type), inverseCtm(_inverseCtm) {}
};

#endif