#ifndef GPU_SHAPE_CUH
#define GPU_SHAPE_CUH

#include <cuda_runtime.h>
#include "vec3.cuh" 
#include "vec4.cuh" 
#include "mat4.cuh" 
 
typedef float (*GPUDistanceFunction)(const vec3&);

enum class GPUPrimitiveType {
    PRIMITIVE_CUBE,
    PRIMITIVE_CONE,
    PRIMITIVE_CYLINDER,
    PRIMITIVE_SPHERE,
    PRIMITIVE_MESH
};

struct GPUSceneMaterial {
    vec4 cAmbient;  // Ambient term
    vec4 cDiffuse;  // Diffuse term
    vec4 cSpecular; // Specular term
    float shininess;      // Specular exponent

    vec4 cReflective; // Used to weight contribution of reflected ray lighting (via multiplication)

    vec4 cTransparent; // Transparency;
    float ior;               // Index of refraction


};

struct GPUScenePrimitive {
    GPUPrimitiveType type;
    GPUSceneMaterial material;
    GPUDistanceFunction distanceFunction;

    
    float distance(const vec3& p) const {
        return distanceFunction(p);
    }

};


struct GPURenderShapeData {
    GPUScenePrimitive primitive;
    mat4 ctm;
    mat4 inverseCtm;

    GPURenderShapeData() = default;
    __host__ __device__ GPURenderShapeData(GPUScenePrimitive _primitive, const mat4& _ctm,  const mat4& _inverseCtm)
        : primitive(_primitive), ctm(_ctm),  inverseCtm(_inverseCtm) {}
};

#endif