#ifndef GPU_SHAPE_CUH
#define GPU_SHAPE_CUH

#include <cuda_runtime.h>
#include <assert.h>
#include "vec3.cuh" 
#include "vec4.cuh" 
#include "mat3.cuh" 
#include "mat4.cuh" 
#include "distance.cuh"

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
   vec4 cReflective; // Used to weight contribution of reflected ray lighting
   vec4 cTransparent; // Transparency;
   float ior;        // Index of refraction

    GPUSceneMaterial() = default;
   
   __host__ __device__ GPUSceneMaterial(
       const vec4& ambient,
       const vec4& diffuse,
       const vec4& specular,
       float shine,
       const vec4& reflective,
       const vec4& transparent,
       float _ior
   ) : cAmbient(ambient),
       cDiffuse(diffuse),
       cSpecular(specular),
       shininess(shine),
       cReflective(reflective),
       cTransparent(transparent),
       ior(_ior) {}
};


inline __device__ float distanceToPrimitive(GPUPrimitiveType type, const vec3& p) {
    
    switch(type) {
        case GPUPrimitiveType::PRIMITIVE_CUBE:
            return distToCube(p);
        case GPUPrimitiveType::PRIMITIVE_SPHERE:
            return distToSphere(p);
        case GPUPrimitiveType::PRIMITIVE_CYLINDER:
            return distToCylinder(p);
        case GPUPrimitiveType::PRIMITIVE_CONE:
            return distToCone(p);
        default:
            assert(1 == 2); // TODO: throw an error correctly
    }
}


struct GPUScenePrimitive {
    GPUPrimitiveType type;
    GPUSceneMaterial material;

    GPUScenePrimitive() = default;
    __host__ __device__ GPUScenePrimitive(GPUPrimitiveType _type, const GPUSceneMaterial& _material)
        : type(_type), material(_material) {}

    
    __device__ float distance(const vec3& p) const {
        return distanceToPrimitive(type, p);
    }

};


struct GPURenderShapeData {
    GPUScenePrimitive primitive;
    mat4 ctm;
    mat4 inverseCtm;
    mat3 invTransposeCtm; // inverse transpose of the upper 3x3 of the ctm

    GPURenderShapeData() = default;
    __host__ __device__ GPURenderShapeData(GPUScenePrimitive _primitive, const mat4& _ctm,  const mat4& _inverseCtm,  const mat3& _inverseTransposeCtm)
        : primitive(_primitive), ctm(_ctm),  inverseCtm(_inverseCtm), invTransposeCtm(_inverseTransposeCtm) {}
};

#endif