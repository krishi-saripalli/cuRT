
#include <cuda_runtime.h>
#include "render.cuh"
#include "light.cuh"
#include "intersect.cuh"

__device__ Hit getIntersection(const GPURenderShapeData* shapes, const int numShapes, const vec4& origin, const vec4& direction) {
    float minT = FLT_MAX;
    Hit closestHit;
    closestHit.shape = nullptr;
    
    for (int i = 0; i < numShapes; ++i) {
        const GPURenderShapeData& shape = shapes[i];
        
        // transform ray to object space
        vec4 objectPos = shape.inverseCtm * origin;
        vec4 objectDir = shape.inverseCtm * direction;
        vec3 p(objectPos.x(), objectPos.y(), objectPos.z());
        vec3 d(objectDir.x(), objectDir.y(), objectDir.z());
        
        Intersection intersection;
        switch(shape.primitive.type) {
            case GPUPrimitiveType::PRIMITIVE_SPHERE:
                intersection = intersectSphere(p, d);
                break;
            case GPUPrimitiveType::PRIMITIVE_CYLINDER:
                intersection = intersectCylinder(p, d);
                break;
            case GPUPrimitiveType::PRIMITIVE_CONE:
                intersection = intersectCone(p, d);
                break;
            case GPUPrimitiveType::PRIMITIVE_CUBE:
                intersection = intersectCube(p, d);
                break;
            default:
                continue;
        }
        
        if (intersection.t < minT && intersection.t > 0) {
            minT = intersection.t;
            closestHit.shape = &shapes[i];
            
            closestHit.intersection = origin + direction * intersection.t;
            
            vec3 worldNormal = shape.invTransposeCtm * intersection.normal;
            closestHit.normal = vec4(worldNormal.x(), worldNormal.y(), worldNormal.z(), 0.0f);
            closestHit.normal.normalize();
            
            closestHit.direction = direction;
        }
    }
    
    return closestHit;
}

__device__ RGBA traceRay(const GPURenderData& renderData, const RGBA& originalColor, const vec4& origin, const vec4& direction) {
    Hit hit = getIntersection(renderData.shapes, renderData.numShapes, origin, direction);
    
    if (hit.shape != nullptr) {
        vec4 color = illumination(renderData, hit);
        return RGBA{
            (unsigned char)(255.f * fminf(fmaxf(color.r(), 0.f), 1.f)),
            (unsigned char)(255.f * fminf(fmaxf(color.g(), 0.f), 1.f)),
            (unsigned char)(255.f * fminf(fmaxf(color.b(), 0.f), 1.f))
        };
    }
    
    return originalColor;
}


 __global__ void renderKernel(
    RGBA* imageData,
    const GPURenderData* renderData,
    const mat4* inverseViewMatrix,
    const int* width,
    const int* height,
    const float* viewPlaneWidth,
    const float* viewPlaneHeight
) {

    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < *width && row < *height) {
        const int w = *width, h = *height;
        const float vw = *viewPlaneWidth, vh = *viewPlaneHeight;
        const mat4 ivm = *inverseViewMatrix;

        
        float x = ((col+0.5f)/float(w)) - 0.5f, y = ((row + 0.5f)/float(h)) - 0.5f;
        vec4 p(0.f,0.f,0.f,1.f);
        vec4 d(vw * x, vh * y, -1.f , 0.f); // TODO: pass in distToViewPlane so that it isn't hardcoded.
        
        p = ivm * p;
        d = ivm * d;
        d.normalize();

        int index = row * (w) + col;
        imageData[index] = traceRay(*renderData, imageData[index], p, d);
    }

}


