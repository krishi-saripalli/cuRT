
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

__global__ void generateRaysKernel(
    Ray* rays,
    const vec4* cameraPos,     
    const vec4* cameraLook,    
    const vec4* cameraUp,      
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
        
        // NDC coordinates for pixel center
        float x = ((col + 0.5f)/float(w)) - 0.5f;
        float y = ((row + 0.5f)/float(h)) - 0.5f;

        vec3 pos3(cameraPos->x(), cameraPos->y(), cameraPos->z());
        vec3 look3(cameraLook->x(), cameraLook->y(), cameraLook->z());
        vec3 up3(cameraUp->x(), cameraUp->y(), cameraUp->z());

        vec3 w3 = -look3;
        w3.normalize();
        vec3 u3 = cross(up3, w3);
        u3.normalize();
        vec3 v3 = cross(w3, u3);
        v3.normalize();

        // calculate ray direction and convert to vec4
        vec3 dir3 = u3 * (vw * x) + v3 * (vh * y) - w3;
        dir3.normalize();
        vec4 direction(dir3.x(), dir3.y(), dir3.z(), 0.0f);

        int index = row * w + col;
        rays[index].origin = *cameraPos;
        rays[index].direction = direction;
    }
}


 __global__ void renderKernel(
    RGBA* imageData,
    const GPURenderData* renderData,
    const Ray* rays,
    const int* width,
    const int* height
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < *width && row < *height) {
        const int w = *width;
        int index = row * w + col;
        const Ray& ray = rays[index];
        imageData[index] = traceRay(*renderData, imageData[index], ray.origin, ray.direction);
    }
}

__global__ void updateCameraKernel(
    vec4* devicePos,  
    vec4* deviceLook,
    vec4* deviceUp,  
    mat4* deviceInverseViewMat,
    float zoomAmount
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *devicePos += *deviceLook * zoomAmount;
        
        vec3 pos(devicePos->x(), devicePos->y(), devicePos->z());
        vec3 look(deviceLook->x(), deviceLook->y(), deviceLook->z());
        vec3 up(deviceUp->x(), deviceUp->y(), deviceUp->z());

        vec3 w = unit_vector(-look);
        vec3 v = unit_vector(up - (dot(up, w) * w));
        vec3 u = cross(v, w);

        mat4 rotation(
            u.x(), u.y(), u.z(), 0.0f,
            v.x(), v.y(), v.z(), 0.0f,
            w.x(), w.y(), w.z(), 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        );

        mat4 translation(
            1.0f, 0.0f, 0.0f, -pos.x(),
            0.0f, 1.0f, 0.0f, -pos.y(),
            0.0f, 0.0f, 1.0f, -pos.z(),
            0.0f, 0.0f, 0.0f, 1.0f
        );

        mat4 viewMatrix = rotation * translation;
        *deviceInverseViewMat = inverse(viewMatrix);
    }
}

