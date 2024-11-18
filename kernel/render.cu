#include "render.cuh"


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

        
        float x = ((col+0.5f)/w) - 0.5f, y = ((row + 0.5f)/h) - 0.5f; // flip y coord so that origin is at bottom-left corner
        vec4 p(0.f,0.f,0.f,1.f);
        vec4 d(vw * x, vh * y, -0.1f , 0.f);
        p = ivm * p;
        d = ivm * d;
        int index = row * (w) + col;

        

        imageData[index] = marchRay(*renderData, imageData[index], p, d);
    }

}


__device__ Hit getClosestHit(const GPURenderShapeData* shapes, const int numShapes, const vec4& worldPos) {
    float minDistance = FLT_MAX;
    vec4 objectPos;
    Hit closestHit;

    for (int i = 0; i < numShapes; ++i) {
        const GPURenderShapeData& shape = shapes[i];
        objectPos = shape.inverseCtm * worldPos;
 
        float distance = shape.primitive.distance(vec3(objectPos.x(),objectPos.y(),objectPos.z()));

        if (distance < minDistance) {
            minDistance = distance;
            closestHit.distance = minDistance;
            closestHit.shape = &shapes[i];
        }

    }

    objectPos = closestHit.shape->inverseCtm * worldPos;
    closestHit.intersection =  closestHit.shape->ctm * objectPos;
    vec3 normal3 = getNormal(closestHit.shape, vec3(objectPos.x(),objectPos.y(),objectPos.z()));
    normal3 = closestHit.shape->invTransposeCtm * normal3;
    normal3.normalize();
    closestHit.normal = vec4(normal3.x(),normal3.y(),normal3.z(),0.0f);

    return closestHit;
}


__device__ RGBA marchRay(const GPURenderData& renderData, const RGBA& originalColor, const vec4& p, const vec4& d) {
    
    float distTravelled = 0.f;
    const int NUMBER_OF_STEPS = 1000;
    const float EPSILON = 1e-5;
    const float MAX_DISTANCE = 1000.0f;

    for (int i = 0; i < NUMBER_OF_STEPS; ++i) {

        vec4 currPos = p + (distTravelled * d);
        Hit closestHit = getClosestHit(renderData.shapes,renderData.numShapes,currPos);
        distTravelled += closestHit.distance;

        
        if (closestHit.distance <= EPSILON) {
            
            vec4 normal = closestHit.normal;

            return RGBA{
                (unsigned char)(255.f * (normal.x() + 1.f) / 2.f),  
                (unsigned char)(255.f * (normal.y() + 1.f) / 2.f), 
                (unsigned char)(255.f * (normal.z() + 1.f) / 2.f)
            };
            

            
        }

        if (distTravelled > MAX_DISTANCE) break; 
    }
    return originalColor;  
}

// Tetrahedron method from https://iquilezles.org/articles/normalsSDF/
__device__ vec3 getNormal(const GPURenderShapeData* shape, const vec3& p) {
    const float h = 0.0001f;
    const vec3 k(1.0f, -1.0f, 0.0f);
    vec3 xyy(k.x(), k.y(), k.y());
    vec3 yyx(k.y(), k.y(), k.x());
    vec3 yxy(k.y(), k.x(), k.y());
    vec3 xxx(k.x(), k.x(), k.x());

    vec3 normal = shape->primitive.distance(p + h * xyy) * xyy  +
            shape->primitive.distance(p + h * yyx) * yyx +
            shape->primitive.distance(p + h * yxy) * yxy +
            shape->primitive.distance(p + h * xxx) * xxx;
    
    normal.normalize();

    return normal;
}



// RGBA marchRay(const GPUShape* shapes, const RGBA originalColor, const vec4& p, const vec4& d)