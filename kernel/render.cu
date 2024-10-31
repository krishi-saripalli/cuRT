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

        
        float x = ((col+0.5f)/w) - 0.5f, y = ((h - 1.f - row + 0.5f)/h) - 0.5f;
        vec4 p(0.f,0.f,0.f,1.f);
        vec4 d(vw * x, vh * y, -0.1f , 0.f);
        p = ivm * p;
        d = ivm * d;
        int index = row * (w) + col;
        // RGBA originalColor = imageData[index];

        imageData[index] = RGBA{0,255,0}; // color the screen green
    }

}


// RGBA marchRay(const GPUShape* shapes, const RGBA originalColor, const vec4& p, const vec4& d)