


 __global__ void renderKernel(
    RGBA* imageData,
    const GPUShape* shapes,
    const mat4 inverseViewMatrix,
    const int width,
    const int height,
    const int numShapes,
    const float viewPlaneWidth,
    const float viewPlaneHeight
) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        float x = ((col+0.5f)/width) - 0.5f, y = ((height - 1.f - row + 0.5f)/height) - 0.5f;
        vec4 p(0.f,0.f,0.f,1.f);
        vec4 d(viewPlaneWidth * x, viewPlaneHeight * y, -1.f * distToViewPlane, 0.f);
        p = inverseViewMatrix * p;
        d = inverseViewMatrix * d;
        int index = row * width + col;
        RGBA originalColor = imageData[index];
        imageData[index] = RGBA{0,255,0}; // color the screen green

    }

}


// RGBA marchRay(const GPUShape* shapes, const RGBA originalColor, const vec4& p, const vec4& d)