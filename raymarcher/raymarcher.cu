#define EIGEN_NO_CUDA

#include <iostream>

#include <cuda_runtime.h>

#include "raymarcher/raymarcher.h"
#include "raymarcher/distance.h"
#include "shader/shader.h"
#include "utils/rgba.cuh"
#include "kernel/render.cuh"


#include <cuda_gl_interop.h>



Raymarcher::Raymarcher(std::unique_ptr<Window> w) : window(std::move(w)) {
    
}

Raymarcher::~Raymarcher() {
    if (cudaPboResource) {
        cudaGraphicsUnregisterResource(cudaPboResource);
        std::cout << "cudaPboResource Unregistered!" << std::endl;
    }
}

void Raymarcher::run(const Scene& scene) {

    if (glfwGetCurrentContext() == nullptr) {
        std::cerr << "Error: No OpenGL context" << std::endl;
        return;
    }

    while(!(*window).shouldClose()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        if (shaderProgram == 0) {
            std::cerr << "Error: shaderProgram is 0" << std::endl;
            break;
        }

        //TODO: add some conditional logic (only when event handlers receive something)
        render(scene);


        glUseProgram(shaderProgram);
        GET_GL_ERROR("After glUseProgram");

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        GLint texLoc = glGetUniformLocation(shaderProgram, "ourTexture");
        if (texLoc == -1) {
            std::cout << "Warning: Could not find texture uniform" << std::endl;
        }
        glUniform1i(texLoc, 0);

        glBindVertexArray(quad.vao);
        GET_GL_ERROR("After glBindVertexArray");

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        GET_GL_ERROR("After glDrawElements");

        glfwSwapBuffers(window->glWindow);
        glfwPollEvents();

        GET_GL_ERROR("Loop() ERROR\n");
    }
}

void Raymarcher::render(const Scene& scene) {

    int width = scene.c_width, height = scene.c_height;
    float distToViewPlane = 0.1f, aspectRatio = scene.getCamera().getAspectRatio(width,height);
    float heightAngle = scene.getCamera().getHeightAngle();
    Eigen::Matrix4f inverseViewMatrix = scene.getCamera().getViewMatrix().inverse();
    float viewPlaneHeight = 2.f * distToViewPlane * std::tan(.5f*float(heightAngle));
    float viewPlaneWidth = viewPlaneHeight * aspectRatio;

    
    //map PBO to CUDA
    void* devPtr;
    size_t numBytes;
    cudaGraphicsMapResources(1, &cudaPboResource, 0);
    cudaGraphicsResourceGetMappedPointer(&devPtr,&numBytes,cudaPboResource);
    RGBA* deviceImageData = (RGBA*)devPtr;


   
    //allocate GPU shapes on the device
    std::vector<GPUShape> hostShapes;
    GPUShape *deviceShapes;
    int numPrimitives = scene.metaData.shapes.size();
    hostShapes.reserve(numPrimitives);

    for ( const RenderShapeData& shapeData : scene.metaData.shapes) {
        hostShapes.push_back(GPUShape{static_cast<GPUPrimitiveType>(shapeData.primitive.type), mat4(shapeData.inverseCtm.data())});
    }
    cudaMalloc(&deviceShapes, sizeof(GPUShape) * numPrimitives);
    cudaMemcpy(deviceShapes, hostShapes.data(), sizeof(GPUShape) * numPrimitives, cudaMemcpyHostToDevice);


    // allocate inverse view matrix on the device
    mat4 hostInverseViewMat = mat4(inverseViewMatrix.data());
    mat4* deviceInverseViewMat;
    cudaMalloc(&deviceInverseViewMat, sizeof(mat4));
    cudaMemcpy(deviceInverseViewMat, &hostInverseViewMat, sizeof(mat4), cudaMemcpyHostToDevice);

    int *deviceWidth, *deviceHeight, *deviceNumPrimitives;
    float *deviceViewPlaneWidth, *deviceViewPlaneHeight;


    // allocate constants on the device
    cudaMalloc(&deviceWidth, sizeof(int));
    cudaMalloc(&deviceHeight, sizeof(int));
    cudaMalloc(&deviceNumPrimitives, sizeof(int));
    cudaMalloc(&deviceViewPlaneHeight, sizeof(float));
    cudaMalloc(&deviceViewPlaneWidth, sizeof(float));

    cudaMemcpy(deviceWidth, &width, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceHeight, &height, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNumPrimitives, &numPrimitives, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceViewPlaneWidth, &viewPlaneWidth, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceViewPlaneHeight, &viewPlaneHeight, sizeof(float), cudaMemcpyHostToDevice);


    dim3 blockSize(16,16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    ); // number of blocks


   
    renderKernel<<<gridSize,blockSize>>>(
        deviceImageData,
        deviceShapes,
        deviceInverseViewMat,
        deviceWidth,
        deviceHeight,
        deviceNumPrimitives,
        deviceViewPlaneWidth,
        deviceViewPlaneHeight
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    // synchronize
    cudaDeviceSynchronize();

    // unmap CUDA resource
    cudaGraphicsUnmapResources(1, &cudaPboResource, 0);

    // update texture from PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaFree(deviceShapes);
    cudaFree(deviceInverseViewMat);
    cudaFree(deviceWidth);
    cudaFree(deviceHeight);

}

RGBA Raymarcher::marchRay(const Scene& scene, const RGBA originalColor, const Eigen::Vector4f& p, const Eigen::Vector4f& d) {
    
    float distTravelled = 0.f;
    const int NUMBER_OF_STEPS = 1000;
    const float EPSILON = 1e-4;
    const float MAX_DISTANCE = 1000.0f;

    for (int i = 0; i < NUMBER_OF_STEPS; ++i) {

        //March our (world space) position forward by distance travelled
        Eigen::Vector4f currPos = p + (distTravelled * d);

        //Find the closest intersection in the scene
        Hit closestHit = getClosestHit(scene,currPos);

        distTravelled += closestHit.distance;

        
        if (closestHit.distance <= EPSILON) {
            
            Eigen::Vector3f normal = closestHit.normal;

            // Shift normal values from [-1,1] to [0,1]
            return RGBA{
                (std::uint8_t)(255.f * (normal.x() + 1.f) / 2.f),  
                (std::uint8_t)(255.f * (normal.y() + 1.f) / 2.f), 
                (std::uint8_t)(255.f * (normal.z() + 1.f) / 2.f)
            };
        }

        if (distTravelled > MAX_DISTANCE) break; 
    }
    return originalColor;  
}

Hit Raymarcher::getClosestHit(const Scene& scene, const Eigen::Vector4f& pos) {

    float minDistance = __FLT_MAX__;
    Eigen::Vector4f objectSpacePos;
    Hit closestHit{nullptr, minDistance};

    for ( const RenderShapeData& shapeData : scene.metaData.shapes) {

        //Transform our position to object space using the shape's inverse CTM
        objectSpacePos = shapeData.inverseCtm * pos;

        float shapeDistance = getShapeDistance(shapeData,objectSpacePos);

        if (shapeDistance < minDistance) {
            minDistance = shapeDistance;
            closestHit = Hit{&shapeData, minDistance};
        }

    }
    //Store the normal of the point
    objectSpacePos = closestHit.shapeData->inverseCtm * pos;
    closestHit.normal = calculateNormal(*(closestHit.shapeData),objectSpacePos.head(3));
    return closestHit;

}

float Raymarcher::getShapeDistance(const RenderShapeData& shapeData, const Eigen::Vector4f& pos) {
    return shapeData.primitive.distance(pos.head(3));
}


