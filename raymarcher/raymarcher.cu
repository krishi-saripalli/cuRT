#define EIGEN_NO_CUDA

#include <iostream>

#include <cuda_runtime.h>

#include "raymarcher/raymarcher.h"
#include "shader/shader.h"
#include "utils/rgba.cuh"
#include "../kernel/render.cuh"
#include "../kernel/renderdata.cuh"
#include "../kernel/cudautils.cuh"
#include "../kernel/distance.cuh"


#include <cuda_gl_interop.h>

Raymarcher::Raymarcher(std::unique_ptr<Window> w, const Scene& s, GLuint p) : window(std::move(w)), scene(s)  {

    //register PBO
    pbo = p;
    //last arg says that we intend on overwriting the contexts of the pbo     
    gpuErrorCheck( cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsMapFlagsWriteDiscard) );


    int width = scene.c_width, height = scene.c_height;
    float distToViewPlane = 0.1f, aspectRatio = scene.getCamera().getAspectRatio(width,height);
    float heightAngle = scene.getCamera().getHeightAngle();

    
    Eigen::Matrix4f inverseViewMatrix = scene.getCamera().getViewMatrix().inverse();
    float viewPlaneHeight = 2.f * distToViewPlane * std::tan(.5f*float(heightAngle));
    float viewPlaneWidth = viewPlaneHeight * aspectRatio;

    //map deviceImage to point to the PBO
    void* devPtr;
    size_t numBytes;
    gpuErrorCheck( cudaGraphicsMapResources(1, &cudaPboResource, 0) );
    gpuErrorCheck( cudaGraphicsResourceGetMappedPointer(&devPtr,&numBytes,cudaPboResource) );
    deviceImageData = (RGBA*)devPtr;

    //allocate GPU shapes on the device
    allocateDeviceRenderData();

    // allocate inverse view matrix on the device
    mat4 hostInverseViewMat = mat4(inverseViewMatrix.data());
    gpuErrorCheck( cudaMalloc(&deviceInverseViewMat, sizeof(mat4)) );
    gpuErrorCheck( cudaMemcpy(deviceInverseViewMat, &hostInverseViewMat, sizeof(mat4), cudaMemcpyHostToDevice) );

    // allocate constants on the device
    gpuErrorCheck( cudaMalloc(&deviceWidth, sizeof(int)) );
    gpuErrorCheck( cudaMalloc(&deviceHeight, sizeof(int)) );
    gpuErrorCheck( cudaMalloc(&deviceViewPlaneHeight, sizeof(float)) );
    gpuErrorCheck( cudaMalloc(&deviceViewPlaneWidth, sizeof(float)) );

    gpuErrorCheck( cudaMemcpy(deviceWidth, &width, sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrorCheck( cudaMemcpy(deviceHeight, &height, sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrorCheck( cudaMemcpy(deviceViewPlaneWidth, &viewPlaneWidth, sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrorCheck( cudaMemcpy(deviceViewPlaneHeight, &viewPlaneHeight, sizeof(float), cudaMemcpyHostToDevice) );


    
}

Raymarcher::~Raymarcher() {
    gpuErrorCheck( cudaFree(deviceLights) );
    gpuErrorCheck( cudaFree(deviceShapes) );
    gpuErrorCheck( cudaFree(deviceRenderData) );
    gpuErrorCheck( cudaFree(deviceInverseViewMat) );
    gpuErrorCheck( cudaFree(deviceWidth) );
    gpuErrorCheck( cudaFree(deviceHeight) );

    gpuErrorCheck( cudaFree(deviceViewPlaneWidth) );
    gpuErrorCheck( cudaFree(deviceViewPlaneHeight) );
    gpuErrorCheck( cudaGraphicsUnmapResources(1, &cudaPboResource, 0) );
    gpuErrorCheck( cudaGraphicsUnregisterResource(cudaPboResource) );
    std::cout << "Raymarcher Cleaned Up!" << std::endl;  
}

void Raymarcher::allocateDeviceRenderData() {
    GPURenderData hostRenderData;

    //global data
    hostRenderData.globalData = GPUSceneGlobalData(scene.globalData);

    //camera data
    hostRenderData.cameraData.pos = vec4(scene.cameraData.pos.data());
    hostRenderData.cameraData.look = vec4(scene.cameraData.look.data());
    hostRenderData.cameraData.up = vec4(scene.cameraData.up.data());
    hostRenderData.cameraData.heightAngle = scene.cameraData.heightAngle;
    hostRenderData.cameraData.aperture = scene.cameraData.aperture;
    hostRenderData.cameraData.focalLength = scene.cameraData.focalLength;

    //array sizes
    hostRenderData.numLights = scene.metaData.lights.size();
    hostRenderData.numShapes = scene.metaData.shapes.size();

    GPUSceneLightData* hostLights = new GPUSceneLightData[hostRenderData.numLights];
    GPURenderShapeData* hostShapes = new GPURenderShapeData[hostRenderData.numShapes];


    //copy lights
    for (int i = 0; i < hostRenderData.numLights; ++i) {
        const SceneLightData& cpuLight = scene.metaData.lights[i];
        hostLights[i].id = cpuLight.id;
        hostLights[i].type = static_cast<GPULightType>(cpuLight.type);
        hostLights[i].color = vec4(cpuLight.color.data());
        hostLights[i].function = vec3(cpuLight.function.data());
        hostLights[i].pos = vec4(cpuLight.pos.data());
        hostLights[i].dir = vec4(cpuLight.dir.data());
        hostLights[i].penumbra = cpuLight.penumbra;
        hostLights[i].angle = cpuLight.angle;
    }


    //copy shapes
    for (int i = 0; i < hostRenderData.numShapes; ++i) {
        const RenderShapeData cpuShape = scene.metaData.shapes[i];
        const SceneMaterial& cpuMaterial = cpuShape.primitive.material;
        
        GPUSceneMaterial gpuMaterial(
        vec4(cpuMaterial.cAmbient.data()),
        vec4(cpuMaterial.cDiffuse.data()),
        vec4(cpuMaterial.cSpecular.data()),
        cpuMaterial.shininess,
        vec4(cpuMaterial.cReflective.data()),
        vec4(cpuMaterial.cTransparent.data()),
        cpuMaterial.ior);

        GPUScenePrimitive gpuPrimitive(
            static_cast<GPUPrimitiveType>(cpuShape.primitive.type),
            gpuMaterial
        );

        printf("CPU primitive type: %d\n", cpuShape.primitive.type);
        printf("GPU primitive type: %d\n", gpuPrimitive.type);

        

        // set shape data
        Eigen::Matrix4f ctm = cpuShape.ctm;
        // std::cout << "Eigen CTM:\n" << ctm << std::endl;
        Eigen::Matrix3f upperBlock = ctm.block<3,3>(0,0);
        // std::cout << "Upper 3x3 block:\n" << upperBlock << std::endl;
        Eigen::Matrix3f inverseTransposeCtm = upperBlock.inverse().transpose();
        // std::cout << "IVT3:\n" << inverseTransposeCtm << std::endl;

        mat4 deviceCtm = mat4(ctm.data());
        mat4 deviceInverseCtm = mat4(cpuShape.inverseCtm.data());
        mat3 deviceIVT3 = mat3(inverseTransposeCtm.data());

        // print(deviceCtm, "DEVICE CTM BEFORE SHAPE DATA constructor: ");
        
        hostShapes[i] = GPURenderShapeData(
            gpuPrimitive,
            deviceCtm,
            deviceInverseCtm,
            deviceIVT3
        );


    }

    //allocate device shapes and lights
    gpuErrorCheck( cudaMalloc(&deviceLights, hostRenderData.numLights * sizeof(GPUSceneLightData)) );
    gpuErrorCheck( cudaMalloc(&deviceShapes, hostRenderData.numShapes * sizeof(GPURenderShapeData)) );
    gpuErrorCheck( cudaMemcpy(deviceLights, hostLights, 
           hostRenderData.numLights * sizeof(GPUSceneLightData), 
           cudaMemcpyHostToDevice) );
    gpuErrorCheck( cudaMemcpy(deviceShapes, hostShapes, 
            hostRenderData.numShapes * sizeof(GPURenderShapeData), 
            cudaMemcpyHostToDevice) );

    hostRenderData.lights = deviceLights;
    hostRenderData.shapes = deviceShapes;

    //allocate device GPURenderData
    gpuErrorCheck( cudaMalloc(&deviceRenderData, sizeof(GPURenderData)) );
    gpuErrorCheck( cudaMemcpy(deviceRenderData, &hostRenderData, sizeof(GPURenderData), cudaMemcpyHostToDevice) );

    //free host arrays
    delete[] hostLights;
    delete[] hostShapes;
}

void Raymarcher::run() {

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
        render();


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

void Raymarcher::render() {

    int width = scene.c_width, height = scene.c_height;
    dim3 blockSize(16,16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    ); // number of blocks

    renderKernel<<<gridSize,blockSize>>>(
        deviceImageData,
        deviceRenderData,
        deviceInverseViewMat,
        deviceWidth,
        deviceHeight,
        deviceViewPlaneWidth,
        deviceViewPlaneHeight
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    gpuErrorCheck( cudaDeviceSynchronize());

    
    // update texture from PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

}
