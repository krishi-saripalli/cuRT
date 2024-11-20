cmake_minimum_required(VERSION 3.16)

# Sets project name
project(raymarcher LANGUAGES CXX CUDA)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};/oscar/rt/9.2/software/0.20-generic/0.20.1/opt/spack/linux-rhel9-x86_64_v3/gcc-11.3.1/glew-2.2.0-plawm2jod5t5dzga2ktzstlqjp4fjt4q/lib64")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Sets C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_OSX_ARCHITECTURES "arm64")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")

# Look up the right value for your hardware at https://developer.nvidia.com/cuda-gpus#compute
set(CMAKE_CUDA_ARCHITECTURES 86)

# Enable CUDA support
enable_language(CUDA)

# Set GLEW paths explicitly before finding packages
set(GLEW_INCLUDE_DIR "/oscar/rt/9.2/software/0.20-generic/0.20.1/opt/spack/linux-rhel9-x86_64_v3/gcc-11.3.1/glew-2.2.0-plawm2jod5t5dzga2ktzstlqjp4fjt4q/include")
set(GLEW_LIBRARY "/oscar/rt/9.2/software/0.20-generic/0.20.1/opt/spack/linux-rhel9-x86_64_v3/gcc-11.3.1/glew-2.2.0-plawm2jod5t5dzga2ktzstlqjp4fjt4q/lib64/libGLEW.so")

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 3.4 REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDA REQUIRED)


# Set CUDA specific compile options to include GLEW
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-I${GLEW_INCLUDE_DIR}")



# Specifies .cpp and .h files to be passed to the compiler
add_executable(${PROJECT_NAME}
    main.cpp

    # utils
    utils/scenefilereader.cpp
    utils/scenefilereader.h
    utils/sceneparser.cpp
    utils/sceneparser.h
    utils/scenedata.h
    utils/openglutils.h
    utils/json.hpp
    utils/rgba.cuh

    #window
    window/window.h
    window/window.cpp

    #camera
    camera/camera.cpp
    camera/camera.h

    #raymarcher
    raymarcher/hit.h
    raymarcher/raymarcher.h
    raymarcher/scene.h
    raymarcher/scene.cpp
    raymarcher/distance.h
    raymarcher/raymarcher.cu

    #shader
    shader/shader.h

    #kernel
    kernel/render.cu
    kernel/cudautils.cuh
    kernel/hit.cuh
    kernel/mat4.cuh
    kernel/render.cuh
    kernel/shape.cuh
    kernel/vec3.cuh
    kernel/vec4.cuh
    kernel/light.cuh
    kernel/light.cu
)


# Set properties for CUDA compilation
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(${PROJECT_NAME} PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-I${GLEW_INCLUDE_DIR}>
)


target_link_libraries(${PROJECT_NAME} PRIVATE
    GLEW::GLEW 
    OpenGL::GL
    OpenGL::GLU
    glfw
    ${CUDA_LIBRARIES}
    cuda
    cudart
    X11
)

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    ${GLEW_INCLUDE_DIR}
)

file(GLOB EIGEN_DIR_CONTENTS ${CMAKE_CURRENT_LIST_DIR}/Eigen/*)
list(LENGTH EIGEN_DIR_CONTENTS EIGEN_DIR_SIZE)
if(EIGEN_DIR_SIZE EQUAL 0)
    message(FATAL_ERROR "Eigen dependency not pulled, please run `git submodule update --init --recursive`")
endif()




# This allows you to `#include <Eigen/...>`
target_include_directories(${PROJECT_NAME} PRIVATE
    ${GLEW_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    Eigen 
)

# Set this flag to silence warnings on Windows
if (MSVC OR MSYS OR MINGW)
  set(CMAKE_CXX_FLAGS "-Wno-volatile")
endif()

# Set this flag to silence warnings on MacOS
if (APPLE)
  set(CMAKE_CXX_FLAGS "-Wno-deprecated-volatile")
endif()