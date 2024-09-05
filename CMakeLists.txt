cmake_minimum_required(VERSION 3.16)

# Sets project name
project(raymarcher LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Sets C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_OSX_ARCHITECTURES "arm64") #TODO openmp does not work with arm64?

# #OpenMP
# find_package(OpenMP)

#Vulkan
find_package(Vulkan REQUIRED)

# Specifies required Qt components
find_package(Qt6 REQUIRED COMPONENTS Core)
find_package(Qt6 REQUIRED COMPONENTS Gui)
find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(Qt6 REQUIRED COMPONENTS Xml)

#find glfw
find_package(glfw3 3.4 REQUIRED)

# Specifies .cpp and .h files to be passed to the compiler
add_executable(${PROJECT_NAME}
    main.cpp
    utils/scenefilereader.cpp
    utils/scenefilereader.h
    utils/sceneparser.cpp
    utils/sceneparser.h
    utils/scenedata.h
    utils/rgba.h
    window/window.h
    window/window.cpp
    camera/camera.cpp
    camera/camera.h
    raymarcher/hit.h
    raymarcher/raymarcher.h
    raymarcher/raymarcher.cpp
    raymarcher/scene.h
    raymarcher/scene.cpp
    raymarcher/distance.h

)


target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt::Core
    Qt::Gui
    Qt::Xml
    Qt::Widgets
    Vulkan::Vulkan
    glfw
     # OpenMP::OpenMP_CXX
)

file(GLOB EIGEN_DIR_CONTENTS ${CMAKE_CURRENT_LIST_DIR}/Eigen/*)
list(LENGTH EIGEN_DIR_CONTENTS EIGEN_DIR_SIZE)
if(EIGEN_DIR_SIZE EQUAL 0)
    message(FATAL_ERROR "Eigen dependency not pulled, please run `git submodule update --init --recursive`")
endif()




# This allows you to `#include <Eigen/...>`
target_include_directories(${PROJECT_NAME} PRIVATE
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