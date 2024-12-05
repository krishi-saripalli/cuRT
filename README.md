# cuRT

cuRT is a GPU-accelerated ray tracer written in CUDA and OpenGL. 

## System Requirements

- NVIDIA GPU with compute capability 8.6 or higher
- CUDA Toolkit 
- CMake 3.16 or higher
- C++20 compatible compiler
- GLEW 2.2.0
- GLFW 3.4
- OpenGL
- OpenMP
- X11 (optional if working on a remote Linux machine)

## Getting Started

1. Clone the repository and initialize submodules:

```
cd raymarcher
git submodule update --init --recursive  # Required for Eigen dependency
```

2. Configure CUDA Architecture:
  - Open `CMakeLists.txt`
  - Locate the line: `set(CMAKE_CUDA_ARCHITECTURES 86)`
  - Change `86` to match your GPU's compute capability
  - You can find your GPU's compute capability at: https://developer.nvidia.com/cuda-gpus#compute

3. GLEW Configuration:
  - Current paths are hardcoded for a specific system. Update these paths to match your GLEW installation:
  ```
  set(GLEW_INCLUDE_DIR "path/to/your/glew/include")
  set(GLEW_LIBRARY "path/to/your/glew/lib64/libGLEW.so")
  ```
  - Also update the `CMAKE_INSTALL_RPATH` if needed

4. Build the project:
```
mkdir build
cd build
cmake ..
make
```

## Running cuRT

The program takes a scene file as input and must be run with an absolute path:

```
./raymarcher /absolute/path/to/your/scenefile`
```

## Scene File Format

cuRT uses a JSON format developed by Brown's [CS1230](https://cs1230.graphics/) course staff to define the 3D environment. Below is a detailed breakdown of the format:

The root object must contain:
- `name`: String identifier for the scene
- `globalData`: Global lighting coefficients
- `cameraData`: Camera setup and position
- `groups`: Array of scene objects and lights

### Global Data
Controls scene-wide lighting parameters:
```json
"globalData": {
   "ambientCoeff": 0.5,    // Ambient light intensity (0-1)
   "diffuseCoeff": 0.5,    // Diffuse reflection intensity (0-1)
   "specularCoeff": 0.5,   // Specular reflection intensity (0-1)
   "transparentCoeff": 0   // Transparency amount (0-1)
}
```

### Camera Data
```json
"cameraData": {
    "position": [-6.0, 4.0, 4.0],  // Camera position [x, y, z]
    "up": [0.0, 1.0, 0.0],         // Up vector [x, y, z]
    "focus": [0, 0, 0],            // Look-at point [x, y, z]
    "heightAngle": 30.0            // Vertical field of view in degrees
}
```

### Light Data
Currently supports point, spot and directional lights

```json
{
    "type": "point",
    "color": [r, g, b],               // RGB values (0-1)
    "attenuationCoeff": [a, b, c]     // Attenuation factors (constant, linear, quadratic)
}
```

### Primitive Data
Support for sphere, cube, cone and cylinder primitives. Meshes are in progress

```json
{
    "type": "sphere",              // or "cube", "cylinder", "cone"
    "diffuse": [r, g, b],         // RGB diffuse color (0-1)
    "specular": [r, g, b],        // Optional: RGB specular color (0-1)
    "shininess": 15.0             // Optional: Specular shininess
}
```

## Known Issues and Limitations

- GLEW paths are currently hardcoded and need to be manually updated
- Compute capability is set to 8.6 by default
- Only tested on Linux systems with X11
- Requires absolute paths for scene files

## Development Notes

- Built with C++20
- Uses CUDA separable compilation
- OpenMP is enabled for CPU parallel processing
- Includes debug flags for CUDA (-G -g)

## TODOs

- [ ] Support Triangle Meshes
- [ ] Remove main render kernel in favor of a wavefront approach based on [this](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf) paper 
- [ ] Add BVH support
