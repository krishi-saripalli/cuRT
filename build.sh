#Clear out build directory
rm -rf build/*


cd build

module load cmake
module load cuda
module load glew

cmake -DCMAKE_INSTALL_PREFIX=/users/ksaripal/libraries/glfw-3.4 ..