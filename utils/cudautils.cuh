#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }

// // Usage in .cu files:
// void someFunction() {
//     cudaCheckError(cudaMalloc(&d_data, size));
//     // Other CUDA calls...
// }