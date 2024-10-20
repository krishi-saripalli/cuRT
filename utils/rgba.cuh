#ifndef RGBA_CUH
#define RGBA_CUH

#include <cuda_runtime.h>

struct __align__(4) RGBA {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;

    __host__ __device__ RGBA() : r(0), g(0), b(0), a(255) {}
    __host__ __device__ RGBA(unsigned char _r, unsigned char _g, unsigned char _b, unsigned char _a = 255)
        : r(_r), g(_g), b(_b), a(_a) {}
};

#endif // RGBA_CUH