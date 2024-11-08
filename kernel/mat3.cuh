#ifndef MAT3H_CUH
#define MAT3H_CUH

#include <cuda_runtime.h>
#include "vec3.cuh"

class mat3 {
public:
    __host__ __device__ mat3() {}
    __host__ __device__ mat3(float m00, float m01, float m02,
                            float m10, float m11, float m12,
                            float m20, float m21, float m22) {
        m[0] = vec3(m00, m01, m02);
        m[1] = vec3(m10, m11, m12);
        m[2] = vec3(m20, m21, m22);
    }
    
    //constructor accepts column-major data and just transposes it
    __host__ __device__ mat3(const float* columnMajorData) {
        m[0] = vec3(columnMajorData[0], columnMajorData[3], columnMajorData[6]);
        m[1] = vec3(columnMajorData[1], columnMajorData[4], columnMajorData[7]);
        m[2] = vec3(columnMajorData[2], columnMajorData[5], columnMajorData[8]);
    }

    

    __host__ __device__ inline vec3& operator[](int i) { return m[i]; }
    __host__ __device__ inline const vec3& operator[](int i) const { return m[i]; }

    vec3 m[3];
};

__host__ __device__ inline void print(const mat3& m, const char* name = "") {
    printf("%s\n", name);
    printf("[%.3f, %.3f, %.3f]\n", m[0][0], m[0][1], m[0][2]);
    printf("[%.3f, %.3f, %.3f]\n", m[1][0], m[1][1], m[1][2]);
    printf("[%.3f, %.3f, %.3f]\n", m[2][0], m[2][1], m[2][2]);
}


__host__ __device__ inline vec3 operator*(const mat3& m, const vec3& v) {
    return vec3(dot(m[0], v),
                dot(m[1], v),
                dot(m[2], v));
    }

#endif

