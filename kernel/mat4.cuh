#ifndef MAT4H_CUH
#define MAT4H_CUH

#include <cuda_runtime.h>
#include "vec4.cuh"

class mat4 {
public:
    __host__ __device__ mat4() {}
    __host__ __device__ mat4(float m00, float m01, float m02, float m03,
                             float m10, float m11, float m12, float m13,
                             float m20, float m21, float m22, float m23,
                             float m30, float m31, float m32, float m33) {
        m[0] = vec4(m00, m01, m02, m03);
        m[1] = vec4(m10, m11, m12, m13);
        m[2] = vec4(m20, m21, m22, m23);
        m[3] = vec4(m30, m31, m32, m33);
    }
     
     //constructor accepts column-major data and just transposes it
    __host__ __device__ mat4(const float* columnMajorData) {
        m[0] = vec4(columnMajorData[0], columnMajorData[4], columnMajorData[8],  columnMajorData[12]);
        m[1] = vec4(columnMajorData[1], columnMajorData[5], columnMajorData[9],  columnMajorData[13]);
        m[2] = vec4(columnMajorData[2], columnMajorData[6], columnMajorData[10], columnMajorData[14]);
        m[3] = vec4(columnMajorData[3], columnMajorData[7], columnMajorData[11], columnMajorData[15]);
    }

    __host__ __device__ inline vec4& operator[](int i) { return m[i]; }
    __host__ __device__ inline const vec4& operator[](int i) const { return m[i]; }

    vec4 m[4];
};

__host__ __device__ inline vec4 operator*(const mat4& m, const vec4& v) {
    return vec4(dot(m[0], v),
                dot(m[1], v),
                dot(m[2], v),
                dot(m[3], v));
}

#endif