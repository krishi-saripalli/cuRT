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
     
     //constructor accepts column-major data and converts it to row-major
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

__host__ __device__ inline mat4 inverse(const mat4& m) {
    float m00 = m[0][0], m01 = m[0][1], m02 = m[0][2], m03 = m[0][3];
    float m10 = m[1][0], m11 = m[1][1], m12 = m[1][2], m13 = m[1][3];
    float m20 = m[2][0], m21 = m[2][1], m22 = m[2][2], m23 = m[2][3];
    float m30 = m[3][0], m31 = m[3][1], m32 = m[3][2], m33 = m[3][3];

    float c00 = m11 * (m22 * m33 - m32 * m23) - m12 * (m21 * m33 - m31 * m23) + m13 * (m21 * m32 - m31 * m22);
    float c01 = -(m10 * (m22 * m33 - m32 * m23) - m12 * (m20 * m33 - m30 * m23) + m13 * (m20 * m32 - m30 * m22));
    float c02 = m10 * (m21 * m33 - m31 * m23) - m11 * (m20 * m33 - m30 * m23) + m13 * (m20 * m31 - m30 * m21);
    float c03 = -(m10 * (m21 * m32 - m31 * m22) - m11 * (m20 * m32 - m30 * m22) + m12 * (m20 * m31 - m30 * m21));

    float c10 = -(m01 * (m22 * m33 - m32 * m23) - m02 * (m21 * m33 - m31 * m23) + m03 * (m21 * m32 - m31 * m22));
    float c11 = m00 * (m22 * m33 - m32 * m23) - m02 * (m20 * m33 - m30 * m23) + m03 * (m20 * m32 - m30 * m22);
    float c12 = -(m00 * (m21 * m33 - m31 * m23) - m01 * (m20 * m33 - m30 * m23) + m03 * (m20 * m31 - m30 * m21));
    float c13 = m00 * (m21 * m32 - m31 * m22) - m01 * (m20 * m32 - m30 * m22) + m02 * (m20 * m31 - m30 * m21);

    float c20 = m01 * (m12 * m33 - m32 * m13) - m02 * (m11 * m33 - m31 * m13) + m03 * (m11 * m32 - m31 * m12);
    float c21 = -(m00 * (m12 * m33 - m32 * m13) - m02 * (m10 * m33 - m30 * m13) + m03 * (m10 * m32 - m30 * m12));
    float c22 = m00 * (m11 * m33 - m31 * m13) - m01 * (m10 * m33 - m30 * m13) + m03 * (m10 * m31 - m30 * m11);
    float c23 = -(m00 * (m11 * m32 - m31 * m12) - m01 * (m10 * m32 - m30 * m12) + m02 * (m10 * m31 - m30 * m11));

    float c30 = -(m01 * (m12 * m23 - m22 * m13) - m02 * (m11 * m23 - m21 * m13) + m03 * (m11 * m22 - m21 * m12));
    float c31 = m00 * (m12 * m23 - m22 * m13) - m02 * (m10 * m23 - m20 * m13) + m03 * (m10 * m22 - m20 * m12);
    float c32 = -(m00 * (m11 * m23 - m21 * m13) - m01 * (m10 * m23 - m20 * m13) + m03 * (m10 * m21 - m20 * m11));
    float c33 = m00 * (m11 * m22 - m21 * m12) - m01 * (m10 * m22 - m20 * m12) + m02 * (m10 * m21 - m20 * m11);

    float det = m00 * c00 + m01 * c01 + m02 * c02 + m03 * c03;

    if (abs(det) < 1e-8f) {
        return mat4(1.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 1.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, 1.0f, 0.0f,
                   0.0f, 0.0f, 0.0f, 1.0f);
    }

    float invDet = 1.0f / det;

    return mat4(c00 * invDet, c10 * invDet, c20 * invDet, c30 * invDet,
                c01 * invDet, c11 * invDet, c21 * invDet, c31 * invDet,
                c02 * invDet, c12 * invDet, c22 * invDet, c32 * invDet,
                c03 * invDet, c13 * invDet, c23 * invDet, c33 * invDet);
}


__host__ __device__ inline void print(const mat4& m, const char* name = "") {
    printf("%s\n", name);
    printf("[%.3f, %.3f, %.3f, %.3f]\n", m[0][0], m[0][1], m[0][2], m[0][3]);
    printf("[%.3f, %.3f, %.3f, %.3f]\n", m[1][0], m[1][1], m[1][2], m[1][3]);
    printf("[%.3f, %.3f, %.3f, %.3f]\n", m[2][0], m[2][1], m[2][2], m[2][3]);
    printf("[%.3f, %.3f, %.3f, %.3f]\n", m[3][0], m[3][1], m[3][2], m[3][3]);
}

__host__ __device__ inline vec4 operator*(const mat4& m, const vec4& v) {
    return vec4(dot(m[0], v),
                dot(m[1], v),
                dot(m[2], v),
                dot(m[3], v));
}

__host__ __device__ inline mat4 operator*(const mat4& a, const mat4& b) {
    vec4 b_col0(b[0][0], b[1][0], b[2][0], b[3][0]);
    vec4 b_col1(b[0][1], b[1][1], b[2][1], b[3][1]);
    vec4 b_col2(b[0][2], b[1][2], b[2][2], b[3][2]);
    vec4 b_col3(b[0][3], b[1][3], b[2][3], b[3][3]);

    return mat4(
        dot(a[0], b_col0), dot(a[0], b_col1), dot(a[0], b_col2), dot(a[0], b_col3),
        dot(a[1], b_col0), dot(a[1], b_col1), dot(a[1], b_col2), dot(a[1], b_col3),
        dot(a[2], b_col0), dot(a[2], b_col1), dot(a[2], b_col2), dot(a[2], b_col3),
        dot(a[3], b_col0), dot(a[3], b_col1), dot(a[3], b_col2), dot(a[3], b_col3)
    );
}

#endif