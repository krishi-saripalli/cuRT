#ifndef VEC4H_CUH
#define VEC4H_CUH

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>

class vec4  {


public:
    __host__ __device__ vec4() {}
    __host__ __device__ vec4(const float i[4]) {e[0] = i[0]; e[1] = i[1]; e[2] = i[2]; e[3] = i[3]; }
    __host__ __device__ vec4(float e0, float e1, float e2, float e3) { e[0] = e0; e[1] = e1; e[2] = e2; e[3] = e3; }
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline const vec4& operator+() const { return *this; }
    __host__ __device__ inline vec4 operator-() const { return vec4(-e[0], -e[1], -e[2], -e[3]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; };

    __host__ __device__ inline vec4& operator+=(const vec4 &v2);
    __host__ __device__ inline vec4& operator-=(const vec4 &v2);
    __host__ __device__ inline vec4& operator*=(const vec4 &v2);
    __host__ __device__ inline vec4& operator/=(const vec4 &v2);
    __host__ __device__ inline vec4& operator*=(const float t);
    __host__ __device__ inline vec4& operator/=(const float t);

    __host__ __device__ inline float length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2] + e[3]*e[3]); }
    __host__ __device__ inline float squared_length() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2] + e[3]*e[3]; }
    __host__ __device__ inline void normalize();


    float e[3];
};



inline std::istream& operator>>(std::istream &is, vec4 &t) {
    is >> t.e[0] >> t.e[1] >> t.e[2] >> t.e[3];
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec4 &t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2] << " " << t.e[3];
    return os;
}

__host__ __device__ inline void vec4::normalize() {
    float k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2] + e[3]*e[3]);
    e[0] *= k; e[1] *= k; e[2] *= k; e[3] *= k;
}

__host__ __device__ inline vec4 operator+(const vec4 &v1, const vec4 &v2) {
    return vec4(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2],  v1.e[3] + v2.e[3]);
}

__host__ __device__ inline vec4 operator-(const vec4 &v1, const vec4 &v2) {
    return vec4(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2],  v1.e[3] - v2.e[3]);
}

__host__ __device__ inline vec4 operator*(const vec4 &v1, const vec4 &v2) {
    return vec4(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2],  v1.e[3] * v2.e[3]);
}

__host__ __device__ inline vec4 operator/(const vec4 &v1, const vec4 &v2) {
    return vec4(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2],  v1.e[3] / v2.e[3]);
}

__host__ __device__ inline vec4 operator*(float t, const vec4 &v) {
    return vec4(t*v.e[0], t*v.e[1], t*v.e[2], t*v.e[3]);
}

__host__ __device__ inline vec4 operator/(const vec4 &v, float t) {
    return vec4(v.e[0]/t, v.e[1]/t, v.e[2]/t, v.e[3]/t);
}

__host__ __device__ inline vec4 operator*(const vec4 &v, float t) {
    return vec4(t*v.e[0], t*v.e[1], t*v.e[2], t*v.e[3]);
}

__host__ __device__ inline float dot(const vec4 &v1, const vec4 &v2) {
    return v1.e[0] *v2.e[0] + v1.e[1] *v2.e[1]  + v1.e[2] *v2.e[2] + v1.e[3] *v2.e[3];
}


__host__ __device__ inline vec4& vec4::operator+=(const vec4 &v){
    e[0]  += v.e[0];
    e[1]  += v.e[1];
    e[2]  += v.e[2];
    e[3]  += v.e[3];
    return *this;
}

__host__ __device__ inline vec4& vec4::operator*=(const vec4 &v){
    e[0]  *= v.e[0];
    e[1]  *= v.e[1];
    e[2]  *= v.e[2];
    e[3]  *= v.e[3];
    return *this;
}

__host__ __device__ inline vec4& vec4::operator/=(const vec4 &v){
    e[0]  /= v.e[0];
    e[1]  /= v.e[1];
    e[2]  /= v.e[2];
    e[3]  /= v.e[3];
    return *this;
}

__host__ __device__ inline vec4& vec4::operator-=(const vec4& v) {
    e[0]  -= v.e[0];
    e[1]  -= v.e[1];
    e[2]  -= v.e[2];
    e[3]  -= v.e[3];
    return *this;
}

__host__ __device__ inline vec4& vec4::operator*=(const float t) {
    e[0]  *= t;
    e[1]  *= t;
    e[2]  *= t;
    e[3]  *= t;
    return *this;
}

__host__ __device__ inline vec4& vec4::operator/=(const float t) {
    float k = 1.0/t;

    e[0]  *= k;
    e[1]  *= k;
    e[2]  *= k;
    e[3]  *= k;
    return *this;
}

__host__ __device__ inline vec4 unit_vector(vec4 v) {
    return v / v.length();
}

#endif