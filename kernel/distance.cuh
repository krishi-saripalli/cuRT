#ifndef DISTANCE_CUH
#define DISTANCE_CUH

#include <cuda_runtime.h>
#include "vec3.cuh"

struct vec2 {
    float x, y;
    __device__ vec2(float x_, float y_) : x(x_), y(y_) {}
};

__device__ float distToCube(const vec3& p) {
    const vec3 b(0.5f, 0.5f, 0.5f);
    // create q vector (absolute value of p minus b)
    vec3 q(fabsf(p.x()) - b.x(), 
           fabsf(p.y()) - b.y(), 
           fabsf(p.z()) - b.z());
    
    // get the maximum component of q
    float maxComp = fmaxf(q[0], fmaxf(q[1], q[2]));
    
    // create vector of max(q, 0)
    vec3 qMax(fmaxf(q[0], 0.0f),
              fmaxf(q[1], 0.0f),
              fmaxf(q[2], 0.0f));
    
    return qMax.length() + fminf(maxComp, 0.0f);
}

__device__ float distToSphere(const vec3& p) {
    const float radius = 0.5f;
    return p.length() - radius;
}

__device__ float distToCylinder(const vec3& p) {
    const float h = 1.f, r = 0.5f;
    
    float xzLen = sqrtf(p.x() * p.x() + p.z() * p.z());
    float d_x = xzLen - r;
    float d_y = fabsf(p.y()) - 0.5f * h;
    
    vec2 d(fmaxf(d_x, 0.0f), fmaxf(d_y, 0.0f));
    return fminf(fmaxf(d_x, d_y), 0.0f) + 
           sqrtf(d.x * d.x + d.y * d.y);
}

__device__ float distToCone(const vec3& p) {
    const float h = 1.0f, r = 0.5f;
    const float angle = atan2f(r, h);
    const vec2 c(sinf(angle), cosf(angle));
    

    vec3 pTranslated = p - vec3(0.0f, 0.5f, 0.0f);
    
    vec2 q =  vec2(h * c.x / c.y, h * -1.0f);
    vec2 w(sqrtf(pTranslated.x() * pTranslated.x() + pTranslated.z() * pTranslated.z()),
           pTranslated.y());
    
    float dot_wq = w.x * q.x + w.y * q.y;
    float dot_qq = q.x * q.x + q.y * q.y;
    
    float ratio = fmaxf(fminf(dot_wq / dot_qq, 1.0f), 0.0f);
    vec2 a(w.x - q.x * ratio, w.y - q.y * ratio);
    
    float clampedRatio = fmaxf(fminf(w.x / q.x, 1.0f), 0.0f);
    vec2 b(w.x - q.x * clampedRatio, w.y - q.y);
    
    float k = copysignf(1.0f, q.y);
    float d = fminf(a.x * a.x + a.y * a.y, b.x * b.x + b.y * b.y);
    float s = fmaxf(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
    
    return sqrtf(d) * copysignf(1.0f, s);
}



#endif