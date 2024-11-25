#ifndef INTERSECT_CUH
#define INTERSECT_CUH

#include <cuda_runtime.h>
#include "hit.cuh"
#include "vec3.cuh"

struct Roots {
    float roots[2];
    int count;
};

struct Intersection {
    vec3 normal;
    float t;
};

inline __device__ Roots getRoots(float A, float B, float C) {
    Roots result;
    
    float discriminant = B * B - 4.0f * A * C;
    
    if (discriminant < 0.0f) {
        result.roots[0] = INFINITY;
        result.count = 1;
    }
    else if (discriminant == 0.0f) {
        result.roots[0] = (-B + sqrtf(discriminant)) / (2.0f * A);
        result.count = 1;
    }
    else {
        result.roots[0] = (-B + sqrtf(discriminant)) / (2.0f * A);
        result.roots[1] = (-B - sqrtf(discriminant)) / (2.0f * A);
        result.count = 2;
    }
    
    return result;
}

inline __device__ Intersection getMinIntersection(const Intersection& a, const Intersection& b) {
    return (a.t < b.t) ? a : b;
}

inline __device__ Intersection intersectSphere(const vec3& p, const vec3& d) {
    float A = d.x()*d.x() + d.y()*d.y() + d.z()*d.z();
    float B = 2.0f*d.x()*p.x() + 2.0f*d.y()*p.y() + 2.0f*d.z()*p.z();
    float C = p.x()*p.x() + p.y()*p.y() + p.z()*p.z() - 0.25f;
    
    Roots r = getRoots(A, B, C);
    
    Intersection result;
    result.t = FLT_MAX;
    
    for(int i = 0; i < r.count; ++i) {
        float t = r.roots[i];
        if(t < result.t) {
            result.t = t;
            vec3 intersection = p + d * t;
            result.normal = vec3(
                2.0f*intersection.x(),
                2.0f*intersection.y(),
                2.0f*intersection.z()
            );
        }
    }
    
    return result;
}

inline __device__ Intersection intersectConeBody(const vec3& p, const vec3& d) {
    float A = d.x()*d.x() + d.z()*d.z() - 0.25f*d.y()*d.y();
    float B = 2.0f*p.x()*d.x() + 2.0f*p.z()*d.z() - 0.5f*p.y()*d.y() + 0.25f*d.y();
    float C = p.x()*p.x() + p.z()*p.z() - 0.25f*p.y()*p.y() + 0.25f*p.y() - (1.0f/16.0f);
    
    Roots r = getRoots(A, B, C);
    
    Intersection result;
    result.t = FLT_MAX;
    
    for(int i = 0; i < r.count; ++i) {
        float t = r.roots[i];
        vec3 intersection = p + d * t;
        
        if(intersection.y() >= -0.5f && intersection.y() <= 0.5f) {
            if(t < result.t) {
                result.t = t;
                result.normal = vec3(
                    2.0f*intersection.x(),
                    (0.5f-intersection.y())/2.0f,
                    2.0f*intersection.z()
                );
            }
        }
    }
    return result;
}

inline __device__ Intersection intersectConeBase(const vec3& p, const vec3& d) {
    float t = (-0.5f - p.y()) / d.y();
    vec3 intersection = p + d * t;
    
    Intersection result;
    result.t = FLT_MAX;
    
    if(intersection.x()*intersection.x() + intersection.z()*intersection.z() <= 0.25f) {
        result.t = t;
        result.normal = vec3(0.0f, -1.0f, 0.0f);
    }
    return result;
}

inline __device__ Intersection intersectCylinderBody(const vec3& p, const vec3& d) {
    float A = d.x()*d.x() + d.z()*d.z();
    float B = 2.0f*d.x()*p.x() + 2.0f*d.z()*p.z();
    float C = p.x()*p.x() + p.z()*p.z() - 0.25f;
    
    Roots r = getRoots(A, B, C);
    
    Intersection result;
    result.t = FLT_MAX;
    
    for(int i = 0; i < r.count; ++i) {
        float t = r.roots[i];
        vec3 intersection = p + d * t;
        
        if(intersection.y() >= -0.5f && intersection.y() <= 0.5f) {
            if(t < result.t) {
                result.t = t;
                result.normal = vec3(
                    2.0f*intersection.x(),
                    0.0f,
                    2.0f*intersection.z()
                );
            }
        }
    }
    return result;
}

inline __device__ Intersection intersectCylinderTop(const vec3& p, const vec3& d) {
    float t = (0.5f - p.y()) / d.y();
    vec3 intersection = p + d * t;
    
    Intersection result;
    result.t = FLT_MAX;
    
    if(intersection.x()*intersection.x() + intersection.z()*intersection.z() <= 0.25f) {
        result.t = t;
        result.normal = vec3(0.0f, 1.0f, 0.0f);
    }
    return result;
}

inline __device__ Intersection intersectCylinderBottom(const vec3& p, const vec3& d) {
    float t = (-0.5f - p.y()) / d.y();
    vec3 intersection = p + d * t;
    
    Intersection result;
    result.t = FLT_MAX;
    
    if(intersection.x()*intersection.x() + intersection.z()*intersection.z() <= 0.25f) {
        result.t = t;
        result.normal = vec3(0.0f, -1.0f, 0.0f);
    }
    return result;
}



inline __device__ Intersection intersectCone(const vec3& p, const vec3& d) {
    Intersection body = intersectConeBody(p, d);
    Intersection base = intersectConeBase(p, d);
    return getMinIntersection(body, base);
}

inline __device__ Intersection intersectCylinder(const vec3& p, const vec3& d) {
    Intersection body = intersectCylinderBody(p, d);
    Intersection top = intersectCylinderTop(p, d);
    Intersection bottom = intersectCylinderBottom(p, d);
    return getMinIntersection(getMinIntersection(body, top), bottom);
}

inline __device__ Intersection intersectCubeFace(const vec3& p, const vec3& d, 
                                               float planeDist, const vec3& normal,
                                               int skipAxis) {
    // skipAxis is the axis perpendicular to this face (e.g., X=0, Y=1, Z=2)
    float t;
    if (normal[skipAxis] > 0) {
        t = (0.5f - p[skipAxis]) / d[skipAxis];
    } else {
        t = (-0.5f - p[skipAxis]) / d[skipAxis];
    }
    
    Intersection result;
    result.t = FLT_MAX;
    
    vec3 intersection = p + d * t;
    
    bool inside = true;
    for(int i = 0; i < 3; i++) {
        if(i != skipAxis) {
            inside = inside && (intersection[i] >= -0.5f && intersection[i] <= 0.5f);
        }
    }
    
    if(inside) {
        result.t = t;
        result.normal = normal;
    }
    return result;
}

inline __device__ Intersection intersectCube(const vec3& p, const vec3& d) {
    Intersection result;
    result.t = FLT_MAX;
    
    Intersection faces[6] = {
        // Right face (+X)
        intersectCubeFace(p, d, 0.5f, vec3(1.0f, 0.0f, 0.0f), 0),
        // Left face (-X)
        intersectCubeFace(p, d, -0.5f, vec3(-1.0f, 0.0f, 0.0f), 0),
        // Top face (+Y)
        intersectCubeFace(p, d, 0.5f, vec3(0.0f, 1.0f, 0.0f), 1),
        // Bottom face (-Y)
        intersectCubeFace(p, d, -0.5f, vec3(0.0f, -1.0f, 0.0f), 1),
        // Front face (+Z)
        intersectCubeFace(p, d, 0.5f, vec3(0.0f, 0.0f, 1.0f), 2),
        // Back face (-Z)
        intersectCubeFace(p, d, -0.5f, vec3(0.0f, 0.0f, -1.0f), 2)
    };
    
    for(int i = 0; i < 6; i++) {
        if(faces[i].t < result.t && faces[i].t > 0) {
            result = faces[i];
        }
    }
    
    return result;
}


#endif