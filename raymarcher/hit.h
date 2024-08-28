#pragma once
#include "../utils/sceneparser.h"

struct Hit {
    const RenderShapeData* shapeData;
    float distance;
    Eigen::Vector3f normal;
};

inline Eigen::Vector3f getNormal(const RenderShapeData& shapeData, const Eigen::Vector3f& p) {
    //Using the centered finite difference, we calculate the gradient of the distance function
    const float h = 0.0001f;
    float dx = shapeData.primitive.distance(p + Eigen::Vector3f(h,0.f,0.f)) - shapeData.primitive.distance(p - Eigen::Vector3f(h,0.f,0.f));
    float dy = shapeData.primitive.distance(p + Eigen::Vector3f(0.f,h,0.f)) - shapeData.primitive.distance(p - Eigen::Vector3f(0.f,h,0.f));
    float dz = shapeData.primitive.distance(p + Eigen::Vector3f(0.f,0.f,h)) - shapeData.primitive.distance(p - Eigen::Vector3f(0.f,0.f,h));

    Eigen::Vector3f normal = Eigen::Vector3f(dx,dy,dz).normalized();
    return normal;


}