#pragma once
#include "../utils/sceneparser.h"

struct Hit {
    const RenderShapeData* shapeData;
    float distance;
    Eigen::Vector3f normal;
};

inline Eigen::Vector3f getNormal(const RenderShapeData& shapeData, const Eigen::Vector3f& p) {
    const float h = 0.0001f;
    const Eigen::Vector2f k(1.0f, -1.0f);

    return (Eigen::Vector3f(k.x(), k.y(), k.y()) * shapeData.primitive.distance(p + h * Eigen::Vector3f(k.x(), k.y(), k.y())) +
            Eigen::Vector3f(k.y(), k.y(), k.x()) * shapeData.primitive.distance(p + h * Eigen::Vector3f(k.y(), k.y(), k.x())) +
            Eigen::Vector3f(k.y(), k.x(), k.y()) * shapeData.primitive.distance(p + h * Eigen::Vector3f(k.y(), k.x(), k.y())) +
            Eigen::Vector3f(k.x(), k.x(), k.x()) * shapeData.primitive.distance(p + h * Eigen::Vector3f(k.x(), k.x(), k.x())))
           .normalized();
}