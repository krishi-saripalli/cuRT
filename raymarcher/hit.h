#pragma once
#include "../utils/sceneparser.h"

struct Hit {
    const RenderShapeData* shapeData;
    float distance;
    Eigen::Vector3f normal;
};

// Uses the tetrahedral method for finite difference calculation (https://iquilezles.org/articles/normalsSDF/)
inline Eigen::Vector3f calculateNormal(const RenderShapeData& shapeData, const Eigen::Vector3f& p) {
    const float h = 0.0001f;
    const Eigen::Vector2f k(1.0f, -1.0f);
    Eigen::Vector3f xyy(k.x(), k.y(), k.y());
    Eigen::Vector3f yyx(k.y(), k.y(), k.x());
    Eigen::Vector3f yxy(k.y(), k.x(), k.y());
    Eigen::Vector3f xxx(k.x(), k.x(), k.x());

    return (xyy * shapeData.primitive.distance(p + h * xyy) +
            yyx * shapeData.primitive.distance(p + h * yyx) +
            yxy * shapeData.primitive.distance(p + h * yxy) +
            xxx * shapeData.primitive.distance(p + h * xxx))
           .normalized();
}

