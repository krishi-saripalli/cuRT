#pragma once

#include<Eigen/Dense>
#include <cmath>

inline float distToCube(const Eigen::Vector3f pos, const Eigen::Vector3f b) {
    Eigen::Vector3f q = pos.cwiseAbs() - b;
    return q.cwiseMax(0.0f).norm() + std::min({std::max({q[0], q[1], q[2]}), 0.0f});
}

inline float distToSphere(const Eigen::Vector3f& pos, float radius) {
    return (pos).norm() - radius;
}


inline float distToCone(const Eigen::Vector3f& p, float h = 1.0f, float r = 0.5f) {
    // Constants for a unit cone
    float angle = std::atan2(r, h);
    const Eigen::Vector2f c(std::sin(angle), std::cos(angle));

    //Translate such that cone apex is at (0.0, 0.5, 0.0) by subtraction
    Eigen::Vector3f pTranslated = p - Eigen::Vector3f(0.0f,0.5f,0.0f);
    

    // q is the point at the base in 2D
    Eigen::Vector2f q = h * Eigen::Vector2f(c.x() / c.y(), -1.0f);
    
    // w is (length of p.xz, p.y)
    Eigen::Vector2f w(Eigen::Vector2f(pTranslated.x(), pTranslated.z()).norm(), pTranslated.y());
    
    // Calculate a
    float dot_wq = w.dot(q);
    float dot_qq = q.dot(q);
    Eigen::Vector2f a = w - q * std::clamp(dot_wq / dot_qq, 0.0f, 1.0f);
    
    // Calculate b
    float clampedRatio = std::clamp(w.x() / q.x(), 0.0f, 1.0f);
    Eigen::Vector2f b = w - q.cwiseProduct(Eigen::Vector2f(clampedRatio, 1.0f));
    
    // Calculate k
    float k = std::copysign(1.0f, q.y());
    
    // Calculate d
    float d = std::min(a.dot(a), b.dot(b));
    
    // Calculate s
    float s = std::max(k * (w.x() * q.y() - w.y() * q.x()), k * (w.y() - q.y()));
    
    // Final result
    return std::sqrt(d) * std::copysign(1.0f, s);
}