#pragma once

#include <Eigen/Dense>
#include <cmath>

inline float distToCube(const Eigen::Vector3f &p)
{
    const Eigen::Vector3f b = Eigen::Vector3f(.5f, .5f, .5f);
    Eigen::Vector3f q = p.cwiseAbs() - b;
    return q.cwiseMax(0.0f).norm() + std::min({std::max({q[0], q[1], q[2]}), 0.0f});
}

inline float distToSphere(const Eigen::Vector3f &p)
{
    const float radius = 0.5f;
    return (p).norm() - radius;
}

inline float distToCylinder(const Eigen::Vector3f &p)
{
    const float h = 1.f, r = 0.5f;
    Eigen::Vector2f d = Eigen::Vector2f(Eigen::Vector2f(p.x(), p.z()).norm(), p.y()).cwiseAbs() - Eigen::Vector2f(r, .5f * h); // modified because h represents the half-height
    return std::min(std::max(d.x(), d.y()), 0.0f) + d.cwiseMax(0.0f).norm();
}

inline float distToCone(const Eigen::Vector3f &p)
{
    const float h = 1.0f, r = 0.5f;
    const float angle = std::atan2(r, h);
    const Eigen::Vector2f c(std::sin(angle), std::cos(angle));
    Eigen::Vector3f pTranslated = p - Eigen::Vector3f(0.0f, 0.5f, 0.0f);
    Eigen::Vector2f q = h * Eigen::Vector2f(c.x() / c.y(), -1.0f);
    Eigen::Vector2f w(Eigen::Vector2f(pTranslated.x(), pTranslated.z()).norm(), pTranslated.y());
    float dot_wq = w.dot(q);
    float dot_qq = q.dot(q);
    Eigen::Vector2f a = w - q * std::clamp(dot_wq / dot_qq, 0.0f, 1.0f);
    float clampedRatio = std::clamp(w.x() / q.x(), 0.0f, 1.0f);
    Eigen::Vector2f b = w - q.cwiseProduct(Eigen::Vector2f(clampedRatio, 1.0f));
    float k = std::copysign(1.0f, q.y());
    float d = std::min(a.dot(a), b.dot(b));
    float s = std::max(k * (w.x() * q.y() - w.y() * q.x()), k * (w.y() - q.y()));
    return std::sqrt(d) * std::copysign(1.0f, s);
}