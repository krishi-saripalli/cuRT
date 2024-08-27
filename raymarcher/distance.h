
#include<Eigen/Dense>

inline float distToCube(const Eigen::Vector3f pos, const Eigen::Vector3f b) {
    Eigen::Vector3f q = pos.cwiseAbs() - b;
    return q.cwiseMax(0.0f).norm() + std::min({std::max({q[0], q[1], q[2]}), 0.0f});
}

inline float distToSphere(const Eigen::Vector3f& pos, float radius) {
    return (pos - Eigen::Vector3f(0.f,0.f,0.f)).norm() - radius;
}