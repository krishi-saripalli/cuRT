#include <iostream>
#include "camera.h"
#include <Eigen/Dense>
#include "utils/scenedata.h"

void Camera::init(SceneCameraData cData) {
    cameraData = cData;


}

Eigen::Matrix4f Camera::getViewMatrix() const {

    Eigen::Vector3f look = cameraData.look.head<3>();
    Eigen::Vector3f up = cameraData.up.head<3>();
    Eigen::Vector3f pos = cameraData.pos.head<3>();

    Eigen::Vector3f w = (-look).normalized();
    Eigen::Vector3f v = (up - ((up.dot(w))*w)).normalized();
    Eigen::Vector3f u = v.cross(w);

    Eigen::Matrix4f translationMatrix;
    translationMatrix << 
        1.0,0.0,0.0,-pos[0], 
        0.0,1.0,0.0,-pos[1],
        0.0,0.0,1.0,-pos[2],
        0.0,0.0,0.0,1.0;

    

    Eigen::Matrix4f rotationMatrix;
    rotationMatrix <<
        u[0],u[1],u[2],0.0,  // Row 1
        v[0],v[1],v[2],0.0,  // Row 2
        w[0],w[1],w[2],0.0,  // Row 3
        0.0,0.0,0.0,1.0;     // Row 4
    
    return rotationMatrix * translationMatrix;
}

float Camera::getAspectRatio(int width, int height) const {
    return float(float(width)/float(height));

}

float Camera::getHeightAngle() const {
    return cameraData.heightAngle;

}

float Camera::getFocalLength() const {
    return cameraData.focalLength;

}

float Camera::getAperture() const {
    return cameraData.aperture;
}