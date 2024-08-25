#include <iostream>
#include <stdexcept>
#include "camera.h"

#include "utils/scenedata.h"

void Camera::init(SceneCameraData cData) {
    cameraData = cData;


}

Eigen::Matrix4f Camera::getViewMatrix() const {

    Eigen::Vector3f look = Eigen::Vector3f{cameraData.look};
    Eigen::Vector3f up = Eigen::Vector3f{cameraData.up};
    Eigen::Vector3f pos = Eigen::Vector3f{cameraData.pos};

    Eigen::Vector3f w = (-look)/look.norm();
    Eigen::Vector3f v = (up - ((up.dot(w))*w))/((up - ((up.dot(w))*w)).norm());
    Eigen::Vector3f u = v.cross(w);

    Eigen::Matrix4f translationMatrix;
    translationMatrix << 
                    1.0,0.0,0.0,0.0, 
                    0.0,1.0,0.0,0.0, 
                    0.0,0.0,1.0,0.0, 
                    -pos[0],-pos[1],-pos[2],1.0;

    
    Eigen::Matrix4f rotationMatrix;
    rotationMatrix <<
                    u[0],v[0],w[0],0.0, 
                    u[1],v[1],w[1],0.0, 
                    u[2],v[2],w[2],0.0, 
                    0.0,0.0,0.0,1.0;

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