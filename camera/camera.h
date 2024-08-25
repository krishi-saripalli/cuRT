#pragma once

#include "utils/scenedata.h"

// A class representing a virtual camera.



class Camera {
public:
    // Returns the view matrix for the current camera settings.
    // You might also want to define another function that return the inverse of the view matrix.
    Eigen::Matrix4f getViewMatrix() const;

    // Returns the aspect ratio of the camera.
    float getAspectRatio(int width, int height) const;

    // Returns the height angle of the camera in RADIANS.
    float getHeightAngle() const;

    // Returns the focal length of this camera (used for DOF)
    float getFocalLength() const;

    // Returns the focal length of this camera (used for DOF)
    float getAperture() const;

    void init(SceneCameraData cameraData);
    SceneCameraData cameraData;


};
