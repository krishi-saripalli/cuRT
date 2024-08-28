
#include <raymarcher/raymarcher.h>
#include <raymarcher/distance.h>
#include <utils/rgba.h>
#include <iostream>


void Raymarcher::render(const Scene& scene, MainWindow& window, RGBA *imageData) {

    float width = scene.c_width, height = scene.c_height, distToViewPlane = 1.0f, aspectRatio = scene.getCamera().getAspectRatio(width,height);
    float heightAngle = scene.getCamera().getHeightAngle();
    Eigen::Matrix4f inverseViewMatrix = scene.getCamera().getViewMatrix().inverse();


    for (int col=0; col < width; col++)  {
        for (int row=0; row < height; row++) {

            //Calculate the center of the pixel in normalize image space coordinates
            float x = ((col+0.5f)/width) - 0.5f, y = ((height - 1 - row+0.5f)/height) - 0.5f;

            //Calculate the view plane dimensions (U,V) and point on view plane
            float viewPlaneHeight = 2.f * distToViewPlane * std::tan(.5f*float(heightAngle));
            float viewPlaneWidth = viewPlaneHeight * aspectRatio;
            Eigen::Vector3f pointOnPlane{viewPlaneWidth * x, viewPlaneHeight * y, -1.f * distToViewPlane};

            //Calculate ray direction
            Eigen::Vector4f p(0.f,0.f,0.f,1.f);
            Eigen::Vector4f d(pointOnPlane.x(), pointOnPlane.y(), pointOnPlane.z(), 0.f);

            //Transform the ray to worldspace
            p = inverseViewMatrix * p;
            d = inverseViewMatrix * d;


            int index = row*scene.c_width + col;
            RGBA originalColor = imageData[index];
            imageData[index] = marchRay(scene,originalColor,p,d);
            window.updatePixel(col,row,imageData[index]); 
        }

    }
    window.updateDisplay();
    std::cout << "Updated Display" << std::endl;
    
}

RGBA Raymarcher::marchRay(const Scene& scene, const RGBA originalColor, Eigen::Vector4f p, Eigen::Vector4f d) {
    
    float distTravelled = 0.f;
    const int NUMBER_OF_STEPS = 10000;
    const float EPSILON = 0.001;
    const float MAX_DISTANCE = 100.0;

    for (int i = 0; i < NUMBER_OF_STEPS; ++i) {

        //March our (world space) position forward by distance travelled
        Eigen::Vector4f currPos = p + (distTravelled * d);

        //Find the closest intersection in the scene
        Hit closestHit = getClosestHit(scene,currPos);

        distTravelled += closestHit.distance;


        if (closestHit.distance <= EPSILON) {
            return RGBA{255,0,0};
        }

        if (distTravelled > MAX_DISTANCE) break; 
    }
    return originalColor;  
}

Hit Raymarcher::getClosestHit(const Scene& scene, const Eigen::Vector4f pos) {

    float minDistance = __FLT_MAX__;
    Hit closestHit{nullptr, minDistance};

    for ( const RenderShapeData& shapeData : scene.metaData.shapes) {

        //Transform our position to object space using the shape's inverse CTM
        Eigen::Vector4f objectSpacePos = shapeData.ctm.inverse() * pos;

        float shapeDistance = getShapeDistance(shapeData,objectSpacePos);

        if (shapeDistance < minDistance) {
            minDistance = shapeDistance;
            closestHit = Hit{&shapeData, minDistance};
        }

    }
    return closestHit;

}

float Raymarcher::getShapeDistance(const RenderShapeData& shapeData, const Eigen::Vector4f pos) {
    return shapeData.primitive.distance(pos.head(3));
}


