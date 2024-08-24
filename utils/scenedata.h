#pragma once

#include <vector>
#include <string>

#include <Eigen/Dense>

// Enum of the types of virtual lights that might be in the scene
enum class LightType {
    LIGHT_POINT,
    LIGHT_DIRECTIONAL,
    LIGHT_SPOT,
};

// Enum of the types of primitives that might be in the scene
enum class PrimitiveType {
    PRIMITIVE_CUBE,
    PRIMITIVE_CONE,
    PRIMITIVE_CYLINDER,
    PRIMITIVE_SPHERE,
    PRIMITIVE_MESH
};

// Enum of the types of transformations that can be applied
enum class TransformationType {
    TRANSFORMATION_TRANSLATE,
    TRANSFORMATION_SCALE,
    TRANSFORMATION_ROTATE,
    TRANSFORMATION_MATRIX
};

// Type which can be used to store an RGBA color in floats [0,1]
using SceneColor = Eigen::Vector4f;

// Struct which contains the global color coefficients of a scene.
// These are multiplied with the object-specific materials in the lighting equation.
struct SceneGlobalData {
    float ka; // Ambient term
    float kd; // Diffuse term
    float ks; // Specular term
    float kt; // Transparency; used for extra credit (refraction)
};

// Struct which contains raw parsed data fro a single light
struct SceneLight {
    int id;
    LightType type;

    SceneColor color;
    Eigen::Vector3f function; // Attenuation function
    Eigen::Vector4f dir;      // Not applicable to point lights

    float penumbra; // Only applicable to spot lights, in RADIANS
    float angle;    // Only applicable to spot lights, in RADIANS

    float width, height; // No longer supported (area lights)
};

// Struct which contains data for a single light with CTM applied
struct SceneLightData {
    int id;
    LightType type;

    SceneColor color;
    Eigen::Vector3f function; // Attenuation function

    Eigen::Vector4f pos; // Position with CTM applied (Not applicable to directional lights)
    Eigen::Vector4f dir; // Direction with CTM applied (Not applicable to point lights)

    float penumbra; // Only applicable to spot lights, in RADIANS
    float angle;    // Only applicable to spot lights, in RADIANS

    float width, height; // No longer supported (area lights)
};

// Struct which contains data for the camera of a scene
struct SceneCameraData {
    Eigen::Vector4f pos;
    Eigen::Vector4f look;
    Eigen::Vector4f up;

    float heightAngle; // The height angle of the camera in RADIANS

    float aperture;    // Only applicable for depth of field
    float focalLength; // Only applicable for depth of field
};

// Struct which contains data for texture mapping files
struct SceneFileMap {
    SceneFileMap() : isUsed(false) {}

    bool isUsed;
    std::string filename;

    float repeatU;
    float repeatV;

    void clear()
    {
        isUsed = false;
        repeatU = 0.0f;
        repeatV = 0.0f;
        filename = std::string();
    }
};

// Struct which contains data for a material (e.g. one which might be assigned to an object)
struct SceneMaterial {
    SceneColor cAmbient;  // Ambient term
    SceneColor cDiffuse;  // Diffuse term
    SceneColor cSpecular; // Specular term
    float shininess;      // Specular exponent

    SceneColor cReflective; // Used to weight contribution of reflected ray lighting (via multiplication)

    SceneColor cTransparent; // Transparency;        used for extra credit (refraction)
    float ior;               // Index of refraction; used for extra credit (refraction)

    SceneFileMap textureMap; // Used for texture mapping
    float blend;             // Used for texture mapping

    SceneColor cEmissive; // Not used
    SceneFileMap bumpMap; // Not used

    void clear()
    {
        cAmbient = Eigen::Vector4f(0.f,0.f,0.f,0.f);
        cDiffuse = Eigen::Vector4f(0.f,0.f,0.f,0.f);
        cSpecular = Eigen::Vector4f(0.f,0.f,0.f,0.f);
        shininess = 0;

        cReflective = Eigen::Vector4f(0.f,0.f,0.f,0.f);

        cTransparent = Eigen::Vector4f(0.f,0.f,0.f,0.f);
        ior = 0;

        textureMap.clear();
        blend = 0;

        cEmissive = Eigen::Vector4f(0.f,0.f,0.f,0.f);
        bumpMap.clear();
    }
};

// Struct which contains data for a single primitive in a scene
struct ScenePrimitive {
    PrimitiveType type;
    SceneMaterial material;
    std::string meshfile; // Used for triangle meshes
};

// Struct which contains data for a transformation.
struct SceneTransformation {
    TransformationType type;

    Eigen::Vector3f translate; // Only applicable when translating. Defines t_x, t_y, and t_z, the amounts to translate by, along each axis.
    Eigen::Vector3f scale;     // Only applicable when scaling.     Defines s_x, s_y, and s_z, the amounts to scale by, along each axis.
    Eigen::Vector3f rotate;    // Only applicable when rotating.    Defines the axis of rotation; should be a unit vector.
    float angle;         // Only applicable when rotating.    Defines the angle to rotate by in RADIANS, following the right-hand rule.
    Eigen::Matrix4f matrix;    // Only applicable when transforming by a custom matrix. This is that custom matrix.
};

// Struct which represents a node in the scene graph/tree, to be parsed by the student's `SceneParser`.
struct SceneNode {
    std::vector<SceneTransformation*> transformations; // Note the order of transformations described in lab 5
    std::vector<ScenePrimitive*> primitives;
    std::vector<SceneLight*> lights;
    std::vector<SceneNode*> children;
};