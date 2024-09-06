#include "scenefilereader.h"
#include "scenedata.h"
#include "../raymarcher/distance.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <fstream>




#define ERROR_AT(e) "error at byte " << e.byte << ": "
#define PARSE_ERROR(e) std::cout << ERROR_AT(e) << "could not parse JSON" << std::endl
#define UNSUPPORTED_ELEMENT(e) std::cout << ERROR_AT(e) << "unsupported element" << std::endl;

ScenefileReader::ScenefileReader(const std::string &name) {
    file_name = name;

    memset(&m_cameraData, 0, sizeof(SceneCameraData));
    memset(&m_globalData, 0, sizeof(SceneGlobalData));

    m_root = new SceneNode;

    m_templates.clear();
    m_nodes.clear();

    m_nodes.push_back(m_root);
}

ScenefileReader::~ScenefileReader() {
    // Delete all Scene Nodes
    for (unsigned int node = 0; node < m_nodes.size(); node++) {
        for (size_t i = 0; i < (m_nodes[node])->transformations.size(); i++)
        {
            delete (m_nodes[node])->transformations[i];
        }
        for (size_t i = 0; i < (m_nodes[node])->primitives.size(); i++)
        {
            delete (m_nodes[node])->primitives[i];
        }
        (m_nodes[node])->transformations.clear();
        (m_nodes[node])->primitives.clear();
        (m_nodes[node])->children.clear();
        delete m_nodes[node];
    }

    m_nodes.clear();
    m_templates.clear();
}

SceneGlobalData ScenefileReader::getGlobalData() const {
    return m_globalData;
}

SceneCameraData ScenefileReader::getCameraData() const {
    return m_cameraData;
}

SceneNode *ScenefileReader::getRootNode() const {
    return m_root;
}

bool ScenefileReader::readJSON() {
    // Read the file
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cout << "could not open " << file_name << std::endl;
        return false;
    }

    // Load the JSON document
    json scenefile;
    try {
        file >> scenefile;
    } catch (json::parse_error& e) {
        PARSE_ERROR(e);
        return false;
    }
    file.close();

    if (!scenefile.is_object()) {
        std::cout << "document is not an object" << std::endl;
        return false;
    }

    if (!scenefile.contains("globalData")) {
        std::cout << "missing required field \"globalData\" on root object" << std::endl;
        return false;
    }
    if (!scenefile.contains("cameraData")) {
        std::cout << "missing required field \"cameraData\" on root object" << std::endl;
        return false;
    }

    std::vector<std::string> requiredFields = {"globalData", "cameraData"};
    std::vector<std::string> optionalFields = {"name", "groups", "templateGroups"};
    std::vector<std::string> allFields = requiredFields;
    allFields.insert(allFields.end(), optionalFields.begin(), optionalFields.end());

    // If other fields are present, raise an error
    for (auto& [key, value] : scenefile.items()) {
        if (std::find(allFields.begin(), allFields.end(), key) == allFields.end()) {
            std::cout << "unknown field \"" << key << "\" on root object" << std::endl;
            return false;
        }
    }

    // Parse the global data
    if (!parseGlobalData(scenefile["globalData"])) {
        std::cout << "could not parse \"globalData\"" << std::endl;
        return false;
    }

    // Parse the camera data
    if (!parseCameraData(scenefile["cameraData"])) {
        std::cout << "could not parse \"cameraData\"" << std::endl;
        return false;
    }

    // Parse the template groups
    if (scenefile.contains("templateGroups")) {
        if (!parseTemplateGroups(scenefile["templateGroups"])) {
            return false;
        }
    }

    // Parse the groups
    if (scenefile.contains("groups")) {
        if (!parseGroups(scenefile["groups"], m_root)) {
            return false;
        }
    }

    std::cout << "Finished reading " << file_name << std::endl;
    return true;
}

bool ScenefileReader::parseGlobalData(const json& globalData) {
    std::vector<std::string> requiredFields = {"ambientCoeff", "diffuseCoeff", "specularCoeff"};
    std::vector<std::string> optionalFields = {"transparentCoeff"};
    std::vector<std::string> allFields = requiredFields;
    allFields.insert(allFields.end(), optionalFields.begin(), optionalFields.end());

    for (auto& [key, value] : globalData.items()) {
        if (std::find(allFields.begin(), allFields.end(), key) == allFields.end()) {
            std::cout << "unknown field \"" << key << "\" on globalData object" << std::endl;
            return false;
        }
    }
    for (const auto& field : requiredFields) {
        if (!globalData.contains(field)) {
            std::cout << "missing required field \"" << field << "\" on globalData object" << std::endl;
            return false;
        }
    }

    // Parse the global data
    if (globalData["ambientCoeff"].is_number()) {
        m_globalData.ka = globalData["ambientCoeff"].get<float>();
    } else {
        std::cout << "globalData ambientCoeff must be a floating-point value" << std::endl;
        return false;
    }
    if (globalData["diffuseCoeff"].is_number()) {
        m_globalData.kd = globalData["diffuseCoeff"].get<float>();
    } else {
        std::cout << "globalData diffuseCoeff must be a floating-point value" << std::endl;
        return false;
    }
    if (globalData["specularCoeff"].is_number()) {
        m_globalData.ks = globalData["specularCoeff"].get<float>();
    } else {
        std::cout << "globalData specularCoeff must be a floating-point value" << std::endl;
        return false;
    }
    if (globalData.contains("transparentCoeff")) {
        if (globalData["transparentCoeff"].is_number()) {
            m_globalData.kt = globalData["transparentCoeff"].get<float>();
        } else {
            std::cout << "globalData transparentCoeff must be a floating-point value" << std::endl;
            return false;
        }
    }

    return true;
}

bool ScenefileReader::parseLightData(const json& lightData, SceneNode *node) {
    std::vector<std::string> requiredFields = {"type", "color"};
    std::vector<std::string> optionalFields = {"name", "attenuationCoeff", "direction", "penumbra", "angle"};
    std::vector<std::string> allFields = requiredFields;
    allFields.insert(allFields.end(), optionalFields.begin(), optionalFields.end());

    for (auto& [key, value] : lightData.items()) {
        if (std::find(allFields.begin(), allFields.end(), key) == allFields.end()) {
            std::cout << "unknown field \"" << key << "\" on light object" << std::endl;
            return false;
        }
    }
    for (const auto& field : requiredFields) {
        if (!lightData.contains(field)) {
            std::cout << "missing required field \"" << field << "\" on light object" << std::endl;
            return false;
        }
    }

    // Create a default light
    SceneLight *light = new SceneLight();
    memset(light, 0, sizeof(SceneLight));
    node->lights.push_back(light);

    light->dir = Eigen::Vector4f(0.f, 0.f, 0.f, 0.f);
    light->function = Eigen::Vector3f(1, 0, 0);

    // parse the color
    if (!lightData["color"].is_array()) {
        std::cout << "light color must be of type array" << std::endl;
        return false;
    }
    auto colorArray = lightData["color"].get<std::vector<float>>();
    if (colorArray.size() != 3) {
        std::cout << "light color must be of size 3" << std::endl;
        return false;
    }
    light->color[0] = colorArray[0];
    light->color[1] = colorArray[1];
    light->color[2] = colorArray[2];

    // parse the type
    if (!lightData["type"].is_string()) {
        std::cout << "light type must be of type string" << std::endl;
        return false;
    }
    std::string lightType = lightData["type"].get<std::string>();

    // parse directional light
    if (lightType == "directional") {
        light->type = LightType::LIGHT_DIRECTIONAL;

        // parse direction
        if (!lightData.contains("direction")) {
            std::cout << "directional light must contain field \"direction\"" << std::endl;
            return false;
        }
        if (!lightData["direction"].is_array()) {
            std::cout << "directional light direction must be of type array" << std::endl;
            return false;
        }
        auto directionArray = lightData["direction"].get<std::vector<float>>();
        if (directionArray.size() != 3) {
            std::cout << "directional light direction must be of size 3" << std::endl;
            return false;
        }
        light->dir[0] = directionArray[0];
        light->dir[1] = directionArray[1];
        light->dir[2] = directionArray[2];
    }
    else if (lightType == "point") {
        light->type = LightType::LIGHT_POINT;

        // parse the attenuation coefficient
        if (!lightData.contains("attenuationCoeff")) {
            std::cout << "point light must contain field \"attenuationCoeff\"" << std::endl;
            return false;
        }
        if (!lightData["attenuationCoeff"].is_array()) {
            std::cout << "point light attenuationCoeff must be of type array" << std::endl;
            return false;
        }
        auto attenuationArray = lightData["attenuationCoeff"].get<std::vector<float>>();
        if (attenuationArray.size() != 3) {
            std::cout << "point light attenuationCoeff must be of size 3" << std::endl;
            return false;
        }
        light->function[0] = attenuationArray[0];
        light->function[1] = attenuationArray[1];
        light->function[2] = attenuationArray[2];
    }
    else if (lightType == "spot") {
        std::vector<std::string> pointRequiredFields = {"direction", "penumbra", "angle", "attenuationCoeff"};
        for (const auto& field : pointRequiredFields) {
            if (!lightData.contains(field)) {
                std::cout << "missing required field \"" << field << "\" on spotlight object" << std::endl;
                return false;
            }
        }
        light->type = LightType::LIGHT_SPOT;

        // parse direction
        if (!lightData["direction"].is_array()) {
            std::cout << "spotlight direction must be of type array" << std::endl;
            return false;
        }
        auto directionArray = lightData["direction"].get<std::vector<float>>();
        if (directionArray.size() != 3) {
            std::cout << "spotlight direction must be of size 3" << std::endl;
            return false;
        }
        light->dir[0] = directionArray[0];
        light->dir[1] = directionArray[1];
        light->dir[2] = directionArray[2];

        // parse attenuation coefficient
        if (!lightData["attenuationCoeff"].is_array()) {
            std::cout << "spotlight attenuationCoeff must be of type array" << std::endl;
            return false;
        }
        auto attenuationArray = lightData["attenuationCoeff"].get<std::vector<float>>();
        if (attenuationArray.size() != 3) {
            std::cout << "spotlight attenuationCoeff must be of size 3" << std::endl;
            return false;
        }
        light->function[0] = attenuationArray[0];
        light->function[1] = attenuationArray[1];
        light->function[2] = attenuationArray[2];

        // parse penumbra
        if (!lightData["penumbra"].is_number()) {
            std::cout << "spotlight penumbra must be of type float" << std::endl;
            return false;
        }
        light->penumbra = lightData["penumbra"].get<float>() * M_PI / 180.f;

        // parse angle
        if (!lightData["angle"].is_number()) {
            std::cout << "spotlight angle must be of type float" << std::endl;
            return false;
        }
        light->angle = lightData["angle"].get<float>() * M_PI / 180.f;
    }
    else {
        std::cout << "unknown light type \"" << lightType << "\"" << std::endl;
        return false;
    }

    return true;
}

bool ScenefileReader::parseCameraData(const json& cameradata) {
    std::vector<std::string> requiredFields = {"position", "up", "heightAngle"};
    std::vector<std::string> optionalFields = {"aperture", "focalLength", "look", "focus"};
    std::vector<std::string> allFields = requiredFields;
    allFields.insert(allFields.end(), optionalFields.begin(), optionalFields.end());

    for (auto& [key, value] : cameradata.items()) {
        if (std::find(allFields.begin(), allFields.end(), key) == allFields.end()) {
            std::cout << "unknown field \"" << key << "\" on cameraData object" << std::endl;
            return false;
        }
    }
    for (const auto& field : requiredFields) {
        if (!cameradata.contains(field)) {
            std::cout << "missing required field \"" << field << "\" on cameraData object" << std::endl;
            return false;
        }
    }

    // Must have either look or focus, but not both
    if (cameradata.contains("look") && cameradata.contains("focus")) {
        std::cout << "cameraData cannot contain both \"look\" and \"focus\"" << std::endl;
        return false;
    }

    // Parse the camera data
    if (cameradata["position"].is_array()) {
        auto position = cameradata["position"].get<std::vector<float>>();
        if (position.size() != 3) {
            std::cout << "cameraData position must have 3 elements" << std::endl;
            return false;
        }
        m_cameraData.pos[0] = position[0];
        m_cameraData.pos[1] = position[1];
        m_cameraData.pos[2] = position[2];
        m_cameraData.pos[3] = 1.f;
    }
    else {
        std::cout << "cameraData position must be an array" << std::endl;
        return false;
    }

    if (cameradata["up"].is_array()) {
        auto up = cameradata["up"].get<std::vector<float>>();
        if (up.size() != 3) {
            std::cout << "cameraData up must have 3 elements" << std::endl;
            return false;
        }
        m_cameraData.up[0] = up[0];
        m_cameraData.up[1] = up[1];
        m_cameraData.up[2] = up[2];
        m_cameraData.up[3] = 0.f;
    }
    else {
        std::cout << "cameraData up must be an array" << std::endl;
        return false;
    }

    if (cameradata["heightAngle"].is_number()) {
        m_cameraData.heightAngle = cameradata["heightAngle"].get<float>() * M_PI / 180.f;
    }
    else {
        std::cout << "cameraData heightAngle must be a floating-point value" << std::endl;
        return false;
    }

    if (cameradata.contains("aperture")) {
        if (cameradata["aperture"].is_number()) {
            m_cameraData.aperture = cameradata["aperture"].get<float>();
        }
        else {
            std::cout << "cameraData aperture must be a floating-point value" << std::endl;
            return false;
        }
    }

    if (cameradata.contains("focalLength")) {
        if (cameradata["focalLength"].is_number()) {
            m_cameraData.focalLength = cameradata["focalLength"].get<float>();
        }
        else {
            std::cout << "cameraData focalLength must be a floating-point value" << std::endl;
            return false;
        }
    }

    // Parse the look or focus
    if (cameradata.contains("look")) {
        if (cameradata["look"].is_array()) {
            auto look = cameradata["look"].get<std::vector<float>>();
            if (look.size() != 3) {
                std::cout << "cameraData look must have 3 elements" << std::endl;
                return false;
            }
            m_cameraData.look[0] = look[0];
            m_cameraData.look[1] = look[1];
            m_cameraData.look[2] = look[2];
            m_cameraData.look[3] = 0.f;
        }
        else {
            std::cout << "cameraData look must be an array" << std::endl;
            return false;
        }
    }
    else if (cameradata.contains("focus")) {
        if (cameradata["focus"].is_array()) {
            auto focus = cameradata["focus"].get<std::vector<float>>();
            if (focus.size() != 3) {
                std::cout << "cameraData focus must have 3 elements" << std::endl;
                return false;
            }
            m_cameraData.look[0] = focus[0];
            m_cameraData.look[1] = focus[1];
            m_cameraData.look[2] = focus[2];
            m_cameraData.look[3] = 1.f;
        }
        else {
            std::cout << "cameraData focus must be an array" << std::endl;
            return false;
        }
    }

    // Convert the focus point (stored in the look vector) into a
    // look vector from the camera position to that focus point.
    if (cameradata.contains("focus")) {
        m_cameraData.look -= m_cameraData.pos;
    }

    return true;
}

bool ScenefileReader::parseTemplateGroups(const json& templateGroups) {
    if (!templateGroups.is_array()) {
        std::cout << "templateGroups must be an array" << std::endl;
        return false;
    }

    for (const auto& templateGroup : templateGroups) {
        if (!templateGroup.is_object()) {
            std::cout << "templateGroup items must be of type object" << std::endl;
            return false;
        }

        if (!parseTemplateGroupData(templateGroup)) {
            return false;
        }
    }

    return true;
}

bool ScenefileReader::parseTemplateGroupData(const json& templateGroup) {
    std::vector<std::string> requiredFields = {"name"};
    std::vector<std::string> optionalFields = {"translate", "rotate", "scale", "matrix", "lights", "primitives", "groups"};
    std::vector<std::string> allFields = requiredFields;
    allFields.insert(allFields.end(), optionalFields.begin(), optionalFields.end());

    for (auto& [key, value] : templateGroup.items()) {
        if (std::find(allFields.begin(), allFields.end(), key) == allFields.end()) {
            std::cout << "unknown field \"" << key << "\" on templateGroup object" << std::endl;
            return false;
        }
    }

    for (const auto& field : requiredFields) {
        if (!templateGroup.contains(field)) {
            std::cout << "missing required field \"" << field << "\" on templateGroup object" << std::endl;
            return false;
        }
    }

    if (!templateGroup["name"].is_string()) {
        std::cout << "templateGroup name must be a string" << std::endl;
        return false;
    }
    if (m_templates.find(templateGroup["name"].get<std::string>()) != m_templates.end()) {
        std::cout << "templateGroups cannot have the same name" << std::endl;
        return false;
    }

    SceneNode *templateNode = new SceneNode;
    m_nodes.push_back(templateNode);
    m_templates[templateGroup["name"].get<std::string>()] = templateNode;

    return parseGroupData(templateGroup, templateNode);
}

bool ScenefileReader::parseGroupData(const json& object, SceneNode *node) {
    std::vector<std::string> optionalFields = {"name", "translate", "rotate", "scale", "matrix", "lights", "primitives", "groups"};
    std::vector<std::string> allFields = optionalFields;

    for (auto& [key, value] : object.items()) {
        if (std::find(allFields.begin(), allFields.end(), key) == allFields.end()) {
            std::cout << "unknown field \"" << key << "\" on group object" << std::endl;
            return false;
        }
    }

    // parse translation if defined
    if (object.contains("translate")) {
        if (!object["translate"].is_array()) {
            std::cout << "group translate must be of type array" << std::endl;
            return false;
        }

        auto translateArray = object["translate"].get<std::vector<float>>();
        if (translateArray.size() != 3) {
            std::cout << "group translate must have 3 elements" << std::endl;
            return false;
        }

        SceneTransformation *translation = new SceneTransformation();
        translation->type = TransformationType::TRANSFORMATION_TRANSLATE;
        translation->translate[0] = translateArray[0];
        translation->translate[1] = translateArray[1];
        translation->translate[2] = translateArray[2];

        node->transformations.push_back(translation);
    }

    // parse rotation if defined
    if (object.contains("rotate")) {
        if (!object["rotate"].is_array()) {
            std::cout << "group rotate must be of type array" << std::endl;
            return false;
        }

        auto rotateArray = object["rotate"].get<std::vector<float>>();
        if (rotateArray.size() != 4) {
            std::cout << "group rotate must have 4 elements" << std::endl;
            return false;
        }

        SceneTransformation *rotation = new SceneTransformation();
        rotation->type = TransformationType::TRANSFORMATION_ROTATE;
        rotation->rotate[0] = rotateArray[0];
        rotation->rotate[1] = rotateArray[1];
        rotation->rotate[2] = rotateArray[2];
        rotation->angle = rotateArray[3] * M_PI / 180.f;

        node->transformations.push_back(rotation);
    }

    // parse scale if defined
    if (object.contains("scale")) {
        if (!object["scale"].is_array()) {
            std::cout << "group scale must be of type array" << std::endl;
            return false;
        }

        auto scaleArray = object["scale"].get<std::vector<float>>();
        if (scaleArray.size() != 3) {
            std::cout << "group scale must have 3 elements" << std::endl;
            return false;
        }

        SceneTransformation *scale = new SceneTransformation();
        scale->type = TransformationType::TRANSFORMATION_SCALE;
        scale->scale[0] = scaleArray[0];
        scale->scale[1] = scaleArray[1];
        scale->scale[2] = scaleArray[2];

        node->transformations.push_back(scale);
    }

    // parse matrix if defined
    if (object.contains("matrix")) {
        if (!object["matrix"].is_array()) {
            std::cout << "group matrix must be of type array of array" << std::endl;
            return false;
        }

        auto matrixArray = object["matrix"].get<std::vector<std::vector<float>>>();
        if (matrixArray.size() != 4) {
            std::cout << "group matrix must be 4x4" << std::endl;
            return false;
        }

        SceneTransformation *matrixTransformation = new SceneTransformation();
        matrixTransformation->type = TransformationType::TRANSFORMATION_MATRIX;

        float *matrixPtr = (matrixTransformation->matrix).data();
        for (int i = 0; i < 4; i++) {
            if (matrixArray[i].size() != 4) {
                std::cout << "group matrix must be 4x4" << std::endl;
                return false;
            }
            for (int j = 0; j < 4; j++) {
                matrixPtr[j * 4 + i] = matrixArray[i][j];
            }
        }

        node->transformations.push_back(matrixTransformation);
    }

    // parse lights if any
    if (object.contains("lights")) {
        if (!object["lights"].is_array()) {
            std::cout << "group lights must be of type array" << std::endl;
            return false;
        }
        for (const auto& light : object["lights"]) {
            if (!light.is_object()) {
                std::cout << "light must be of type object" << std::endl;
                return false;
            }

            if (!parseLightData(light, node)) {
                return false;
            }
        }
    }

    // parse primitives if any
    if (object.contains("primitives")) {
        if (!object["primitives"].is_array()) {
            std::cout << "group primitives must be of type array" << std::endl;
            return false;
        }
        for (const auto& primitive : object["primitives"]) {
            if (!primitive.is_object()) {
                std::cout << "primitive must be of type object" << std::endl;
                return false;
            }

            if (!parsePrimitive(primitive, node)) {
                return false;
            }
        }
    }

    // parse children groups if any
    if (object.contains("groups")) {
        if (!parseGroups(object["groups"], node)) {
            return false;
        }
    }

    return true;
}

bool ScenefileReader::parseGroups(const json& groups, SceneNode *parent) {
    if (!groups.is_array()) {
        std::cout << "groups must be of type array" << std::endl;
        return false;
    }

    for (const auto& group : groups) {
        if (!group.is_object()) {
            std::cout << "group items must be of type object" << std::endl;
            return false;
        }

        if (group.contains("name")) {
            if (!group["name"].is_string()) {
                std::cout << "group name must be of type string" << std::endl;
                return false;
            }

            // if its a reference to a template group append it
            std::string groupName = group["name"].get<std::string>();
            if (m_templates.find(groupName) != m_templates.end()) {
                parent->children.push_back(m_templates[groupName]);
                continue;
            }
        }

        SceneNode *node = new SceneNode;
        m_nodes.push_back(node);
        parent->children.push_back(node);

        if (!parseGroupData(group, node)) {
            return false;
        }
    }

    return true;
}

bool ScenefileReader::parsePrimitive(const json& prim, SceneNode *node) {
    std::vector<std::string> requiredFields = {"type"};
    std::vector<std::string> optionalFields = {
        "meshFile", "ambient", "diffuse", "specular", "reflective", "transparent", "shininess", "ior",
        "blend", "textureFile", "textureU", "textureV", "bumpMapFile", "bumpMapU", "bumpMapV"
    };

    std::vector<std::string> allFields = requiredFields;
    allFields.insert(allFields.end(), optionalFields.begin(), optionalFields.end());

    for (const auto& [key, value] : prim.items()) {
        if (std::find(allFields.begin(), allFields.end(), key) == allFields.end()) {
            std::cout << "unknown field \"" << key << "\" on primitive object" << std::endl;
            return false;
        }
    }
    for (const auto& field : requiredFields) {
        if (!prim.contains(field)) {
            std::cout << "missing required field \"" << field << "\" on primitive object" << std::endl;
            return false;
        }
    }

    if (!prim["type"].is_string()) {
        std::cout << "primitive type must be of type string" << std::endl;
        return false;
    }
    std::string primType = prim["type"].get<std::string>();

    // Default primitive
    ScenePrimitive *primitive;

    std::filesystem::path basepath = std::filesystem::path(file_name).parent_path().parent_path();
    if (primType == "sphere") {
        primitive = new ScenePrimitive(distToSphere);
        primitive->type = PrimitiveType::PRIMITIVE_SPHERE;
    } 
    else if (primType == "cube") {
        primitive = new ScenePrimitive(distToCube);
        primitive->type = PrimitiveType::PRIMITIVE_CUBE;
    }   
    else if (primType == "cylinder") {
        primitive = new ScenePrimitive(distToCylinder);
        primitive->type = PrimitiveType::PRIMITIVE_CYLINDER;
    }
    else if (primType == "cone") {
        primitive = new ScenePrimitive(distToCone);
        primitive->type = PrimitiveType::PRIMITIVE_CONE;
    } 
    else if (primType == "mesh") {
        primitive = new ScenePrimitive(nullptr);
        primitive->type = PrimitiveType::PRIMITIVE_MESH;
        std::cout << "SDF for Mesh Primitive Needs to Be Implemented" << std::endl;
        return false;
        if (!prim.contains("meshFile")) {
            std::cout << "primitive type mesh must contain field meshFile" << std::endl;
            return false;
        }
        if (!prim["meshFile"].is_string()) {
            std::cout << "primitive meshFile must be of type string" << std::endl;
            return false;
        }

        std::filesystem::path relativePath(prim["meshFile"].get<std::string>());
        primitive->meshfile = (basepath / relativePath).string();
    }
    else {
        std::cout << "unknown primitive type \"" << primType << "\"" << std::endl;
        return false;
    }

    SceneMaterial &mat = primitive->material;
    mat.clear();
    primitive->type = PrimitiveType::PRIMITIVE_CUBE;
    mat.textureMap.isUsed = false;
    mat.bumpMap.isUsed = false;
    mat.cDiffuse[0] = mat.cDiffuse[2] = mat.cDiffuse[3] = 1;
    node->primitives.push_back(primitive);

    if (prim.contains("ambient")) {
        if (!prim["ambient"].is_array()) {
            std::cout << "primitive ambient must be of type array" << std::endl;
            return false;
        }
        auto ambientArray = prim["ambient"].get<std::vector<float>>();
        if (ambientArray.size() != 3) {
            std::cout << "primitive ambient array must be of size 3" << std::endl;
            return false;
        }

        for (int i = 0; i < 3; i++) {
            mat.cAmbient[i] = ambientArray[i];
        }
    }

    if (prim.contains("diffuse")) {
        if (!prim["diffuse"].is_array()) {
            std::cout << "primitive diffuse must be of type array" << std::endl;
            return false;
        }
        auto diffuseArray = prim["diffuse"].get<std::vector<float>>();
        if (diffuseArray.size() != 3) {
            std::cout << "primitive diffuse array must be of size 3" << std::endl;
            return false;
        }

        for (int i = 0; i < 3; i++) {
            mat.cDiffuse[i] = diffuseArray[i];
        }
    }

    if (prim.contains("specular")) {
        if (!prim["specular"].is_array()) {
            std::cout << "primitive specular must be of type array" << std::endl;
            return false;
        }
        auto specularArray = prim["specular"].get<std::vector<float>>();
        if (specularArray.size() != 3) {
            std::cout << "primitive specular array must be of size 3" << std::endl;
            return false;
        }

        for (int i = 0; i < 3; i++) {
            mat.cSpecular[i] = specularArray[i];
        }
    }

    if (prim.contains("reflective")) {
        if (!prim["reflective"].is_array()) {
            std::cout << "primitive reflective must be of type array" << std::endl;
            return false;
        }
        auto reflectiveArray = prim["reflective"].get<std::vector<float>>();
        if (reflectiveArray.size() != 3) {
            std::cout << "primitive reflective array must be of size 3" << std::endl;
            return false;
        }

        for (int i = 0; i < 3; i++) {
            mat.cReflective[i] = reflectiveArray[i];
        }
    }

    if (prim.contains("transparent")) {
        if (!prim["transparent"].is_array()) {
            std::cout << "primitive transparent must be of type array" << std::endl;
            return false;
        }
        auto transparentArray = prim["transparent"].get<std::vector<float>>();
        if (transparentArray.size() != 3) {
            std::cout << "primitive transparent array must be of size 3" << std::endl;
            return false;
        }

        for (int i = 0; i < 3; i++) {
            mat.cTransparent[i] = transparentArray[i];
        }
    }

    if (prim.contains("shininess")) {
        if (!prim["shininess"].is_number()) {
            std::cout << "primitive shininess must be of type float" << std::endl;
            return false;
        }

        mat.shininess = prim["shininess"].get<float>();
    }

    if (prim.contains("ior")) {
        if (!prim["ior"].is_number()) {
            std::cout << "primitive ior must be of type float" << std::endl;
            return false;
        }

        mat.ior = prim["ior"].get<float>();
    }

    if (prim.contains("blend")) {
        if (!prim["blend"].is_number()) {
            std::cout << "primitive blend must be of type float" << std::endl;
            return false;
        }

        mat.blend = prim["blend"].get<float>();
    }

    if (prim.contains("textureFile")) {
        if (!prim["textureFile"].is_string()) {
            std::cout << "primitive textureFile must be of type string" << std::endl;
            return false;
        }
        std::filesystem::path fileRelativePath(prim["textureFile"].get<std::string>());

        mat.textureMap.filename = (basepath / fileRelativePath).string();
        mat.textureMap.repeatU = prim.contains("textureU") && prim["textureU"].is_number() ? prim["textureU"].get<float>() : 1;
        mat.textureMap.repeatV = prim.contains("textureV") && prim["textureV"].is_number() ? prim["textureV"].get<float>() : 1;
        mat.textureMap.isUsed = true;
    }

    if (prim.contains("bumpMapFile")) {
        if (!prim["bumpMapFile"].is_string()) {
            std::cout << "primitive bumpMapFile must be of type string" << std::endl;
            return false;
        }
        std::filesystem::path fileRelativePath(prim["bumpMapFile"].get<std::string>());

        mat.bumpMap.filename = (basepath / fileRelativePath).string();
        mat.bumpMap.repeatU = prim.contains("bumpMapU") && prim["bumpMapU"].is_number() ? prim["bumpMapU"].get<float>() : 1;
        mat.bumpMap.repeatV = prim.contains("bumpMapV") && prim["bumpMapV"].is_number() ? prim["bumpMapV"].get<float>() : 1;
        mat.bumpMap.isUsed = true;
    }

    return true;
}