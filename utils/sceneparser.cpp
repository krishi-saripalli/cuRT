#include "sceneparser.h"
#include "scenefilereader.h"

#include <chrono>
#include <iostream>

Eigen::Matrix4f calculateCTM(std::vector<SceneTransformation*>& transformations, Eigen::Matrix4f& parentTransform) {

    Eigen::Matrix4f ctm = parentTransform; // Start with the parent transform
    
    for (const SceneTransformation* transformation : transformations) {
        Eigen::Matrix4f transformMatrix = Eigen::Matrix4f::Identity();
        
        switch (transformation->type) {
            case TransformationType::TRANSFORMATION_SCALE:
                transformMatrix.block<3,3>(0,0) = Eigen::Scaling(transformation->scale.x(), transformation->scale.y(), transformation->scale.z());
                break;
            case TransformationType::TRANSFORMATION_ROTATE:
                transformMatrix.block<3,3>(0,0) = Eigen::AngleAxisf(transformation->angle, transformation->rotate).matrix();
                break;
            case TransformationType::TRANSFORMATION_TRANSLATE:
                transformMatrix.block<3,1>(0,3) = transformation->translate;
                break;
            default:
                transformMatrix = transformation->matrix;
                break;
        }
        
        ctm = transformMatrix * ctm;
    }
    
    return ctm;
}

void dfs(RenderData &renderData, SceneNode* node, Eigen::Matrix4f ctm) {

    //Calculate new CTM
    Eigen::Matrix4f new_ctm = calculateCTM(node->transformations, ctm);

    //Construct RenderShapeData for each primitive
    for (ScenePrimitive* primitive : node->primitives) {

        RenderShapeData shape_data{ *primitive, new_ctm, new_ctm.inverse()};
        std::cout << "AT PUSH BACK: " << (int) primitive->type << std::endl;
        renderData.shapes.push_back(shape_data);
    }

    //Construct SceneLightData for each light
    for (SceneLight* light : node->lights) {
        SceneLightData light_data;
        light_data.id = light->id;
        light_data.type = light->type;
        light_data.color = light->color;
        light_data.function = light->function;
        light_data.dir = light->dir;
        light_data.pos = Eigen::Vector4f(0, 0, 0, 1); //TODO: Is this assumption correct?
        light_data.penumbra = light->penumbra;
        light_data.angle = light->angle;
        light_data.width = light->width;
        light_data.height = light->height;

        //Transform pos and dir to world space
        light_data.dir = new_ctm * light_data.dir;
        light_data.pos = new_ctm * light_data.pos;

        renderData.lights.push_back(light_data);
    }

    //Call the function recursivley on children
    for (SceneNode* child : node->children) {
        dfs(renderData,child,new_ctm);
    }

    return;
}

bool SceneParser::parse(std::string filepath, RenderData &renderData) {
    ScenefileReader fileReader = ScenefileReader(filepath);
    bool success = fileReader.readJSON();
    if (!success) {
        return false;
    }

    // Populate RenderData
    renderData.globalData = fileReader.getGlobalData();
    renderData.cameraData = fileReader.getCameraData();

    //Clear out shapes and lights vectors
    renderData.shapes.clear();
    renderData.lights.clear();

    // Get root node
    SceneNode* root = fileReader.getRootNode();

    //Recursivley parse the scene graph
    Eigen::Matrix4f ctm;
    ctm.setIdentity();
    dfs(renderData, root, ctm);

    std::cout << "AFTER DFS " << (int) renderData.shapes[0].primitive.type << std::endl;

    std::cout << "Scene Successfully Parsed!" << std::endl;
    return true;
}