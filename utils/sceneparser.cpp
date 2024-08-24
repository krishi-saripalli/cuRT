#include "sceneparser.h"
#include "scenefilereader.h"

#include <chrono>
#include <iostream>

Eigen::Matrix4f calculateCTM(std::vector<SceneTransformation*>& transformations, Eigen::Matrix4f old_ctm) {

    Eigen::Matrix4f new_ctm = old_ctm;
    
    for (SceneTransformation* transformation : transformations) {
        if (transformation->type == TransformationType::TRANSFORMATION_SCALE) {
            Eigen::Matrix4f scaleMatrix = Eigen::Matrix4f::Identity();
            scaleMatrix.block<3,3>(0,0) = Eigen::Scaling(transformation->scale[0], transformation->scale[1], transformation->scale[2]);
            new_ctm = new_ctm * scaleMatrix;

        }
        else if (transformation->type == TransformationType::TRANSFORMATION_ROTATE) {
            Eigen::AngleAxisf rotation(transformation->angle, transformation->rotate);
            Eigen::Matrix4f rotationMatrix = Eigen::Matrix4f::Identity();
            rotationMatrix.block<3,3>(0,0) = rotation.matrix();
            new_ctm = new_ctm * rotationMatrix;
        }
        else if (transformation->type == TransformationType::TRANSFORMATION_TRANSLATE) {
            Eigen::Matrix4f translateMatrix = Eigen::Affine3f(Eigen::Translation3f(transformation->translate)).matrix();
            new_ctm = new_ctm * translateMatrix;
        }
        else {
            new_ctm = new_ctm * transformation->matrix;
        }

    }
    return new_ctm;
}

void dfs(RenderData &renderData, SceneNode* node, Eigen::Matrix4f ctm) {

    //Calculate new CTM
    Eigen::Matrix4f new_ctm = calculateCTM(node->transformations, ctm);

    //Construct RenderShapeData for each primitive
    for (ScenePrimitive* primitive : node->primitives) {

        RenderShapeData shape_data{ *primitive, new_ctm};
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

    std::cout << "Scene Successfully Parsed!" << std::endl;
    return true;
}