#pragma once

#include "scenedata.h"
#include "utils/json.hpp"



#include <vector>
#include <map>
#include <QJsonDocument>
#include <QJsonObject>

using json = nlohmann::json;

// This class parses the scene graph specified by the CS123 Xml file format.
class ScenefileReader {
public:
    // Create a ScenefileReader, passing it the scene file.
    ScenefileReader(const std::string &filename);

    // Clean up all data for the scene
    ~ScenefileReader();

    // Parse the JSON scene file. Returns false if scene is invalid.
    bool readJSON();

    SceneGlobalData getGlobalData() const;

    SceneCameraData getCameraData() const;

    SceneNode *getRootNode() const;

private:
    // The filename should be contained within this parser implementation.
    // If you want to parse a new file, instantiate a different parser.
    bool parseGlobalData(const json& globaldata);
    bool parseCameraData(const json& cameradata);
    bool parseTemplateGroups(const json& templateGroups);
    bool parseTemplateGroupData(const json& templateGroup);
    bool parseGroups(const json& groups, SceneNode *parent);
    bool parseGroupData(const json& object, SceneNode *node);
    bool parsePrimitive(const json& prim, SceneNode *node);
    bool parseLightData(const json& lightData, SceneNode *node);

    std::string file_name;

    mutable std::map<std::string, SceneNode *> m_templates;

    SceneGlobalData m_globalData;
    SceneCameraData m_cameraData;

    SceneNode *m_root;
    std::vector<SceneNode *> m_nodes;
};