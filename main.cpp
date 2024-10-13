//To build, cd into build dir and run cmake ..
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <stdexcept>
#include <Eigen/Dense>
#include <utils/json.hpp>


#include "utils/sceneparser.h"
#include "utils/rgba.h"
#include "raymarcher/scene.h"
#include "raymarcher/raymarcher.h"



using json = nlohmann::json;

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "Not enough arguments. Please provide a path to a config file (.json) as a command-line argument." << std::endl;
        return 1;
    }

    std::string configFilePath = argv[1];
    std::ifstream file(configFilePath);
    if (!file.is_open()) {
        std::cerr << "Could not open config file." << std::endl;
        return 1;
    }

    json jsonObj;
    try {
        file >> jsonObj;
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "Failed to parse JSON: " << e.what() << std::endl;
        return 1;
    }

    // Check if "IO" object exists
    if (!jsonObj.contains("IO") || !jsonObj["IO"].is_object()) {
        std::cerr << "JSON is missing 'IO' object." << std::endl;
        return 1;
    }

    const auto& ioObj = jsonObj["IO"];

    // Check if required fields exist
    if (!ioObj.contains("scene") || !ioObj.contains("output")) {
        std::cerr << "JSON is missing required fields in 'IO' object." << std::endl;
        return 1;
    }

    std::string iScenePath = ioObj["scene"];
    std::string oImagePath = ioObj["output"];

    // Check if "Canvas" object exists
    if (!jsonObj.contains("Canvas") || !jsonObj["Canvas"].is_object()) {
        std::cerr << "JSON is missing 'Canvas' object." << std::endl;
        return 1;
    }

    const auto& canvasObj = jsonObj["Canvas"];

    // Check if required fields exist
    if (!canvasObj.contains("width") || !canvasObj.contains("height")) {
        std::cerr << "JSON is missing required fields in 'Canvas' object." << std::endl;
        return 1;
    }

    int width = canvasObj["width"];
    int height = canvasObj["height"];

    RenderData metaData;
    bool success = SceneParser::parse(iScenePath, metaData);


    if (!success) {
        std::cerr << "Error loading scene: \"" << iScenePath << "\"" << std::endl;
        return 1;
    }


    std::unique_ptr<Window> window(new Window(width, height, "THE CLAW"));
    Raymarcher raymarcher{std::move(window)};
    const Scene scene{width, height, metaData};

    TextureQuad quad = setupTextureDisplayQuad(scene.getCamera().getAspectRatio(width,height));
    auto [texture, pbo] = createTexture(width,height,4);

    GLuint pinkQuadShader = createPinkQuadShader();
    raymarcher.setShader(pinkQuadShader);
    raymarcher.setTextureQuad(quad);


    try {
        raymarcher.run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    glDeleteVertexArrays(1, &(quad.vao));
    glDeleteBuffers(1, &(quad.vbo));
    glDeleteBuffers(1, &(quad.vboTexture));
    glDeleteBuffers(1, &(quad.ebo));
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texture);
    glDeleteProgram(pinkQuadShader);

    GET_GL_ERROR("Free() ERROR");

    std::cout << "Finished" << std::endl;

    return 0;
}