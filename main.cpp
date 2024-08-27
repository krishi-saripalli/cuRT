//To build, cd into build dir and run cmake -DCMAKE_PREFIX_PATH=/Users/saripallikrishi/Qt/6.2.4/macos ..
#include <iostream>
#include <Eigen/Dense>


#include <QApplication>
#include <QCommandLineParser>
#include <QImage>
#include <QtCore>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>

#include <iostream>
#include "utils/sceneparser.h"
#include "utils/rgba.h"
#include "window/mainwindow.h"
#include "raymarcher/raymarcher.h"


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addPositionalArgument("config", "Path of the config file.");
    parser.process(a);

    auto positionalArgs = parser.positionalArguments();
    if (positionalArgs.size() != 1) {
        std::cerr << "Not enough arguments. Please provide a path to a config file (.json) as a command-line argument." << std::endl;
        a.exit(1);
        return 1;
    }

    QString configFilePath = positionalArgs[0];
    QFile file(configFilePath);
    if (!file.open(QIODevice::ReadOnly)) {
        std::cerr << "Could not open config file." << std::endl;
        a.exit(1);
        return 1;
    }

    QByteArray fileData = file.readAll();
    QJsonDocument jsonDoc(QJsonDocument::fromJson(fileData));

    if (jsonDoc.isNull()) {
        std::cerr << "Failed to parse JSON." << std::endl;
        a.exit(1);
        return 1;
    }

    QJsonObject jsonObj = jsonDoc.object();
    
    // Check if "IO" object exists
    if (!jsonObj.contains("IO") || !jsonObj["IO"].isObject()) {
        std::cerr << "JSON is missing 'IO' object." << std::endl;
        a.exit(1);
        return 1;
    }

    QJsonObject ioObj = jsonObj["IO"].toObject();

    // Check if required fields exist
    if (!ioObj.contains("scene") || !ioObj.contains("output")) {
        std::cerr << "JSON is missing required fields in 'IO' object." << std::endl;
        a.exit(1);
        return 1;
    }

    QString iScenePath = ioObj["scene"].toString();
    QString oImagePath = ioObj["output"].toString();


    // Check if "Canvas" object exists
    if (!jsonObj.contains("Canvas") || !jsonObj["Canvas"].isObject()) {
        std::cerr << "JSON is missing 'Canvas' object." << std::endl;
        a.exit(1);
        return 1;
    }

    QJsonObject canvasObj = jsonObj["Canvas"].toObject();

    // Check if required fields exist
    if (!canvasObj.contains("width") || !canvasObj.contains("height")) {
        std::cerr << "JSON is missing required fields in 'Canvas' object." << std::endl;
        a.exit(1);
        return 1;
    }

    int width = canvasObj["width"].toInt();
    int height = canvasObj["height"].toInt();

    RenderData metaData;
    bool success = SceneParser::parse(iScenePath.toStdString(), metaData);

    if (!success) {
        std::cerr << "Error loading scene: \"" << iScenePath.toStdString() << "\"" << std::endl;
        a.exit(1);
        return 1;
    }

    MainWindow window(width, height);
    
    window.show();

    // Extracting data pointer from Qt's image API
    QImage image = QImage(width, height, QImage::Format_RGBX8888);
    image.fill(Qt::black);
    RGBA *data = reinterpret_cast<RGBA *>(image.bits());

    // // Setting up the raytracer
    // RayTracer::Config rtConfig{};
    // rtConfig.enableShadow        = settings.value("Feature/shadows").toBool();
    // rtConfig.enableReflection    = settings.value("Feature/reflect").toBool();
    // rtConfig.enableRefraction    = settings.value("Feature/refract").toBool();
    // rtConfig.enableTextureMap    = settings.value("Feature/texture").toBool();
    // rtConfig.enableTextureFilter = settings.value("Feature/texture-filter").toBool();
    // rtConfig.enableParallelism   = settings.value("Feature/parallel").toBool();
    // rtConfig.enableSuperSample   = settings.value("Feature/super-sample").toBool();
    // rtConfig.enableAcceleration  = settings.value("Feature/acceleration").toBool();
    // rtConfig.enableDepthOfField  = settings.value("Feature/depthoffield").toBool();

    Raymarcher raymarcher{};

    const Scene scene{ width, height, metaData };

    // // Note that we're passing `data` as a pointer (to its first element)
    // // Recall from Lab 1 that you can access its elements like this: `data[i]`
    raymarcher.render(scene, window, data);

    // // Saving the image
    success = image.save(oImagePath);
    if (!success) {
        success = image.save(oImagePath, "PNG");
    }
    if (success) {
        std::cout << "Saved rendered image to \"" << oImagePath.toStdString() << "\"" << std::endl;
    } else {
        std::cerr << "Error: failed to save image to \"" << oImagePath.toStdString() << "\"" << std::endl;
    }

    // a.exit();
    return a.exec();
}