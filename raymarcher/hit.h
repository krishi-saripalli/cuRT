#pragma once
#include "../utils/sceneparser.h"

struct Hit {
    const RenderShapeData* shapeData;
    float distance;
};