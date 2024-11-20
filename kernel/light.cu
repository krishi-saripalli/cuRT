#include "light.cuh"


__device__ vec4 illumination(const GPURenderData& renderData, const Hit& hit) {

    vec4 total = vec4(0.f,0.f,0.f,0.f);
    for (int i = 0; i < renderData.numLights; ++i) {
        GPUSceneLightData light = renderData.lights[i];
        switch (light.type) {
            case GPULightType::LIGHT_DIRECTIONAL:
                total += directionalIllumination(renderData,hit,light);
                break;
            case GPULightType::LIGHT_POINT:
                break;
            case GPULightType::LIGHT_SPOT:
                break;
            default:
                break;
        }

    }
    return total;
}

__device__ vec4 directionalIllumination(const GPURenderData& renderData, const Hit& hit, const GPUSceneLightData& light) {
    float ka = renderData.globalData.ka;
    vec4 oa = hit.shape->primitive.material.cAmbient;

    vec4 ambient = ka * oa;

    vec4 intensity = light.color;
    float kd = renderData.globalData.kd;
    vec4 od = hit.shape->primitive.material.cDiffuse;
    vec4 li = (-light.dir);
    li.normalize(); // TODO: do this once on CPU, not here
    vec4 diffuse = kd * od * max(0.0f, dot(hit.normal,li));

    float ks = renderData.globalData.ks;
    float n = hit.shape->primitive.material.shininess;
    vec4 os = hit.shape->primitive.material.cSpecular;
    // Normalize vectors
    vec4 v = -hit.direction;  // View vector
    v.normalize();

    // Calculate reflection vector (using negated light direction to match li)
    vec4 ri = (-li) - (2.f * dot(-li, hit.normal) * hit.normal);
    ri.normalize();

    vec4 specular = ks * powf(max(0.0f, dot(ri, v)), n) * os;
    //print(hit.direction);


    return ambient + (intensity * (diffuse + specular));
  
}