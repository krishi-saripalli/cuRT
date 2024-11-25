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
                total += pointIllumination(renderData,hit,light);
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
    vec4 v = -hit.direction;  
    v.normalize();

    vec4 ri = (-li) - (2.f * dot(-li, hit.normal) * hit.normal);
    ri.normalize();

    vec4 specular = ks * powf(max(0.0f, dot(ri, v)), n) * os;


    return ambient + (intensity * (diffuse + specular));
  
}

__device__ vec4 pointIllumination(const GPURenderData& renderData, const Hit& hit, const GPUSceneLightData& light) {
    float ka = renderData.globalData.ka;
    vec4 oa = hit.shape->primitive.material.cAmbient;

    vec4 ambient = ka * oa;

    vec4 intensity = light.color;
    float kd = renderData.globalData.kd;
    vec4 od = hit.shape->primitive.material.cDiffuse;
    vec4 li = light.pos - hit.intersection;
    li.normalize(); // TODO: do this once on CPU, not here
    vec4 diffuse = kd * od * max(0.0f, dot(hit.normal,li));

    
    float ks = renderData.globalData.ks;
    float n = hit.shape->primitive.material.shininess;
    vec4 os = hit.shape->primitive.material.cSpecular;
    vec4 v = -hit.direction; 
    v.normalize();

    vec4 ri = (-li) - (2.f * dot(-li, hit.normal) * hit.normal);
    ri.normalize();

    vec4 specular = ks * powf(max(0.0f, dot(ri, v)), n) * os;

    //attenuation
    vec3 c = light.function;
    float distance = li.length();
    float att = fmin(1.f, 1.f/(c[0] + (distance*c[1]) + (distance*distance*c[2])));


    return ambient + ( att * intensity * (diffuse + specular));
  
}


