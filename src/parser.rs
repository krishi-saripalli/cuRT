use nalgebra_glm as glm;

pub struct GlobalData {
    pub ka: f32, //ambient
    pub kd: f32, //diffuse
    pub ks: f32, //specular
    pub kt: f32, //transparency
}
pub struct CameraData {
    pub pos: glm::Vec4,
    pub look: glm::Vec4,
    pub up: glm::Vec4,

    pub height_angle: f32,
    pub aspect_ratio: f32,
    pub focal_length: f32,
    pub aperture: f32,
}

enum LightType {
    LIGHT_POINT,
    LIGHT_DIRECTIONAL,
    LIGHT_SPOT,
    LIGHT_AREA,
}
pub struct LightData {
    pub id: i32,
    pub light_type: LightType,
    pub color: glm::Vec4,

    pub pos: Option<glm::Vec4>,
    pub dir: Option<glm::Vec4>,

    pub radius: Option<f32>,
    pub angle: Option<f32>,
    pub penumbra: Option<f32>,

    pub width: Option<f32>,
    height: Option<f32>,
}

enum TransformationType {
    TRANSFORMATION_TRANSLATE,
    TRANSFORMATION_SCALE,
    TRANSFORMATION_ROTATE,
    TRANSFORMATION_MATRIX,
}

pub struct TranformationData {
    pub tranformation_type: TransformationType,
    pub rotation: Option<glm::Vec3>,
    pub scale: Option<glm::Vec3>,
    pub translate: Option<glm::Vec3>,

    pub rotation_angle: Option<f32>,
}
