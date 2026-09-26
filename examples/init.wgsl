struct Camera {
    position: vec4<f32>,
    rot: mat4x4<f32>,
}

@vertex
fn main_vs(@location(0) camera: Camera) -> @builtin(position) vec4<f32> {
    return camera.position;
}

@fragment
fn main_fs() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
