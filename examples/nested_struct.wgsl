// caso: nested struct
struct Light {
    color: vec3<f32>,
    intensity: f32,
}
struct Scene {
    ambient: vec3<f32>,
    light: Light,
}
@group(0) @binding(0) var<uniform> scene: Scene;

// caso: array de tamanho fixo
struct Lights {
    items: array<Light, 4>,
}
@group(0) @binding(1) var<storage, read> lights: Lights;

// caso: array dinâmico (storage buffer)
struct Particles {
    data: array<vec4<f32>>,
}
@group(0) @binding(2) var<storage, read_write> particles: Particles;

// caso: múltiplos entry points, texturas/samplers
@group(1) @binding(0) var t: texture_2d<f32>;
@group(1) @binding(1) var s: sampler;
