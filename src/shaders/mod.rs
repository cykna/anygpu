pub mod image;
pub mod link;
pub mod pipeline;
pub mod types;
pub mod wesl;
use naga::{Handle, Module, Type, TypeInner, proc::Layouter};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::shaders::{
    pipeline::types::{BindGroupLayoutEntry, PipelineDescriptor},
    types::{DeclarationIdentity, TypeInfo, TypeInterner},
};
#[derive(Debug, Serialize, Deserialize)]
pub struct SharedTypes {
    #[serde(default)]
    pub types: Vec<TypeInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ShaderBindings {
    #[serde(default)]
    pub types: Vec<TypeInfo>,
    #[serde(default)]
    pub shared: Option<SharedTypes>,
    /// The pipeline this shader needs, or `None` when the schema carries types
    /// only. A generator emits whatever a shader's consumers need, so a missing
    /// pipeline is not an error: it only means there is nothing to describe one
    /// with.
    #[serde(default, alias = "pipeline_descriptor")]
    pub pipelines: Option<PipelineDescriptor>,
    #[serde(default)]
    pub bindgroups: Vec<BindGroupLayoutEntry>,
}

#[derive(Debug)]
pub struct ReflectedModule {
    pub id: String,
    pub bindings: ShaderBindings,
    pub identities: std::collections::BTreeMap<crate::TypeId, DeclarationIdentity>,
}

#[derive(Debug)]
pub struct ShaderMetadata<'a> {
    external_types: HashSet<Handle<Type>>,
    module: &'a Module,
    module_info: Option<naga::valid::ModuleInfo>,
    layouter: Layouter,
}
impl<'a> ShaderMetadata<'a> {
    pub fn get_external_types(module: &Module) -> HashSet<Handle<Type>> {
        let mut types = HashSet::new();
        let mut stack = Vec::new();
        for (_, var) in module.global_variables.iter() {
            if var.binding.is_some() {
                stack.push(var.ty);
            }
        }
        for entry_point in &module.entry_points {
            for arg in &entry_point.function.arguments {
                stack.push(arg.ty);
            }
            if let Some(result) = &entry_point.function.result {
                stack.push(result.ty);
            }
        }

        while let Some(handle) = stack.pop() {
            if !types.insert(handle) {
                continue; // já visitado
            }
            match &module.types[handle].inner {
                TypeInner::Struct { members, .. } => {
                    for m in members {
                        stack.push(m.ty);
                    }
                }
                TypeInner::Array { base, .. }
                | TypeInner::Pointer { base, .. }
                | TypeInner::BindingArray { base, .. } => {
                    stack.push(*base);
                }
                _ => {} // escalares, vetores, matrizes: sem filhos
            }
        }
        types
    }

    pub fn new(module: &'a Module) -> color_eyre::Result<Self> {
        let mut layouter = Layouter::default();
        layouter.update(module.to_ctx())?;
        // Validation is where naga records which globals each entry point (and
        // everything it calls) actually uses, so it is also where visibility comes
        // from. Some shaders parse but fail interface validation (for example an
        // input type WebGPU cannot fetch); those still get a best-effort IR walk
        // rather than losing all of their reflection.
        let module_info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .subgroup_stages(naga::valid::ShaderStages::all())
        .subgroup_operations(naga::valid::SubgroupOperationSet::all())
        .validate(module)
        .ok();
        Ok(Self {
            external_types: Self::get_external_types(module),
            module,
            module_info,
            layouter,
        })
    }

    /// Every type reachable from a binding or an entry point argument, defined
    /// once each and in dependency order: a type is always listed after
    /// everything it is built from.
    fn get_types_with_ids(&self) -> (Vec<TypeInfo>, HashMap<Handle<Type>, crate::TypeId>) {
        let mut interner = TypeInterner::new();
        let mut memo = HashMap::new();
        // `external_types` is a set, so it is walked in handle order to keep the
        // ids — and therefore the schema — identical from one run to the next.
        let mut roots: Vec<Handle<Type>> = self.external_types.iter().copied().collect();
        roots.sort_by_key(|handle| handle.index());
        for handle in roots {
            self.type_id(&mut interner, &mut memo, handle);
        }
        (interner.finish(), memo)
    }

    pub fn get_types(&self) -> Vec<TypeInfo> {
        self.get_types_with_ids().0
    }

    pub fn generate_bindings(self) -> color_eyre::Result<ShaderBindings> {
        let (types, type_ids) = self.get_types_with_ids();
        // The types intrinsic to WGSL itself - vectors, scalars, matrices - are
        // shared by every shader, so they are described once, under the shared
        // section, while the rest are the types this shader declares for itself.
        let mut shared_types = Vec::new();
        let mut shader_types = Vec::new();
        for ty in types {
            if ty.descriptor.is_shared() {
                shared_types.push(ty);
            } else {
                shader_types.push(ty);
            }
        }
        let pipelines = self.get_pipeline_with_ids(&type_ids)?;
        Ok(ShaderBindings {
            types: shader_types,
            shared: Some(SharedTypes {
                types: shared_types,
            }),
            pipelines: Some(pipelines),
            bindgroups: Vec::new(),
        })
    }

    pub fn generate_reflected_module(
        self,
        id: String,
        identities: HashMap<Handle<Type>, DeclarationIdentity>,
    ) -> color_eyre::Result<ReflectedModule> {
        let (_, type_ids) = self.get_types_with_ids();
        let identities = identities
            .into_iter()
            .filter_map(|(handle, identity)| {
                type_ids
                    .get(&handle)
                    .map(|type_id| (type_id.clone(), identity))
            })
            .collect();
        let bindings = self.generate_bindings()?;

        Ok(ReflectedModule {
            id,
            bindings,
            identities,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shaders::types::TypeDescriptor;

    const CAMERA_WGSL: &str = r"
struct Camera {
    position: vec4<f32>,
    rot: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
";

    const TRIANGLE_WGSL: &str = r"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 1.0);
    return output;
}
";

    const PACKAGE_GLOBALS_WGSL: &str = r"
struct package_globals_Camera {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    inverse_view_projection: mat4x4<f32>,
}

struct package_globals_MeshData {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera: package_globals_Camera;
@group(1) @binding(0) var<storage, read> instances: array<package_globals_MeshData>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

fn camera_origin() -> vec4<f32> {
    return camera.view * vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let mesh = instances[0];
    output.position = camera.view_projection * vec4<f32>(input.position, 1.0)
        + mesh.model[0]
        + camera_origin();
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
";

    const MULTI_STAGE_WGSL: &str = r"
struct Data {
    value: vec4<f32>,
}

@group(0) @binding(0) var<uniform> shared_data: Data;
@group(0) @binding(1) var<storage, read_write> writable: array<Data>;
@group(1) @binding(0) var<uniform> unused: Data;

fn read_shared() -> vec4<f32> {
    return shared_data.value;
}

@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return read_shared();
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    let value = read_shared();
    writable[0].value = value;
    return value;
}
";

    const INTERFACE_ATTRIBUTES_WGSL: &str = r"
struct Attributes {
    @builtin(position) @invariant position: vec4<f32>,
    @location(0) @interpolate(flat) id: u32,
    @location(1) @interpolate(linear, sample) uv: vec2<f32>,
}

@vertex
fn vs_main() -> Attributes {
    var output: Attributes;
    output.position = vec4<f32>(0.0);
    output.id = 0u;
    output.uv = vec2<f32>(0.0);
    return output;
}
";

    fn entry(
        layouts: &[crate::BindGroupLayout],
        group: u32,
        binding: u32,
    ) -> &crate::BindGroupLayoutEntry {
        layouts[group as usize]
            .entries
            .iter()
            .find(|entry| entry.binding == binding)
            .expect("the binding should be reflected")
    }

    #[test]
    fn bind_group_visibility_and_minimum_sizes_come_from_the_layouter() {
        let module =
            naga::front::wgsl::parse_str(PACKAGE_GLOBALS_WGSL).expect("the shader should parse");
        let bindings = ShaderMetadata::new(&module)
            .expect("the shader should expose bindings")
            .generate_bindings()
            .expect("reflection should succeed");
        let layouts = &bindings
            .pipelines
            .as_ref()
            .unwrap()
            .layout
            .bind_group_layouts;

        assert_eq!(layouts.len(), 2);
        let camera = entry(layouts, 0, 0);
        assert_eq!(camera.name, "camera");
        assert_eq!(camera.visibility, vec![crate::ShaderStage::Vertex]);
        match &camera.ty {
            crate::BindingType::Buffer {
                buffer_type: crate::BufferBindingType::Uniform,
                min_binding_size: Some(256),
                ..
            } => {}
            other => panic!("camera should be a 256-byte uniform buffer: {other:?}"),
        }

        let instances = entry(layouts, 1, 0);
        assert_eq!(instances.name, "instances");
        assert_eq!(instances.visibility, vec![crate::ShaderStage::Vertex]);
        match &instances.ty {
            crate::BindingType::Buffer {
                buffer_type:
                    crate::BufferBindingType::Storage {
                        read_only: crate::StorageAccess::Load,
                    },
                min_binding_size: Some(128),
                ..
            } => {}
            other => panic!("instances should be a 128-byte read-only storage array: {other:?}"),
        }

        let json = serde_json::to_value(&bindings).expect("the schema should serialize");
        assert_eq!(
            json["pipelines"]["layout"]["bind_group_layouts"][0]["entries"][0]["visibility"][0],
            "Vertex"
        );
        assert_eq!(
            json["pipelines"]["layout"]["bind_group_layouts"][0]["entries"][0]["ty"]["Buffer"]["min_binding_size"],
            256
        );
    }

    #[test]
    fn struct_layouts_and_runtime_arrays_match_the_shader() {
        let module =
            naga::front::wgsl::parse_str(PACKAGE_GLOBALS_WGSL).expect("the shader should parse");
        let bindings = ShaderMetadata::new(&module)
            .expect("the shader should expose bindings")
            .generate_bindings()
            .expect("reflection should succeed");
        let types = &bindings.types;

        let camera = types
            .iter()
            .find(|ty| ty.name == "package_globals_Camera")
            .expect("camera type");
        match &camera.descriptor {
            crate::TypeDescriptor::Struct {
                size, alignment, ..
            } => {
                assert_eq!(*size, 256);
                assert_eq!(*alignment, 16);
            }
            other => panic!("camera should be a struct: {other:?}"),
        }

        let mesh = types
            .iter()
            .find(|ty| ty.name == "package_globals_MeshData")
            .expect("mesh type");
        match &mesh.descriptor {
            crate::TypeDescriptor::Struct {
                size, alignment, ..
            } => {
                assert_eq!(*size, 128);
                assert_eq!(*alignment, 16);
            }
            other => panic!("mesh data should be a struct: {other:?}"),
        }

        let instances = types
            .iter()
            .find_map(|ty| match &ty.descriptor {
                crate::TypeDescriptor::Array {
                    base,
                    size: None,
                    stride,
                } if base == &mesh.id => Some((ty, *stride)),
                _ => None,
            })
            .expect("the runtime-sized instances array should be reflected");
        assert_eq!(instances.1, 128, "{}", instances.0.name);
    }

    #[test]
    fn interface_attributes_survive_reflection() {
        let module =
            naga::front::wgsl::parse_str(PACKAGE_GLOBALS_WGSL).expect("the shader should parse");
        let bindings = ShaderMetadata::new(&module)
            .expect("the shader should expose bindings")
            .generate_bindings()
            .expect("reflection should succeed");

        let output = bindings
            .types
            .iter()
            .find(|ty| ty.name == "VertexOutput")
            .expect("the vertex output type");
        let crate::TypeDescriptor::Struct { members, .. } = &output.descriptor else {
            panic!("VertexOutput should be a struct: {output:?}");
        };
        assert_eq!(members[0].name, "position");
        assert_eq!(members[0].builtin.as_deref(), Some("position"));
        assert_eq!(members[0].location, None);
        assert!(!members[0].invariant);
        assert_eq!(members[1].name, "color");
        assert_eq!(members[1].location, Some(0));
        assert_eq!(members[1].builtin, None);

        let pipeline = bindings.pipelines.as_ref().unwrap();
        let vertex = pipeline.vertex.as_ref().unwrap();
        assert_eq!(vertex.outputs.len(), 2);
        assert_eq!(vertex.outputs[0].builtin.as_deref(), Some("position"));
        assert_eq!(vertex.outputs[1].location, Some(0));

        let fragment = pipeline.fragment.as_ref().unwrap();
        assert_eq!(fragment.outputs.len(), 1);
        assert_eq!(fragment.outputs[0].location, Some(0));
        assert_eq!(fragment.inputs.len(), 2);
        assert_eq!(fragment.inputs[0].builtin.as_deref(), Some("position"));
        assert_eq!(fragment.inputs[1].location, Some(0));
    }

    #[test]
    fn interpolation_and_invariance_are_kept_on_members() {
        let module = naga::front::wgsl::parse_str(INTERFACE_ATTRIBUTES_WGSL)
            .expect("the shader should parse");
        let bindings = ShaderMetadata::new(&module)
            .expect("the shader should expose bindings")
            .generate_bindings()
            .expect("reflection should succeed");
        let output = bindings
            .types
            .iter()
            .find(|ty| ty.name == "Attributes")
            .expect("the attribute type");
        let crate::TypeDescriptor::Struct { members, .. } = &output.descriptor else {
            panic!("Attributes should be a struct: {output:?}");
        };

        assert_eq!(members[0].builtin.as_deref(), Some("position"));
        assert!(members[0].invariant);
        assert_eq!(members[1].interpolation, Some(crate::Interpolation::Flat));
        assert_eq!(members[1].sampling, None);
        assert_eq!(members[2].interpolation, Some(crate::Interpolation::Linear));
        assert_eq!(members[2].sampling, Some(crate::Sampling::Sample));
    }

    #[test]
    fn visibility_is_transitive_across_stages_and_unused_resources_stay_empty() {
        let module =
            naga::front::wgsl::parse_str(MULTI_STAGE_WGSL).expect("the shader should parse");
        let bindings = ShaderMetadata::new(&module)
            .expect("the shader should expose bindings")
            .generate_bindings()
            .expect("reflection should succeed");
        let layouts = &bindings
            .pipelines
            .as_ref()
            .unwrap()
            .layout
            .bind_group_layouts;

        let shared = entry(layouts, 0, 0);
        assert_eq!(
            shared.visibility,
            vec![crate::ShaderStage::Vertex, crate::ShaderStage::Fragment]
        );

        let writable = entry(layouts, 0, 1);
        assert_eq!(writable.visibility, vec![crate::ShaderStage::Fragment]);
        match &writable.ty {
            crate::BindingType::Buffer {
                buffer_type:
                    crate::BufferBindingType::Storage {
                        read_only: crate::StorageAccess::LoadStore,
                    },
                ..
            } => {}
            other => panic!("writable should be read-write storage: {other:?}"),
        }

        let unused = entry(layouts, 1, 0);
        assert!(unused.visibility.is_empty(), "{unused:?}");
    }

    #[test]
    fn invalid_interfaces_still_report_bind_group_visibility() {
        let module = naga::front::wgsl::parse_str(
            r"
struct Camera {
    value: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;

fn read_camera() -> vec4<f32> {
    return camera.value;
}

@vertex
fn vs_main(@location(0) enabled: bool) -> @builtin(position) vec4<f32> {
    return read_camera();
}
",
        )
        .expect("the shader should parse");
        let bindings = ShaderMetadata::new(&module)
            .expect("the shader should expose bindings")
            .generate_bindings()
            .expect("logical shader inputs should still be reflected");
        let layouts = &bindings
            .pipelines
            .as_ref()
            .unwrap()
            .layout
            .bind_group_layouts;
        let camera = entry(layouts, 0, 0);
        assert_eq!(camera.visibility, vec![crate::ShaderStage::Vertex]);
    }

    #[test]
    fn wgsl_intrinsics_go_to_the_shared_section() {
        let module = naga::front::wgsl::parse_str(CAMERA_WGSL).expect("the shader should parse");
        let bindings = ShaderMetadata::new(&module)
            .expect("the shader should expose bindings")
            .generate_bindings()
            .expect("vertex input reflection should succeed");

        // The vectors and matrices WGSL itself defines are shared by every shader,
        // so they are described once, under the shared section, as `st0`, `st1`, and
        // so on.
        let shared = bindings.shared.expect("the intrinsics should be shared");
        let shared_types: Vec<&TypeInfo> = shared.types.iter().collect();
        assert_eq!(shared_types.len(), 2, "{shared_types:?}");
        assert_eq!(shared_types[0].id.as_str(), "st0");
        assert_eq!(shared_types[0].name, "vec4<f32>");
        assert_eq!(shared_types[1].id.as_str(), "st1");
        assert_eq!(shared_types[1].name, "mat4x4<f32>");

        // The struct the shader declares is its own, and its members point at the
        // shared types they are built from.
        let types = &bindings.types;
        assert_eq!(types.len(), 1, "{types:?}");
        let TypeDescriptor::Struct { members, .. } = &types[0].descriptor else {
            panic!("the type should be a struct: {types:?}");
        };
        assert_eq!(members[0].ty.as_str(), "st0");
        assert_eq!(members[1].ty.as_str(), "st1");
    }

    #[test]
    fn vertex_pipeline_exposes_typed_attributes_and_matching_buffer_layout() {
        let module =
            naga::front::wgsl::parse_str(TRIANGLE_WGSL).expect("the triangle shader should parse");
        let bindings = ShaderMetadata::new(&module)
            .expect("the shader should expose bindings")
            .generate_bindings()
            .expect("the vertex input layout should be representable");
        let vertex = bindings
            .pipelines
            .expect("the shader should expose a pipeline")
            .vertex
            .expect("the shader should expose a vertex stage");

        assert_eq!(vertex.entry_point, "vs_main");
        let buffers = vertex
            .buffers
            .expect("the vertex attributes should have a default buffer layout");
        assert_eq!(buffers.len(), 1);
        assert_eq!(buffers[0].array_stride, 32);
        assert_eq!(buffers[0].step_mode, crate::VertexStepMode::Vertex);
        assert_eq!(buffers[0].attributes.len(), 2);

        let position = &buffers[0].attributes[0];
        assert_eq!(position.name, "position");
        assert_eq!(position.shader_location, 0);
        assert_eq!(position.format, crate::VertexFormat::Float32x3);
        assert_eq!(position.offset, 0);
        assert_eq!(position.ty.as_str(), "st0");

        let color = &buffers[0].attributes[1];
        assert_eq!(color.name, "color");
        assert_eq!(color.shader_location, 1);
        assert_eq!(color.format, crate::VertexFormat::Float32x4);
        assert_eq!(color.offset, 16);
        assert_eq!(color.ty.as_str(), "st1");

        let json = serde_json::to_value(
            ShaderMetadata::new(&module)
                .expect("the shader should expose bindings")
                .generate_bindings()
                .expect("the vertex input layout should be representable"),
        )
        .expect("the schema should serialize");
        assert_eq!(
            json["pipelines"]["vertex"]["buffers"][0]["array_stride"],
            32
        );
        assert_eq!(
            json["pipelines"]["vertex"]["buffers"][0]["attributes"][1]["shader_location"],
            1
        );
        assert_eq!(
            json["pipelines"]["vertex"]["buffers"][0]["attributes"][1]["format"],
            "Float32x4"
        );
    }

    #[test]
    fn unsupported_vertex_formats_remain_visible_without_a_guessed_layout() {
        let module = naga::front::wgsl::parse_str(
            r"
@vertex
fn vs_main(@location(0) enabled: bool) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
",
        )
        .expect("the shader should parse");
        let bindings = ShaderMetadata::new(&module)
            .expect("the shader should expose bindings")
            .generate_bindings()
            .expect("logical shader inputs should still be reflected");
        let vertex = bindings.pipelines.unwrap().vertex.unwrap();
        assert_eq!(vertex.inputs.len(), 1);
        assert_eq!(vertex.inputs[0].format, None);
        assert!(vertex.buffers.is_none());
    }
}
