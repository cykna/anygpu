//! The pipeline half of the schema: bind group layouts, the pipeline layout, and
//! the states a render or compute pipeline is made of.
//!
//! Only the data model lives here. Filling it in means walking a `naga::Module`
//! — finding bind groups, inferring which stages can see each binding, and so on
//! — which is a separate task, so everything below is a definition with no
//! behaviour behind it.
//!
//! Names and shapes follow the equivalent `wgpu` descriptors, plus a default
//! vertex-buffer layout derived from the vertex entry point's reflected input
//! types. Render-target formats, blend states, primitive topology, and the
//! depth/stencil format remain host decisions.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{StorageAccess, TypeId};

/// The stages a shader can run on.
///
/// `wgpu` models this as a `ShaderStages` bitmask; a `Vec` keeps the JSON
/// readable, and an empty one is the same as "not visible to any stage".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

/// A multiview render mask, naming the view indices a pipeline renders to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiviewMask(pub u32);

/// A driver's compiled pipeline blob.
///
/// Reflecting a real cache is not something that can be read out of a shader,
/// and a blob is not portable across drivers or platforms, so this is only a
/// shape to hand one through: no loading, no saving, no validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineCache {
    pub data: Option<Vec<u8>>,
}

/// What a binding points at, mirroring `wgpu::BindingType`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BindingType {
    Buffer {
        buffer_type: BufferBindingType,
        has_dynamic_offset: bool,
        min_binding_size: Option<u64>,
        ty: String,
    },
    Sampler(SamplerBindingType),
    Texture {
        sample_type: TextureSampleType,
        view_dimension: TextureViewDimension,
        multisampled: bool,
    },
    StorageTexture {
        access: StorageTextureAccess,
        /// The format as declared in the WGSL, e.g. the `rgba8unorm` of
        /// `texture_storage_2d<rgba8unorm, write>`. Unlike the depth/stencil
        /// format, this one comes from the shader rather than from the host.
        format: String,
        view_dimension: TextureViewDimension,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceAddressSpace {
    Uniform,
    Storage,
    Handle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceAccess {
    Read,
    Write,
    ReadWrite,
    Atomic,
    NotApplicable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BufferBindingType {
    Uniform,
    Storage { read_only: StorageAccess },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplerBindingType {
    Filtering,
    NonFiltering,
    Comparison,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextureSampleType {
    Float { filterable: bool },
    Depth,
    Sint,
    Uint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextureViewDimension {
    D1,
    D2,
    D2Array,
    Cube,
    CubeArray,
    D3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageTextureAccess {
    WriteOnly,
    ReadOnly,
    ReadWrite,
}

/// One binding of a group, mirroring `wgpu::BindGroupLayoutEntry`.
///
/// `name` is not a `wgpu` concept: it is the WGSL global variable behind the
/// binding, so a downstream generator can refer to it by name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindGroupLayoutEntry {
    pub binding: u32,
    pub name: String,
    pub visibility: Vec<ShaderStage>,
    pub ty: BindingType,
    pub address_space: ResourceAddressSpace,
    pub access: ResourceAccess,
    pub resource_type: TypeId,
    /// `None` for a plain binding. `wgpu` uses `Option<NonZeroU32>` here; a
    /// plain `Option<u32>` keeps the JSON simple, at the cost of allowing a
    /// count of zero that `NonZeroU32` rules out.
    pub count: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindGroupLayout {
    pub group: u32,
    pub entries: Vec<BindGroupLayoutEntry>,
}

/// A slice of the push constant buffer a stage can write.
///
/// `wgpu` stores this as a single `Range<u32>`; a `Range` does not implement
/// `Serialize`, so the two bounds are kept apart instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushConstantRange {
    pub stages: Vec<ShaderStage>,
    pub range_start: u32,
    pub range_end: u32,
}

/// Mirrors `wgpu::PipelineLayoutDescriptor`, without `label`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineLayout {
    pub bind_group_layouts: Vec<BindGroupLayout>,
    pub push_constant_ranges: Vec<PushConstantRange>,
}

/// Mirrors `wgpu::PipelineCompilationOptions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineCompilationOptions {
    /// Overrides for the shader's pipeline-overridable constants.
    pub constants: HashMap<String, f64>,
    pub zero_initialize_workgroup_memory: bool,
}

/// A vertex attribute exposed by an entry point and its default buffer layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexAttribute {
    pub name: String,
    pub shader_location: u32,
    pub ty: crate::TypeId,
    pub format: VertexFormat,
    pub offset: u64,
}

/// One logical vertex-stage input. `format` is `None` when the WGSL input
/// cannot be represented by one WebGPU vertex attribute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexInput {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shader_location: Option<u32>,
    pub ty: crate::TypeId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<VertexFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<u64>,
}

/// One value crossing an entry point's interface, flattened out of a struct when
/// the shader groups several of them together. Unlike [`VertexInput`], this keeps
/// builtins and interpolation qualifiers: it describes the shader's interface,
/// not only the fixed-function vertex fetch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterfaceVariable {
    pub name: String,
    pub ty: crate::TypeId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub builtin: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interpolation: Option<crate::Interpolation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<crate::Sampling>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub invariant: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// Vertex formats representable by WebGPU's fixed-function vertex fetch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VertexFormat {
    Float16x2,
    Float16x4,
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Sint32,
    Sint32x2,
    Sint32x3,
    Sint32x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VertexStepMode {
    Vertex,
}

/// A default interleaved buffer layout using the reflected input struct's
/// member offsets and size. This is a usable convention, not a WGSL requirement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexBufferLayout {
    pub array_stride: u64,
    pub step_mode: VertexStepMode,
    pub attributes: Vec<VertexAttribute>,
}

/// Mirrors `wgpu::VertexState`, minus `module` (the shader itself, which the
/// generator already has).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexState {
    pub entry_point: String,
    pub compilation_options: PipelineCompilationOptions,
    #[serde(default)]
    pub inputs: Vec<VertexInput>,
    #[serde(default)]
    pub outputs: Vec<InterfaceVariable>,
    #[serde(default)]
    pub buffers: Option<Vec<VertexBufferLayout>>,
}

/// Mirrors `wgpu::FragmentState`, minus `module` and minus `targets`: the render
/// target format and its blend state come from the destination the host binds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentState {
    pub entry_point: String,
    pub compilation_options: PipelineCompilationOptions,
    #[serde(default)]
    pub inputs: Vec<InterfaceVariable>,
    #[serde(default)]
    pub outputs: Vec<InterfaceVariable>,
}

/// Mirrors `wgpu::ComputeState`, minus `module`.
///
/// A compute pipeline has no fixed-function state at all, so nothing else is
/// missing: the entry point and the overridable constants are the whole state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeState {
    pub entry_point: String,
    pub compilation_options: PipelineCompilationOptions,
}

/// One pipeline: the layout it binds against plus the states its stages run with.
///
/// There is no primitive topology here on purpose. It is a host decision, so the
/// schema stays silent about it rather than guessing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineDescriptor {
    pub layout: PipelineLayout,
    /// `None` when the shader has no such stage. A render pipeline needs a vertex
    /// stage and may have a fragment one; a compute pipeline needs neither.
    #[serde(default)]
    pub vertex: Option<VertexState>,
    #[serde(default)]
    pub fragment: Option<FragmentState>,
    #[serde(default)]
    pub compute: Option<ComputeState>,
}
