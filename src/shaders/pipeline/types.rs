//! The pipeline half of the schema: bind group layouts, the pipeline layout, and
//! the states a render or compute pipeline is made of.
//!
//! Only the data model lives here. Filling it in means walking a `naga::Module`
//! — finding bind groups, inferring which stages can see each binding, and so on
//! — which is a separate task, so everything below is a definition with no
//! behaviour behind it.
//!
//! Names and shapes follow the equivalent `wgpu` descriptors, minus everything
//! only the consuming host can decide when it builds the real pipeline: vertex
//! buffer layouts, render target formats and blend states, primitive topology,
//! and the depth/stencil format. What is left is what can be read back out of the
//! WGSL itself.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::StorageAccess;

/// The stages a shader can run on.
///
/// `wgpu` models this as a `ShaderStages` bitmask; a `Vec` keeps the JSON
/// readable, and an empty one is the same as "not visible to any stage".
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Mirrors `wgpu::VertexState`, minus `module` (the shader itself, which the
/// generator already has) and minus `buffers`: how vertex data is laid out is
/// the host's decision, and none of it is visible in the WGSL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexState {
    pub entry_point: String,
    pub compilation_options: PipelineCompilationOptions,
}

/// Mirrors `wgpu::FragmentState`, minus `module` and minus `targets`: the render
/// target format and its blend state come from the destination the host binds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentState {
    pub entry_point: String,
    pub compilation_options: PipelineCompilationOptions,
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
