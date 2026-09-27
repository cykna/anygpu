//! The pipeline half of a generated file, end to end.
//!
//! A generated file always carries the same support section — the WebGPU types a
//! descriptor is built from, and one generic `PipelineHelper` — and, when the
//! schema describes a pipeline, one instance of it named after the shader. These
//! tests run the generator over a real example's types and check the shape of what
//! comes out.
//!
//! The pipeline in `a_render_shader_gets_a_named_instance` is written by hand on
//! purpose: `nested_struct.wgsl` declares no entry points at all, so the entry
//! points and the push constant range there are a fixture, not reflection. What
//! the generator is being asked to do here is the conversion, not the guesswork.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anygpu::{
    BindGroupLayout, BindGroupLayoutEntry, BindingType, BufferBindingType, ComputeState,
    FragmentState, PipelineCompilationOptions, PipelineDescriptor, PipelineLayout,
    PushConstantRange, SamplerBindingType, ShaderBindings, ShaderMetadata, ShaderStage,
    StorageAccess, TextureSampleType, TextureViewDimension, VertexState, naga,
};
use anygpu_gen_ts::codegen::generate_typescript;
use color_eyre::eyre::Result;

fn example(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("anygpu-ts should live inside the workspace")
        .join("examples")
        .join(format!("{name}.wgsl"))
}

/// The types a `.wgsl` example reflects to, run through the real schema.
fn types_of(name: &str) -> Result<Vec<anygpu::TypeInfo>> {
    let source = std::fs::read_to_string(example(name))?;
    let module = naga::front::wgsl::parse_str(&source)?;
    Ok(ShaderMetadata::new(&module)?.get_types())
}

fn buffer(
    binding: u32,
    name: &str,
    visibility: Vec<ShaderStage>,
    buffer_type: BufferBindingType,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        name: name.to_string(),
        visibility,
        ty: BindingType::Buffer {
            buffer_type,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn options() -> PipelineCompilationOptions {
    PipelineCompilationOptions {
        constants: HashMap::new(),
        zero_initialize_workgroup_memory: false,
    }
}

/// The pipeline `nested_struct.wgsl` would describe if it had entry points.
///
/// Its bind groups are the ones the shader really declares; the entry points,
/// the push constant range and the visibility of each binding are a fixture,
/// since resolving those is a job of its own.
fn synthetic_render_pipeline() -> PipelineDescriptor {
    PipelineDescriptor {
        layout: PipelineLayout {
            bind_group_layouts: vec![
                BindGroupLayout {
                    group: 0,
                    entries: vec![
                        buffer(
                            0,
                            "scene",
                            vec![ShaderStage::Vertex, ShaderStage::Fragment],
                            BufferBindingType::Uniform,
                        ),
                        buffer(
                            1,
                            "lights",
                            vec![ShaderStage::Fragment],
                            BufferBindingType::Storage {
                                read_only: StorageAccess::Load,
                            },
                        ),
                        buffer(
                            2,
                            "particles",
                            vec![ShaderStage::Compute],
                            BufferBindingType::Storage {
                                read_only: StorageAccess::LoadStore,
                            },
                        ),
                    ],
                },
                BindGroupLayout {
                    group: 1,
                    entries: vec![
                        BindGroupLayoutEntry {
                            binding: 0,
                            name: "t".to_string(),
                            visibility: vec![ShaderStage::Fragment],
                            ty: BindingType::Texture {
                                sample_type: TextureSampleType::Float { filterable: true },
                                view_dimension: TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            name: "s".to_string(),
                            visibility: vec![ShaderStage::Fragment],
                            ty: BindingType::Sampler(SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                },
            ],
            push_constant_ranges: vec![PushConstantRange {
                stages: vec![ShaderStage::Vertex, ShaderStage::Fragment],
                range_start: 0,
                range_end: 16,
            }],
        },
        vertex: Some(VertexState {
            entry_point: "vs_main".to_string(),
            compilation_options: options(),
        }),
        fragment: Some(FragmentState {
            entry_point: "fs_main".to_string(),
            compilation_options: options(),
        }),
        compute: None,
    }
}

fn generated(
    name: &str,
    types: Vec<anygpu::TypeInfo>,
    pipeline: Option<PipelineDescriptor>,
) -> String {
    let schema = ShaderBindings {
        types,
        pipelines: pipeline,
        bindgroups: vec![],
    };
    generate_typescript(&schema, name).expect("the schema should generate")
}

/// The block a file ends with: the one export that belongs to the shader.
fn instance(typescript: &str) -> &str {
    typescript
        .split_once("new PipelineHelper({")
        .map(|(_, rest)| rest)
        .expect("the file should export a pipeline helper")
}

#[test]
fn the_support_section_is_in_every_file() {
    // Even a schema that describes no pipeline carries the class: it is what a
    // consumer reaches for, and it is the same class either way.
    let typescript = generated("camera", types_of("init").unwrap(), None);

    assert!(
        typescript.contains("export type AnygpuBindGroupLayoutEntry = GPUBindGroupLayoutEntry;"),
        "{typescript}"
    );
    assert!(
        typescript.contains("export type AnygpuBindGroupLayout = GPUBindGroupLayout;"),
        "{typescript}"
    );
    assert!(
        typescript.contains("export type AnygpuPipelineLayout = GPUPipelineLayout;"),
        "{typescript}"
    );
    assert!(
        typescript.contains("export interface AnygpuStageState {"),
        "{typescript}"
    );
    assert!(
        typescript.contains("export class PipelineHelper {"),
        "{typescript}"
    );
    // A shader with no pipeline gets no instance: there would be nothing to fill
    // it in with.
    assert!(!typescript.contains("new PipelineHelper({"), "{typescript}");
    // The types are still there, which is the point of generating at all.
    assert!(typescript.contains("export class Camera {"), "{typescript}");
}

#[test]
fn a_render_shader_gets_a_named_instance() {
    let typescript = generated(
        "nested_struct",
        types_of("nested_struct").unwrap(),
        Some(synthetic_render_pipeline()),
    );
    let instance = instance(&typescript);

    // Named after the file it was generated from.
    assert!(
        typescript.contains("export const NestedStructPipelineHelper = new PipelineHelper({"),
        "{typescript}"
    );
    // Every binding becomes a native `GPUBindGroupLayoutEntry`, in binding order.
    assert!(
        instance.contains(
            r#"{ binding: 0, visibility: AnygpuShaderStage.VERTEX | AnygpuShaderStage.FRAGMENT, buffer: { type: "uniform" } },"#
        ),
        "{instance}"
    );
    assert!(
        instance.contains(
            r#"{ binding: 1, visibility: AnygpuShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },"#
        ),
        "{instance}"
    );
    assert!(
        instance.contains(
            r#"{ binding: 2, visibility: AnygpuShaderStage.COMPUTE, buffer: { type: "storage" } },"#
        ),
        "{instance}"
    );
    assert!(
        instance.contains(
            r#"{ binding: 0, visibility: AnygpuShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "2d" } },"#
        ),
        "{instance}"
    );
    assert!(
        instance.contains(
            r#"{ binding: 1, visibility: AnygpuShaderStage.FRAGMENT, sampler: { type: "filtering" } },"#
        ),
        "{instance}"
    );
    assert!(
        instance.contains(
            "{ stages: AnygpuShaderStage.VERTEX | AnygpuShaderStage.FRAGMENT, start: 0, end: 16 },"
        ),
        "{instance}"
    );
    assert!(
        instance.contains(r#"vertex: { entryPoint: "vs_main", constants: {} },"#),
        "{instance}"
    );
    assert!(
        instance.contains(r#"fragment: { entryPoint: "fs_main", constants: {} },"#),
        "{instance}"
    );

    // Nothing the host owns leaks in: no topology, no buffers, no targets.
    for host_state in [
        "topology",
        "primitive",
        "targets",
        "depthStencil",
        "multisample",
        "createRenderPipeline",
    ] {
        let block = instance.split("});").next().unwrap_or(instance);
        assert!(
            !block.contains(host_state),
            "the schema says nothing about `{host_state}`: {block}"
        );
    }
}

#[test]
fn a_compute_shader_gets_a_compute_helper() {
    let mut pipeline = synthetic_render_pipeline();
    pipeline.vertex = None;
    pipeline.fragment = None;
    pipeline.compute = Some(ComputeState {
        entry_point: "cs_main".to_string(),
        compilation_options: PipelineCompilationOptions {
            constants: HashMap::from([("workgroup_size".to_string(), 64.0)]),
            zero_initialize_workgroup_memory: true,
        },
    });

    let typescript = generated(
        "particle_sim",
        types_of("nested_struct").unwrap(),
        Some(pipeline),
    );
    assert!(
        typescript.contains("export const ParticleSimPipelineHelper = new PipelineHelper({"),
        "{typescript}"
    );
    let instance = instance(&typescript);
    assert!(
        instance.contains(
            r#"compute: { entryPoint: "cs_main", constants: { workgroup_size: 64 }, zeroInitializeWorkgroupMemory: true },"#
        ),
        "{instance}"
    );
    assert!(!instance.contains("vertex:"), "{instance}");
    assert!(!instance.contains("fragment:"), "{instance}");
}

#[test]
fn a_name_that_is_not_an_identifier_still_produces_a_class() {
    let typescript = generated(
        "weird-name.2",
        types_of("init").unwrap(),
        Some(synthetic_render_pipeline()),
    );
    assert!(
        typescript.contains("export const WeirdName2PipelineHelper = new PipelineHelper({"),
        "{typescript}"
    );
}
