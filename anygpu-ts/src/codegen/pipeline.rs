//! The pipeline half of a generated file.
//!
//! A schema says what a WGSL file needs: which bindings each `@group` holds,
//! which entry point each stage runs, and which constants a stage can override.
//! It says nothing about the half of a pipeline that only the host can decide —
//! topology, vertex buffers, render target formats, the depth/stencil format,
//! the sample count.
//!
//! None of that depends on the shader, so the support section — the native
//! WebGPU types a descriptor is built from, plus one generic `PipelineHelper` —
//! is not written by the generator at all. It is a real TypeScript file,
//! [`SUPPORT`], compiled into the binary and emitted verbatim. A batch writes it
//! out once for the whole folder; a single schema carries it inline, so a
//! one-file run is still self-contained.
//!
//! What is left for the generator is the part that does depend on the shader:
//! when the schema carries a pipeline, one concrete instance of the helper.

use std::collections::{BTreeSet, HashMap};

use color_eyre::eyre::{Result, eyre};

use crate::codegen::emit::CodeBuilder;
use crate::codegen::view::identifier;
use anygpu::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, PipelineCompilationOptions,
    PipelineDescriptor, SamplerBindingType, ShaderStage, StorageAccess, StorageTextureAccess,
    TextureSampleType, TextureViewDimension,
};

/// The class name a shader's helper is exported under: the schema's own file
/// name, in PascalCase, with `PipelineHelper` behind it.
///
/// `nested_struct` becomes `NestedStructPipelineHelper`, so a project reads the
/// generated files as the shaders it wrote.
pub fn helper_class(name: &str) -> String {
    let pascal: String = name
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect();
    let pascal = if pascal.is_empty() {
        // A file called `.wgsl` still has to produce a nameable class.
        "Shader".to_string()
    } else {
        pascal
    };
    format!("{}PipelineHelper", identifier(&pascal))
}

/// The support section, verbatim: the WebGPU types the descriptor is assembled
/// from, and the class that assembles it.
///
/// It is a real `.ts` file rather than something the generator assembles, since
/// none of it depends on a schema. `include_str!` reads it at compile time, so a
/// generated `support.ts` is a copy of the checked-in source, character for
/// character, and the file is editable and type-checkable like any other.
pub const SUPPORT: &str = include_str!("templates/support.ts");

/// Writes the support section into a file that carries everything inline.
///
/// A batch has a `support.ts` of its own and imports from it instead; this is
/// the single-file path, where the same text has to be in the one output.
pub fn emit_support(code: &mut CodeBuilder) -> Result<()> {
    code.blank();
    for line in SUPPORT.lines() {
        code.line(line);
    }
    Ok(())
}

/// The names from the support section that this pipeline's instance refers to.
///
/// A batch's per-shader file gets the support section from `support.ts`, so it
/// has to say which pieces of it it uses. `PipelineHelper` is named by the
/// instance itself, always. `AnygpuShaderStage` is named wherever a stage mask
/// is written, which is wherever a binding or a push constant range resolved at
/// least one stage — the same rule [`stages`] applies, minus the case where it
/// writes a bare `0`.
pub fn support_symbols(pipeline: &PipelineDescriptor) -> BTreeSet<&'static str> {
    let mut symbols = BTreeSet::from(["PipelineHelper"]);
    let mask = pipeline
        .layout
        .bind_group_layouts
        .iter()
        .flat_map(|group| &group.entries)
        .any(|entry| !entry.visibility.is_empty())
        || pipeline
            .layout
            .push_constant_ranges
            .iter()
            .any(|range| !range.stages.is_empty());
    if mask {
        symbols.insert("AnygpuShaderStage");
    }
    symbols
}

/// Writes the one export that belongs to a single shader: the pipeline its
/// schema described, already filled in.
pub fn emit_shader(
    code: &mut CodeBuilder,
    name: &str,
    pipeline: &PipelineDescriptor,
) -> Result<()> {
    let class = helper_class(name);
    let groups = &pipeline.layout.bind_group_layouts;
    let ranges = &pipeline.layout.push_constant_ranges;
    code.blank();
    code.line(&format!(
        "/** `{name}`, resolved: the pipeline this shader describes, with every"
    ));
    code.line(" * stage the schema knows about already in place. `generateDescriptor` turns it");
    code.line(" * into a WebGPU descriptor once the caller has a module and its own");
    code.line(" * fixed-function state. */");
    code.line(&format!("export const {class} = new PipelineHelper({{"));
    code.indented(|code| {
        if groups.is_empty() {
            code.line("bindGroupLayouts: [],");
        } else {
            code.line("bindGroupLayouts: [");
            code.indented(|code| {
                for group in groups {
                    // An empty group is a group: `@group(1)` with nothing in it still
                    // takes a layout, since the pipeline layout is positional.
                    if group.entries.is_empty() {
                        code.line("[],");
                        continue;
                    }
                    code.line("[");
                    code.indented(|code| {
                        // WebGPU requires `binding` to ascend within a layout, while
                        // the schema lists globals in the order naga walked them.
                        let mut entries: Vec<&BindGroupLayoutEntry> =
                            group.entries.iter().collect();
                        entries.sort_by_key(|entry| entry.binding);
                        for entry in entries {
                            code.line(&format!("{},", entry_layout(entry)?));
                        }
                        Ok(())
                    })?;
                    code.line("],");
                }
                Ok(())
            })?;
            code.line("],");
        }

        if ranges.is_empty() {
            code.line("pushConstantRanges: [],");
        } else {
            code.line("pushConstantRanges: [");
            code.indented(|code| {
                for range in ranges {
                    code.line(&format!(
                        "{{ stages: {}, start: {}, end: {} }},",
                        stages(&range.stages)?,
                        range.range_start,
                        range.range_end
                    ));
                }
                Ok(())
            })?;
            code.line("],");
        }

        if let Some(vertex) = &pipeline.vertex {
            code.line(&format!(
                "vertex: {},",
                stage(&vertex.entry_point, &vertex.compilation_options, false)?
            ));
        }
        if let Some(fragment) = &pipeline.fragment {
            code.line(&format!(
                "fragment: {},",
                stage(&fragment.entry_point, &fragment.compilation_options, false)?
            ));
        }
        if let Some(compute) = &pipeline.compute {
            code.line(&format!(
                "compute: {},",
                stage(&compute.entry_point, &compute.compilation_options, true)?
            ));
        }
        Ok(())
    })?;
    code.line("});");
    Ok(())
}

/// One stage, as `AnygpuStage` wants it.
///
/// `zero_initialize_workgroup_memory` is a compute-only flag and is only emitted
/// when it is set: leaving `false` in the file would claim a decision the shader
/// did not make.
fn stage(entry_point: &str, options: &PipelineCompilationOptions, compute: bool) -> Result<String> {
    let mut fields = vec![
        format!("entryPoint: {}", text(entry_point)),
        format!("constants: {}", constants(&options.constants)?),
    ];
    if compute && options.zero_initialize_workgroup_memory {
        fields.push("zeroInitializeWorkgroupMemory: true".to_string());
    }
    Ok(format!("{{ {} }}", fields.join(", ")))
}

/// The overridable constants of a stage.
///
/// They go through a `BTreeMap` because a hash map has no order of its own: the
/// same shader has to generate the same file every time.
fn constants(constants: &HashMap<String, f64>) -> Result<String> {
    let constants: std::collections::BTreeMap<&String, &f64> = constants.iter().collect();
    let mut fields = Vec::with_capacity(constants.len());
    for (name, value) in constants {
        if !value.is_finite() {
            return Err(eyre!(
                "the overridable constant `{name}` is {value}, which has no TypeScript literal"
            ));
        }
        fields.push(format!("{}: {value}", key(name)));
    }
    if fields.is_empty() {
        return Ok("{}".to_string());
    }
    Ok(format!("{{ {} }}", fields.join(", ")))
}

/// An object key. A WGSL override is an identifier, so it is written the way the
/// shader spells it; anything else falls back to a quoted key.
fn key(name: &str) -> String {
    if identifier(name) == name {
        name.to_string()
    } else {
        text(name)
    }
}

/// One bind group layout entry, as a native `GPUBindGroupLayoutEntry` literal.
///
/// The WGSL name has no WebGPU counterpart, and neither does the length of a
/// binding array, so both are left out rather than smuggled into a field the API
/// does not have.
fn entry_layout(entry: &BindGroupLayoutEntry) -> Result<String> {
    let mut fields = vec![
        format!("binding: {}", entry.binding),
        format!("visibility: {}", stages(&entry.visibility)?),
    ];
    match &entry.ty {
        BindingType::Buffer {
            buffer_type,
            has_dynamic_offset,
            min_binding_size,
        } => {
            let mut buffer = vec![format!("type: {}", text(buffer_type_name(buffer_type)))];
            if *has_dynamic_offset {
                buffer.push("hasDynamicOffset: true".to_string());
            }
            // `minBindingSize` is optional: leaving it out is what asks the
            // implementation to infer the size from the shader.
            if let Some(size) = min_binding_size {
                buffer.push(format!("minBindingSize: {size}"));
            }
            fields.push(format!("buffer: {{ {} }}", buffer.join(", ")));
        }
        BindingType::Sampler(sampler) => {
            fields.push(format!(
                "sampler: {{ type: {} }}",
                text(sampler_name(sampler))
            ));
        }
        BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled,
        } => {
            let mut texture = vec![
                format!("sampleType: {}", text(sample_type_name(sample_type))),
                format!("viewDimension: {}", text(dimension_name(view_dimension))),
            ];
            if *multisampled {
                texture.push("multisampled: true".to_string());
            }
            fields.push(format!("texture: {{ {} }}", texture.join(", ")));
        }
        BindingType::StorageTexture {
            access,
            format,
            view_dimension,
        } => {
            fields.push(format!(
                "storageTexture: {{ access: {}, format: {}, viewDimension: {} }}",
                text(storage_access_name(access)),
                text(format),
                text(dimension_name(view_dimension))
            ));
        }
    }
    Ok(format!("{{ {} }}", fields.join(", ")))
}

/// The stages that can see a binding, as a stage mask.
///
/// An empty list means no stage was resolved, which is `0` rather than a
/// pipeline the implementation will reject without saying why. A stage named
/// twice is written once: a mask is a set, and a repeated term would only be a
/// longer way of saying the same thing.
fn stages(stages: &[ShaderStage]) -> Result<String> {
    if stages.is_empty() {
        return Ok("0".to_string());
    }
    let mut names: Vec<&str> = Vec::with_capacity(stages.len());
    for stage in stages {
        let name = stage_name(stage);
        if !names.contains(&name) {
            names.push(name);
        }
    }
    Ok(names.join(" | "))
}

fn stage_name(stage: &ShaderStage) -> &'static str {
    match stage {
        ShaderStage::Vertex => "AnygpuShaderStage.VERTEX",
        ShaderStage::Fragment => "AnygpuShaderStage.FRAGMENT",
        ShaderStage::Compute => "AnygpuShaderStage.COMPUTE",
    }
}

/// How a buffer binding is typed, from what WGSL declared about its address
/// space.
fn buffer_type_name(buffer: &BufferBindingType) -> &'static str {
    match buffer {
        BufferBindingType::Uniform => "uniform",
        BufferBindingType::Storage { read_only } => match read_only {
            StorageAccess::Load => "read-only-storage",
            // A `read_write` buffer, a write-only one and an atomic one are all
            // plain storage as far as WebGPU is concerned: only a shader that
            // never writes can be told apart.
            StorageAccess::Store | StorageAccess::LoadStore | StorageAccess::Atomic => "storage",
        },
    }
}

fn sampler_name(sampler: &SamplerBindingType) -> &'static str {
    match sampler {
        SamplerBindingType::Filtering => "filtering",
        SamplerBindingType::NonFiltering => "non-filtering",
        SamplerBindingType::Comparison => "comparison",
    }
}

fn sample_type_name(sample_type: &TextureSampleType) -> &'static str {
    match sample_type {
        // A sampled float texture is filterable unless WGSL asked for an unfilterable
        // float, which is a different sample type in the API.
        TextureSampleType::Float { filterable: true } => "float",
        TextureSampleType::Float { filterable: false } => "unfilterable-float",
        TextureSampleType::Depth => "depth",
        TextureSampleType::Sint => "sint",
        TextureSampleType::Uint => "uint",
    }
}

fn dimension_name(dimension: &TextureViewDimension) -> &'static str {
    match dimension {
        TextureViewDimension::D1 => "1d",
        TextureViewDimension::D2 => "2d",
        TextureViewDimension::D2Array => "2d-array",
        TextureViewDimension::Cube => "cube",
        TextureViewDimension::CubeArray => "cube-array",
        TextureViewDimension::D3 => "3d",
    }
}

fn storage_access_name(access: &StorageTextureAccess) -> &'static str {
    match access {
        StorageTextureAccess::WriteOnly => "write-only",
        StorageTextureAccess::ReadOnly => "read-only",
        StorageTextureAccess::ReadWrite => "read-write",
    }
}

/// A TypeScript string literal.
///
/// The schema only ever holds WGSL identifiers and texture formats, so escaping
/// the two characters that can end a literal is enough; `{:?}` would emit Rust
/// escapes (`\u{..}`) for anything outside ASCII, which is not TypeScript.
fn text(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use anygpu::TypeDescriptor;

    use super::*;

    fn interner_types() -> Vec<anygpu::TypeInfo> {
        let mut interner = anygpu::TypeInterner::new();
        interner.intern("f32".to_string(), |_| TypeDescriptor::Scalar {
            scalar: anygpu::ScalarInfo {
                name: "float".to_string(),
                width: 4,
            },
        });
        interner.finish()
    }

    fn buffer_entry(buffer: BindingType) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding: 0,
            name: "buffer".to_string(),
            visibility: vec![ShaderStage::Compute],
            ty: buffer,
            count: None,
        }
    }

    fn entry_with(ty: BindingType) -> Result<String> {
        entry_layout(&buffer_entry(ty))
    }

    /// Generates a file from a schema, with or without a pipeline.
    fn generated(name: &str, pipeline: Option<PipelineDescriptor>) -> String {
        let schema = anygpu::ShaderBindings {
            types: interner_types(),
            pipelines: pipeline,
            bindgroups: vec![],
        };
        crate::codegen::generate_typescript(&schema, name).unwrap()
    }

    fn empty_pipeline() -> PipelineDescriptor {
        PipelineDescriptor {
            layout: anygpu::PipelineLayout {
                bind_group_layouts: vec![],
                push_constant_ranges: vec![],
            },
            vertex: None,
            fragment: None,
            compute: None,
        }
    }

    #[test]
    fn a_helper_is_named_after_its_shader() {
        assert_eq!(helper_class("nested_struct"), "NestedStructPipelineHelper");
        assert_eq!(helper_class("init"), "InitPipelineHelper");
        // A name that is not already a legal identifier still has to produce one.
        assert_eq!(helper_class("my-shader.2"), "MyShader2PipelineHelper");
        assert_eq!(helper_class(""), "ShaderPipelineHelper");
        assert_eq!(helper_class("__"), "ShaderPipelineHelper");
    }

    #[test]
    fn the_support_section_is_emitted_without_a_pipeline() {
        let typescript = generated("camera", None);
        assert!(
            typescript
                .contains("export type AnygpuBindGroupLayoutEntry = GPUBindGroupLayoutEntry;")
        );
        assert!(typescript.contains("export type AnygpuBindGroupLayout = GPUBindGroupLayout;"));
        assert!(typescript.contains("export type AnygpuPipelineLayout = GPUPipelineLayout;"));
        assert!(typescript.contains("export interface AnygpuStageState {"));
        assert!(typescript.contains("export class PipelineHelper {"));
        assert!(
            !typescript.contains("new PipelineHelper({"),
            "nothing to instantiate without a pipeline: {typescript}"
        );
    }

    #[test]
    fn a_render_shader_is_exported_by_name() {
        let mut pipeline = empty_pipeline();
        pipeline.vertex = Some(anygpu::VertexState {
            entry_point: "vs_main".to_string(),
            compilation_options: anygpu::PipelineCompilationOptions {
                constants: HashMap::from([("tint".to_string(), 0.5), ("b".to_string(), 2.0)]),
                zero_initialize_workgroup_memory: false,
            },
        });
        pipeline.fragment = Some(anygpu::FragmentState {
            entry_point: "fs_main".to_string(),
            compilation_options: anygpu::PipelineCompilationOptions {
                constants: HashMap::new(),
                zero_initialize_workgroup_memory: false,
            },
        });

        let typescript = generated("nested_struct", Some(pipeline));
        assert!(
            typescript.contains("export const NestedStructPipelineHelper = new PipelineHelper({"),
            "{typescript}"
        );
        // Constants are sorted by name, so the output does not depend on hashing.
        assert!(
            typescript
                .contains(r#"vertex: { entryPoint: "vs_main", constants: { b: 2, tint: 0.5 } },"#),
            "{typescript}"
        );
        assert!(
            typescript.contains(r#"fragment: { entryPoint: "fs_main", constants: {} },"#),
            "{typescript}"
        );
    }

    #[test]
    fn a_compute_stage_keeps_its_workgroup_flag() {
        let mut pipeline = empty_pipeline();
        pipeline.compute = Some(anygpu::ComputeState {
            entry_point: "cs_main".to_string(),
            compilation_options: anygpu::PipelineCompilationOptions {
                constants: HashMap::new(),
                zero_initialize_workgroup_memory: true,
            },
        });

        let typescript = generated("compute_only", Some(pipeline));
        assert!(
            typescript.contains(
                r#"compute: { entryPoint: "cs_main", constants: {}, zeroInitializeWorkgroupMemory: true },"#
            ),
            "{typescript}"
        );
    }

    #[test]
    fn buffers_are_typed_by_their_address_space() {
        let uniform = entry_with(BindingType::Buffer {
            buffer_type: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        })
        .unwrap();
        assert_eq!(
            uniform,
            r#"{ binding: 0, visibility: AnygpuShaderStage.COMPUTE, buffer: { type: "uniform" } }"#
        );

        // A read-only storage buffer, the only kind WebGPU can tell apart.
        let read_only = entry_with(BindingType::Buffer {
            buffer_type: BufferBindingType::Storage {
                read_only: StorageAccess::Load,
            },
            has_dynamic_offset: true,
            min_binding_size: Some(64),
        })
        .unwrap();
        assert_eq!(
            read_only,
            r#"{ binding: 0, visibility: AnygpuShaderStage.COMPUTE, buffer: { type: "read-only-storage", hasDynamicOffset: true, minBindingSize: 64 } }"#
        );

        for access in [
            StorageAccess::Store,
            StorageAccess::LoadStore,
            StorageAccess::Atomic,
        ] {
            let read_write = entry_with(BindingType::Buffer {
                buffer_type: BufferBindingType::Storage { read_only: access },
                has_dynamic_offset: false,
                min_binding_size: None,
            })
            .unwrap();
            assert!(
                read_write.ends_with(r#"buffer: { type: "storage" } }"#),
                "{read_write}"
            );
        }
    }

    #[test]
    fn textures_samplers_and_storage_textures_use_native_shapes() {
        for (sample_type, expected) in [
            (TextureSampleType::Float { filterable: true }, "float"),
            (
                TextureSampleType::Float { filterable: false },
                "unfilterable-float",
            ),
            (TextureSampleType::Depth, "depth"),
            (TextureSampleType::Sint, "sint"),
            (TextureSampleType::Uint, "uint"),
        ] {
            let entry = entry_with(BindingType::Texture {
                sample_type,
                view_dimension: TextureViewDimension::D2Array,
                multisampled: true,
            })
            .unwrap();
            assert_eq!(
                entry,
                format!(
                    r#"{{ binding: 0, visibility: AnygpuShaderStage.COMPUTE, texture: {{ sampleType: "{expected}", viewDimension: "2d-array", multisampled: true }} }}"#
                )
            );
        }

        for (sampler, expected) in [
            (SamplerBindingType::Filtering, "filtering"),
            (SamplerBindingType::NonFiltering, "non-filtering"),
            (SamplerBindingType::Comparison, "comparison"),
        ] {
            assert_eq!(
                entry_with(BindingType::Sampler(sampler)).unwrap(),
                format!(
                    r#"{{ binding: 0, visibility: AnygpuShaderStage.COMPUTE, sampler: {{ type: "{expected}" }} }}"#
                )
            );
        }

        for (access, expected) in [
            (StorageTextureAccess::WriteOnly, "write-only"),
            (StorageTextureAccess::ReadOnly, "read-only"),
            (StorageTextureAccess::ReadWrite, "read-write"),
        ] {
            let entry = entry_with(BindingType::StorageTexture {
                access,
                format: "rgba8unorm".to_string(),
                view_dimension: TextureViewDimension::D2,
            })
            .unwrap();
            assert_eq!(
                entry,
                format!(
                    r#"{{ binding: 0, visibility: AnygpuShaderStage.COMPUTE, storageTexture: {{ access: "{expected}", format: "rgba8unorm", viewDimension: "2d" }} }}"#
                )
            );
        }
    }

    #[test]
    fn every_view_dimension_has_a_name() {
        for (dimension, expected) in [
            (TextureViewDimension::D1, "1d"),
            (TextureViewDimension::D2, "2d"),
            (TextureViewDimension::D2Array, "2d-array"),
            (TextureViewDimension::Cube, "cube"),
            (TextureViewDimension::CubeArray, "cube-array"),
            (TextureViewDimension::D3, "3d"),
        ] {
            assert_eq!(dimension_name(&dimension), expected);
        }
    }

    #[test]
    fn visibility_is_a_stage_mask_or_nothing() {
        assert_eq!(stages(&[]).unwrap(), "0");
        assert_eq!(
            stages(&[ShaderStage::Vertex, ShaderStage::Fragment]).unwrap(),
            "AnygpuShaderStage.VERTEX | AnygpuShaderStage.FRAGMENT"
        );
    }

    #[test]
    fn entries_are_emitted_in_binding_order() {
        let mut pipeline = empty_pipeline();
        pipeline.layout.bind_group_layouts = vec![anygpu::BindGroupLayout {
            group: 0,
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 2,
                    name: "third".to_string(),
                    visibility: vec![],
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 0,
                    name: "first".to_string(),
                    visibility: vec![],
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    name: "second".to_string(),
                    visibility: vec![],
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        }];
        pipeline.layout.push_constant_ranges = vec![anygpu::PushConstantRange {
            stages: vec![ShaderStage::Vertex, ShaderStage::Compute],
            range_start: 16,
            range_end: 32,
        }];

        let typescript = generated("sorted", Some(pipeline.clone()));
        let group = typescript
            .split_once("bindGroupLayouts: [")
            .unwrap()
            .1
            .split_once("pushConstantRanges:")
            .unwrap()
            .0
            .to_string();
        let bindings: Vec<usize> = group
            .lines()
            .filter_map(|line| line.trim().strip_prefix("{ binding: "))
            .filter_map(|line| line.split(',').next())
            .filter_map(|binding| binding.parse().ok())
            .collect();
        assert_eq!(bindings, [0, 1, 2], "{group}");

        // An empty group still takes a slot, since the layout is positional.
        let typescript = generated(
            "empty_group",
            Some(PipelineDescriptor {
                layout: anygpu::PipelineLayout {
                    bind_group_layouts: vec![
                        pipeline.layout.bind_group_layouts[0].clone(),
                        anygpu::BindGroupLayout {
                            group: 1,
                            entries: vec![],
                        },
                    ],
                    push_constant_ranges: vec![],
                },
                vertex: None,
                fragment: None,
                compute: None,
            }),
        );
        assert!(
            typescript.contains("    [],\n  ],\n  pushConstantRanges:"),
            "{typescript}"
        );
    }

    #[test]
    fn strings_are_escaped() {
        assert_eq!(text("rgba8unorm"), "\"rgba8unorm\"");
        assert_eq!(text("a\"b\\c"), "\"a\\\"b\\\\c\"");
    }
}
