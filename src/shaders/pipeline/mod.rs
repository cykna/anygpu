use std::collections::HashMap;

use color_eyre::eyre::{Result, eyre};
use naga::{AddressSpace, Binding, Handle, ScalarKind, ShaderStage, Type, TypeInner, VectorSize};

use crate::{
    ComputeState, FragmentState, PipelineCompilationOptions, ShaderMetadata, StorageAccess,
    types::{
        BindGroupLayout, BindGroupLayoutEntry, BufferBindingType, PipelineDescriptor,
        PipelineLayout, VertexAttribute, VertexBufferLayout, VertexFormat, VertexInput,
        VertexState, VertexStepMode,
    },
};

pub mod types;

impl<'a> ShaderMetadata<'a> {
    pub fn get_vertex(&self) -> Result<Option<VertexState>> {
        let (_, type_ids) = self.get_types_with_ids();
        self.get_vertex_with_ids(&type_ids)
    }

    fn get_vertex_with_ids(
        &self,
        type_ids: &HashMap<Handle<Type>, crate::TypeId>,
    ) -> Result<Option<VertexState>> {
        let Some(entry) = self
            .module
            .entry_points
            .iter()
            .find(|entry| entry.stage == ShaderStage::Vertex)
        else {
            return Ok(None);
        };
        let mut buffers = Vec::new();
        let mut inputs = Vec::new();
        let mut can_build_buffers = true;
        for argument in &entry.function.arguments {
            if let Some(Binding::BuiltIn(_)) = argument.binding.as_ref() {
                continue;
            }
            // Add input entry referencing the argument type
            if let Some(type_id) = type_ids.get(&argument.ty) {
                let location = match argument.binding.as_ref() {
                    Some(Binding::Location { location, .. }) => Some(*location),
                    _ => None,
                };
                inputs.push(VertexInput {
                    name: argument.name.clone().unwrap_or_else(|| "input".to_string()),
                    shader_location: location,
                    ty: type_id.clone(),
                    format: vertex_format(self.module, argument.ty),
                    offset: None,
                });
            }

            let mut argument_inputs = Vec::new();
            self.collect_vertex_inputs(
                argument.ty,
                argument.binding.as_ref(),
                argument.name.as_deref(),
                0,
                type_ids,
                &mut argument_inputs,
            )?;
            if argument_inputs.is_empty() {
                continue;
            }
            if argument_inputs.iter().any(|input| input.format.is_none()) {
                can_build_buffers = false;
                continue;
            }

            let mut attributes: Vec<VertexAttribute> = argument_inputs
                .into_iter()
                .map(|input| VertexAttribute {
                    name: input.name,
                    shader_location: input.shader_location.expect("location required for buffer layout"),
                    ty: input.ty,
                    format: input.format.expect("checked above"),
                    offset: input.offset.expect("offset required for buffer layout"),
                })
                .collect();
            attributes.sort_by_key(|attribute| attribute.shader_location);
            let array_stride = u64::from(self.layouter[argument.ty].size);
            if array_stride == 0 || array_stride % 4 != 0 {
                can_build_buffers = false;
                continue;
            }
            for attribute in &attributes {
                let end = attribute.offset + vertex_format_size(attribute.format);
                if end > array_stride {
                    can_build_buffers = false;
                    break;
                }
            }
            if !can_build_buffers {
                continue;
            }
            buffers.push(VertexBufferLayout {
                array_stride,
                step_mode: VertexStepMode::Vertex,
                attributes,
            });
        }
        Ok(Some(VertexState {
            entry_point: entry.name.clone(),
            compilation_options: PipelineCompilationOptions {
                constants: HashMap::new(),
                zero_initialize_workgroup_memory: false,
            },
            inputs,
            buffers: can_build_buffers.then_some(buffers),
        }))
    }

    fn collect_vertex_inputs(
        &self,
        ty: Handle<Type>,
        binding: Option<&Binding>,
        name: Option<&str>,
        base_offset: u64,
        type_ids: &HashMap<Handle<Type>, crate::TypeId>,
        inputs: &mut Vec<VertexInput>,
    ) -> Result<()> {
        match binding {
            Some(Binding::BuiltIn(_)) => return Ok(()),
            Some(Binding::Location { location, .. }) => {
                let type_id = type_ids.get(&ty).cloned().ok_or_else(|| {
                    eyre!("vertex input type at location {location} was not interned")
                })?;
                inputs.push(VertexInput {
                    name: name.unwrap_or("input").to_owned(),
                    shader_location: Some(*location),
                    ty: type_id,
                    format: vertex_format(self.module, ty),
                    offset: Some(base_offset),
                });
                return Ok(());
            }
            None => {}
        }

        match &self.module.types[ty].inner {
            TypeInner::Struct { members, .. } => {
                for member in members {
                    let member_name = member.name.as_deref().or(name);
                    self.collect_vertex_inputs(
                        member.ty,
                        member.binding.as_ref(),
                        member_name,
                        base_offset + u64::from(member.offset),
                        type_ids,
                        inputs,
                    )?;
                }
                Ok(())
            }
            _ => Err(eyre!(
                "vertex input `{}` has no location or builtin binding",
                name.unwrap_or("unnamed")
            )),
        }
    }

    pub fn get_fragment(&self) -> Option<FragmentState> {
        self.module.entry_points.iter().find_map(|entry| {
            if entry.stage == ShaderStage::Fragment {
                let out = FragmentState {
                    entry_point: entry.name.clone(),
                    compilation_options: PipelineCompilationOptions {
                        constants: HashMap::new(),
                        zero_initialize_workgroup_memory: false,
                    },
                };
                Some(out)
            } else {
                None
            }
        })
    }

    pub fn get_compute(&self) -> Option<ComputeState> {
        self.module.entry_points.iter().find_map(|entry| {
            if entry.stage == ShaderStage::Compute {
                let out = ComputeState {
                    entry_point: entry.name.clone(),
                    compilation_options: PipelineCompilationOptions {
                        constants: HashMap::new(),
                        zero_initialize_workgroup_memory: false,
                    },
                };
                Some(out)
            } else {
                None
            }
        })
    }

    pub fn get_bindgroups(&self) -> Vec<BindGroupLayout> {
        let mut bindings: Vec<BindGroupLayout> = Vec::new();
        for (_, global) in self.module.global_variables.iter() {
            if let Some(bind) = global.binding {
                let index = bind.group as usize;
                // Groups do not have to be declared in order, so the layouts are
                // grown to fit rather than inserted at a fixed position.
                while bindings.len() <= index {
                    bindings.push(BindGroupLayout {
                        group: bindings.len() as u32,
                        entries: vec![],
                    });
                }
                let buffer_type = match &global.space {
                    AddressSpace::Uniform => BufferBindingType::Uniform,
                    AddressSpace::Storage { access } => BufferBindingType::Storage {
                        read_only: StorageAccess::from(*access),
                    },
                    // Textures and samplers live in the handle address space and
                    // are not buffers; they need a different `BindingType` variant.
                    _ => continue,
                };
                bindings[index].entries.push(BindGroupLayoutEntry {
                    binding: bind.binding,
                    name: global.name.clone().unwrap(),
                    visibility: vec![],
                    count: None,
                    ty: types::BindingType::Buffer {
                        buffer_type,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                });
            }
        }
        bindings
    }
    pub fn get_pipeline(&self) -> Result<PipelineDescriptor> {
        let (_, type_ids) = self.get_types_with_ids();
        self.get_pipeline_with_ids(&type_ids)
    }

    pub(super) fn get_pipeline_with_ids(
        &self,
        type_ids: &HashMap<Handle<Type>, crate::TypeId>,
    ) -> Result<PipelineDescriptor> {
        Ok(PipelineDescriptor {
            layout: PipelineLayout {
                bind_group_layouts: self.get_bindgroups(),
                push_constant_ranges: vec![],
            },
            vertex: self.get_vertex_with_ids(type_ids)?,
            fragment: self.get_fragment(),
            compute: self.get_compute(),
        })
    }
}

fn vertex_format(module: &naga::Module, handle: Handle<Type>) -> Option<VertexFormat> {
    let (kind, width, length) = match module.types[handle].inner {
        TypeInner::Scalar(scalar) => (scalar.kind, scalar.width, 1),
        TypeInner::Vector { size, scalar } => (scalar.kind, scalar.width, vector_length(size)),
        _ => return None,
    };
    let format = match (kind, width, length) {
        (ScalarKind::Float, 2, 2) => VertexFormat::Float16x2,
        (ScalarKind::Float, 2, 4) => VertexFormat::Float16x4,
        (ScalarKind::Float, 4, 1) => VertexFormat::Float32,
        (ScalarKind::Float, 4, 2) => VertexFormat::Float32x2,
        (ScalarKind::Float, 4, 3) => VertexFormat::Float32x3,
        (ScalarKind::Float, 4, 4) => VertexFormat::Float32x4,
        (ScalarKind::Sint, 4, 1) => VertexFormat::Sint32,
        (ScalarKind::Sint, 4, 2) => VertexFormat::Sint32x2,
        (ScalarKind::Sint, 4, 3) => VertexFormat::Sint32x3,
        (ScalarKind::Sint, 4, 4) => VertexFormat::Sint32x4,
        (ScalarKind::Uint, 4, 1) => VertexFormat::Uint32,
        (ScalarKind::Uint, 4, 2) => VertexFormat::Uint32x2,
        (ScalarKind::Uint, 4, 3) => VertexFormat::Uint32x3,
        (ScalarKind::Uint, 4, 4) => VertexFormat::Uint32x4,
        _ => return None,
    };
    Some(format)
}

fn vertex_format_size(format: VertexFormat) -> u64 {
    match format {
        VertexFormat::Float16x2 => 4,
        VertexFormat::Float16x4 => 8,
        VertexFormat::Float32 | VertexFormat::Sint32 | VertexFormat::Uint32 => 4,
        VertexFormat::Float32x2 | VertexFormat::Sint32x2 | VertexFormat::Uint32x2 => 8,
        VertexFormat::Float32x3 | VertexFormat::Sint32x3 | VertexFormat::Uint32x3 => 12,
        VertexFormat::Float32x4 | VertexFormat::Sint32x4 | VertexFormat::Uint32x4 => 16,
    }
}

fn vector_length(size: VectorSize) -> u8 {
    match size {
        VectorSize::Bi => 2,
        VectorSize::Tri => 3,
        VectorSize::Quad => 4,
    }
}
