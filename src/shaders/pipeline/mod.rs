use std::collections::HashMap;

use naga::{AddressSpace, ShaderStage};

use crate::{
    ComputeState, FragmentState, PipelineCompilationOptions, ShaderMetadata, StorageAccess,
    VertexState,
    types::{
        BindGroupLayout, BindGroupLayoutEntry, BufferBindingType, PipelineDescriptor,
        PipelineLayout,
    },
};

pub mod types;

impl<'a> ShaderMetadata<'a> {
    pub fn get_vertex(&self) -> Option<VertexState> {
        self.module.entry_points.iter().find_map(|entry| {
            if entry.stage == ShaderStage::Vertex {
                let out = VertexState {
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
    pub fn get_pipeline(&self) -> PipelineDescriptor {
        PipelineDescriptor {
            layout: PipelineLayout {
                bind_group_layouts: self.get_bindgroups(),
                push_constant_ranges: vec![],
            },
            vertex: self.get_vertex(),
            fragment: self.get_fragment(),
            compute: self.get_compute(),
        }
    }
}
