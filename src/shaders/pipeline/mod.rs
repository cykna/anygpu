use std::collections::HashMap;

use color_eyre::eyre::{Result, eyre};
use naga::{
    AddressSpace, Binding, Handle, ScalarKind, ShaderStage as NagaShaderStage, Type, TypeInner,
    VectorSize,
};

use crate::{
    ComputeState, FragmentState, InterfaceVariable, PipelineCompilationOptions, ShaderMetadata,
    ShaderStage, StorageAccess, TypeInterner,
    shaders::types::binding_attributes,
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
            .find(|entry| entry.stage == NagaShaderStage::Vertex)
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
                    shader_location: input
                        .shader_location
                        .expect("location required for buffer layout"),
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
        let mut outputs = Vec::new();
        if let Some(result) = &entry.function.result {
            self.collect_interface_variables(
                result.ty,
                result.binding.as_ref(),
                Some("output"),
                type_ids,
                &mut outputs,
            )?;
        }
        Ok(Some(VertexState {
            entry_point: entry.name.clone(),
            compilation_options: PipelineCompilationOptions {
                constants: HashMap::new(),
                zero_initialize_workgroup_memory: false,
            },
            inputs,
            outputs,
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

    fn collect_interface_variables(
        &self,
        ty: Handle<Type>,
        binding: Option<&Binding>,
        name: Option<&str>,
        type_ids: &HashMap<Handle<Type>, crate::TypeId>,
        variables: &mut Vec<InterfaceVariable>,
    ) -> Result<()> {
        if matches!(
            binding,
            Some(Binding::Location { .. }) | Some(Binding::BuiltIn(_))
        ) {
            let type_id = type_ids.get(&ty).cloned().ok_or_else(|| {
                eyre!(
                    "interface type `{}` was not interned",
                    self.get_type_name(ty)
                )
            })?;
            let attributes = binding_attributes(binding);
            variables.push(InterfaceVariable {
                name: name.unwrap_or("value").to_owned(),
                ty: type_id,
                location: attributes.location,
                builtin: attributes.builtin,
                interpolation: attributes.interpolation,
                sampling: attributes.sampling,
                invariant: attributes.invariant,
            });
            return Ok(());
        }

        match &self.module.types[ty].inner {
            TypeInner::Struct { members, .. } => {
                for member in members {
                    let member_name = member.name.as_deref().or(name);
                    self.collect_interface_variables(
                        member.ty,
                        member.binding.as_ref(),
                        member_name,
                        type_ids,
                        variables,
                    )?;
                }
                Ok(())
            }
            _ => Err(eyre!(
                "interface value `{}` has no location or builtin binding",
                name.unwrap_or("unnamed")
            )),
        }
    }

    pub fn get_fragment(&self) -> Option<FragmentState> {
        let (_, type_ids) = self.get_types_with_ids();
        self.get_fragment_with_ids(&type_ids).ok().flatten()
    }

    fn get_fragment_with_ids(
        &self,
        type_ids: &HashMap<Handle<Type>, crate::TypeId>,
    ) -> Result<Option<FragmentState>> {
        let Some(entry) = self
            .module
            .entry_points
            .iter()
            .find(|entry| entry.stage == NagaShaderStage::Fragment)
        else {
            return Ok(None);
        };
        let mut inputs = Vec::new();
        for argument in &entry.function.arguments {
            self.collect_interface_variables(
                argument.ty,
                argument.binding.as_ref(),
                argument.name.as_deref(),
                type_ids,
                &mut inputs,
            )?;
        }
        let mut outputs = Vec::new();
        if let Some(result) = &entry.function.result {
            self.collect_interface_variables(
                result.ty,
                result.binding.as_ref(),
                Some("output"),
                type_ids,
                &mut outputs,
            )?;
        }
        Ok(Some(FragmentState {
            entry_point: entry.name.clone(),
            compilation_options: PipelineCompilationOptions {
                constants: HashMap::new(),
                zero_initialize_workgroup_memory: false,
            },
            inputs,
            outputs,
        }))
    }

    pub fn get_compute(&self) -> Option<ComputeState> {
        self.module.entry_points.iter().find_map(|entry| {
            if entry.stage == NagaShaderStage::Compute {
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

    /// The stages whose entry points actually use `global`, directly or through
    /// a function they call. Naga's validation already folds callee usage into
    /// each entry point's `global_uses`, so a helper's accesses are included.
    fn visibility(&self, global: Handle<naga::GlobalVariable>) -> Vec<ShaderStage> {
        let mut visibility = Vec::new();
        for (index, entry) in self.module.entry_points.iter().enumerate() {
            let used = match &self.module_info {
                Some(info) => !info.get_entry_point(index)[global].is_empty(),
                None => function_uses_global(self.module, &entry.function, global),
            };
            if !used {
                continue;
            }
            let stage = match entry.stage {
                NagaShaderStage::Vertex => ShaderStage::Vertex,
                NagaShaderStage::Fragment => ShaderStage::Fragment,
                NagaShaderStage::Compute => ShaderStage::Compute,
                _ => continue,
            };
            if !visibility.contains(&stage) {
                visibility.push(stage);
            }
        }
        // A stable order keeps the schema identical across runs, regardless of
        // the order the entry points happen to be declared in.
        visibility.sort_by_key(|stage| match stage {
            ShaderStage::Vertex => 0,
            ShaderStage::Fragment => 1,
            ShaderStage::Compute => 2,
        });
        visibility
    }

    pub fn get_bindgroups(&self) -> Vec<BindGroupLayout> {
        let mut bindings: Vec<BindGroupLayout> = Vec::new();
        let (types, types_mapping) = self.get_types_with_ids();

        for (handle, global) in self.module.global_variables.iter() {
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
                    visibility: self.visibility(handle),
                    count: None,
                    ty: types::BindingType::Buffer {
                        buffer_type,
                        has_dynamic_offset: false,
                        // The layouter reports a runtime-sized array as one
                        // element, which is exactly the minimum a binding of that
                        // type must provide. A fixed type reports its full size.
                        min_binding_size: Some(u64::from(self.layouter[global.ty].size)),
                        ty: types_mapping.get(&global.ty).unwrap().to_string(),
                    },
                });
            }
        }
        // The shader may declare bindings in any order; the layout does not.
        for group in &mut bindings {
            group.entries.sort_by_key(|entry| entry.binding);
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
            fragment: self.get_fragment_with_ids(type_ids)?,
            compute: self.get_compute(),
        })
    }
}

/// A best-effort IR walk for modules naga could not validate. Expressions hold
/// direct global references and calls point at the callee, so following both
/// gives the same transitive answer `ModuleInfo` would have provided.
fn function_uses_global(
    module: &naga::Module,
    function: &naga::Function,
    global: Handle<naga::GlobalVariable>,
) -> bool {
    if function.expressions.iter().any(|(_, expression)| {
        matches!(expression, naga::Expression::GlobalVariable(handle) if *handle == global)
    }) {
        return true;
    }
    block_uses_global(module, &function.body, global)
}

fn block_uses_global(
    module: &naga::Module,
    block: &naga::Block,
    global: Handle<naga::GlobalVariable>,
) -> bool {
    block.iter().any(|statement| match statement {
        naga::Statement::Block(block) => block_uses_global(module, block, global),
        naga::Statement::If { accept, reject, .. } => {
            block_uses_global(module, accept, global) || block_uses_global(module, reject, global)
        }
        naga::Statement::Switch { cases, .. } => cases
            .iter()
            .any(|case| block_uses_global(module, &case.body, global)),
        naga::Statement::Loop {
            body, continuing, ..
        } => {
            block_uses_global(module, body, global) || block_uses_global(module, continuing, global)
        }
        naga::Statement::Call { function, .. } => {
            function_uses_global(module, &module.functions[*function], global)
        }
        _ => false,
    })
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
