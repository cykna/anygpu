use std::collections::HashMap;

use color_eyre::eyre::{Result, eyre};
use naga::{
    AddressSpace, Binding, Handle, ScalarKind, ShaderStage as NagaShaderStage,
    StorageAccess as NagaStorageAccess, Type, TypeInner, VectorSize,
};

use crate::{
    ComputeState, FragmentState, InterfaceVariable, PipelineCompilationOptions, ShaderMetadata,
    ShaderStage, StorageAccess,
    shaders::types::binding_attributes,
    types::{
        BindGroupLayout, BindGroupLayoutEntry, BindingType, BufferBindingType, PipelineDescriptor,
        PipelineLayout, ResourceAccess, ResourceAddressSpace, SamplerBindingType,
        StorageTextureAccess, TextureSampleType, TextureViewDimension, VertexAttribute,
        VertexBufferLayout, VertexFormat, VertexInput, VertexState, VertexStepMode,
    },
};

pub mod types;

impl<'a> ShaderMetadata<'a> {
    pub(super) fn validate_bound_resources(&self, module_id: Option<&str>) -> Result<()> {
        for (_, global) in self.module.global_variables.iter() {
            if global.binding.is_none() {
                continue;
            }

            let inner = &self.module.types[global.ty].inner;
            let supported_buffer_type = matches!(
                inner,
                TypeInner::Scalar(_)
                    | TypeInner::Vector { .. }
                    | TypeInner::Matrix { .. }
                    | TypeInner::Atomic(_)
                    | TypeInner::Array { .. }
                    | TypeInner::Struct { .. }
            );
            match (&global.space, inner) {
                (AddressSpace::Uniform, _) | (AddressSpace::Storage { .. }, _)
                    if supported_buffer_type => {}
                (AddressSpace::Handle, TypeInner::Sampler { .. })
                | (AddressSpace::Handle, TypeInner::Image { .. }) => {
                    if let TypeInner::Image {
                        class: naga::ImageClass::External,
                        ..
                    } = inner
                    {
                        return Err(unsupported_binding(
                            module_id,
                            global,
                            inner,
                            "external textures are unsupported",
                        ));
                    }
                    if let TypeInner::Image {
                        class: naga::ImageClass::Storage { access, .. },
                        ..
                    } = inner
                        && matches!(
                            StorageAccess::from(*access),
                            StorageAccess::Atomic
                                | StorageAccess::AtomicLoad
                                | StorageAccess::AtomicStore
                                | StorageAccess::AtomicLoadStore
                        )
                    {
                        return Err(unsupported_binding(
                            module_id,
                            global,
                            inner,
                            "atomic storage textures are unsupported",
                        ));
                    }
                    if let TypeInner::Image {
                        class: naga::ImageClass::Sampled { kind, .. },
                        ..
                    } = inner
                        && !matches!(
                            kind,
                            ScalarKind::Float | ScalarKind::Sint | ScalarKind::Uint
                        )
                    {
                        return Err(unsupported_binding(
                            module_id,
                            global,
                            inner,
                            format!("sampled texture scalar kind `{kind:?}` is unsupported"),
                        ));
                    }
                }
                (_, TypeInner::BindingArray { .. }) => {
                    return Err(unsupported_binding(
                        module_id,
                        global,
                        inner,
                        "binding arrays are not supported",
                    ));
                }
                _ => {
                    return Err(unsupported_binding(
                        module_id,
                        global,
                        inner,
                        "resource type or address space is not supported",
                    ));
                }
            }
        }
        Ok(())
    }

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

    pub fn get_bindgroups(&self) -> Result<Vec<BindGroupLayout>> {
        self.validate_bound_resources(None)?;
        self.get_bindgroups_for_module(None)
    }

    fn get_bindgroups_for_module(&self, module_id: Option<&str>) -> Result<Vec<BindGroupLayout>> {
        let mut bindings: Vec<BindGroupLayout> = Vec::new();
        let (_, types_mapping) = self.get_types_with_ids();

        for (handle, global) in self.module.global_variables.iter() {
            if let Some(bind) = global.binding {
                let resource_name = global.name.as_deref().unwrap_or("<unnamed>");
                let resource_type = &self.module.types[global.ty].inner;
                if matches!(resource_type, TypeInner::BindingArray { .. }) {
                    return Err(unsupported_binding(
                        module_id,
                        global,
                        resource_type,
                        "binding arrays are not supported",
                    ));
                }

                let index = bind.group as usize;
                // Groups do not have to be declared in order, so the layouts are
                // grown to fit rather than inserted at a fixed position.
                while bindings.len() <= index {
                    bindings.push(BindGroupLayout {
                        group: bindings.len() as u32,
                        entries: vec![],
                    });
                }
                let (address_space, access, ty) = match &global.space {
                    AddressSpace::Uniform => (
                        ResourceAddressSpace::Uniform,
                        ResourceAccess::Read,
                        BindingType::Buffer {
                            buffer_type: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(u64::from(self.layouter[global.ty].size)),
                            ty: types_mapping
                                .get(&global.ty)
                                .ok_or_else(|| {
                                    unsupported_binding(
                                        module_id,
                                        global,
                                        resource_type,
                                        "the uniform type has no reflected type id",
                                    )
                                })?
                                .to_string(),
                        },
                    ),
                    AddressSpace::Storage { access } => (
                        ResourceAddressSpace::Storage,
                        resource_access(*access).map_err(|error| {
                            unsupported_binding(module_id, global, resource_type, error.to_string())
                        })?,
                        BindingType::Buffer {
                            buffer_type: BufferBindingType::Storage {
                                read_only: StorageAccess::from(*access),
                            },
                            has_dynamic_offset: false,
                            min_binding_size: Some(u64::from(self.layouter[global.ty].size)),
                            ty: types_mapping
                                .get(&global.ty)
                                .ok_or_else(|| {
                                    unsupported_binding(
                                        module_id,
                                        global,
                                        resource_type,
                                        "the storage type has no reflected type id",
                                    )
                                })?
                                .to_string(),
                        },
                    ),
                    AddressSpace::Handle => {
                        let (binding_type, access) = match resource_type {
                            TypeInner::Sampler { comparison } => (
                                BindingType::Sampler(if *comparison {
                                    SamplerBindingType::Comparison
                                } else {
                                    SamplerBindingType::Filtering
                                }),
                                ResourceAccess::NotApplicable,
                            ),
                            TypeInner::Image {
                                dim,
                                arrayed,
                                class,
                            } => {
                                let view_dimension = texture_view_dimension(*dim, *arrayed);
                                let (binding_type, access) = match class {
                                    naga::ImageClass::Sampled { kind, multi } => {
                                        let sample_type = match kind {
                                            ScalarKind::Float => {
                                                TextureSampleType::Float { filterable: true }
                                            }
                                            ScalarKind::Sint => TextureSampleType::Sint,
                                            ScalarKind::Uint => TextureSampleType::Uint,
                                            _ => {
                                                return Err(unsupported_binding(
                                                    module_id,
                                                    global,
                                                    resource_type,
                                                    format!(
                                                        "sampled texture scalar kind `{kind:?}` is unsupported"
                                                    ),
                                                ));
                                            }
                                        };
                                        (
                                            BindingType::Texture {
                                                sample_type,
                                                view_dimension,
                                                multisampled: *multi,
                                            },
                                            ResourceAccess::Read,
                                        )
                                    }
                                    naga::ImageClass::Depth { multi } => (
                                        BindingType::Texture {
                                            sample_type: TextureSampleType::Depth,
                                            view_dimension,
                                            multisampled: *multi,
                                        },
                                        ResourceAccess::Read,
                                    ),
                                    naga::ImageClass::Storage { format, access } => {
                                        let storage_access = StorageAccess::from(*access);
                                        let layout_access = match storage_access {
                                            StorageAccess::Load => StorageTextureAccess::ReadOnly,
                                            StorageAccess::Store => StorageTextureAccess::WriteOnly,
                                            StorageAccess::LoadStore => {
                                                StorageTextureAccess::ReadWrite
                                            }
                                            StorageAccess::Atomic
                                            | StorageAccess::AtomicLoad
                                            | StorageAccess::AtomicStore
                                            | StorageAccess::AtomicLoadStore => {
                                                return Err(unsupported_binding(
                                                    module_id,
                                                    global,
                                                    resource_type,
                                                    "atomic storage textures are unsupported",
                                                ));
                                            }
                                            StorageAccess::Unknown(bits) => {
                                                return Err(unsupported_binding(
                                                    module_id,
                                                    global,
                                                    resource_type,
                                                    format!(
                                                        "storage texture access flags `{bits:#x}` are unsupported"
                                                    ),
                                                ));
                                            }
                                        };
                                        (
                                            BindingType::StorageTexture {
                                                access: layout_access,
                                                format: format!("{format:?}").to_ascii_lowercase(),
                                                view_dimension,
                                            },
                                            resource_access(*access)?,
                                        )
                                    }
                                    naga::ImageClass::External => {
                                        return Err(unsupported_binding(
                                            module_id,
                                            global,
                                            resource_type,
                                            "external textures are unsupported",
                                        ));
                                    }
                                };
                                (binding_type, access)
                            }
                            _ => {
                                return Err(unsupported_binding(
                                    module_id,
                                    global,
                                    resource_type,
                                    "handle address-space bindings must be images or samplers",
                                ));
                            }
                        };
                        (ResourceAddressSpace::Handle, access, binding_type)
                    }
                    _ => {
                        return Err(unsupported_binding(
                            module_id,
                            global,
                            resource_type,
                            format!(
                                "address space `{:?}` cannot be represented as a bind group binding",
                                global.space
                            ),
                        ));
                    }
                };
                let resource_type = types_mapping.get(&global.ty).ok_or_else(|| {
                    unsupported_binding(
                        module_id,
                        global,
                        resource_type,
                        "the resource type has no reflected type id",
                    )
                })?;
                bindings[index].entries.push(BindGroupLayoutEntry {
                    binding: bind.binding,
                    name: resource_name.to_owned(),
                    visibility: self.visibility(handle),
                    count: None,
                    ty,
                    address_space,
                    access,
                    resource_type: resource_type.clone(),
                });
            }
        }
        // The shader may declare bindings in any order; the layout does not.
        for group in &mut bindings {
            group.entries.sort_by_key(|entry| entry.binding);
        }
        Ok(bindings)
    }
    pub fn get_pipeline(&self) -> Result<PipelineDescriptor> {
        self.validate_bound_resources(None)?;
        let (_, type_ids) = self.get_types_with_ids();
        self.get_pipeline_with_ids(&type_ids, None)
    }

    pub(super) fn get_pipeline_with_ids(
        &self,
        type_ids: &HashMap<Handle<Type>, crate::TypeId>,
        module_id: Option<&str>,
    ) -> Result<PipelineDescriptor> {
        Ok(PipelineDescriptor {
            layout: PipelineLayout {
                bind_group_layouts: self.get_bindgroups_for_module(module_id)?,
                push_constant_ranges: vec![],
            },
            vertex: self.get_vertex_with_ids(type_ids)?,
            fragment: self.get_fragment_with_ids(type_ids)?,
            compute: self.get_compute(),
        })
    }
}

fn unsupported_binding(
    module_id: Option<&str>,
    global: &naga::GlobalVariable,
    resource_type: &TypeInner,
    reason: impl std::fmt::Display,
) -> color_eyre::Report {
    let name = global.name.as_deref().unwrap_or("<unnamed>");
    eyre!(
        "unsupported binding in module `{}` at group {} binding {} (`{name}`, type {resource_type:?}, address space {:?}): {reason}",
        module_id.unwrap_or("<WGSL module>"),
        global.binding.map_or(0, |binding| binding.group),
        global.binding.map_or(0, |binding| binding.binding),
        global.space
    )
}

fn resource_access(access: NagaStorageAccess) -> Result<ResourceAccess> {
    Ok(match StorageAccess::from(access) {
        StorageAccess::Load => ResourceAccess::Read,
        StorageAccess::Store => ResourceAccess::Write,
        StorageAccess::LoadStore => ResourceAccess::ReadWrite,
        StorageAccess::Atomic
        | StorageAccess::AtomicLoad
        | StorageAccess::AtomicStore
        | StorageAccess::AtomicLoadStore => ResourceAccess::Atomic,
        StorageAccess::Unknown(bits) => {
            return Err(eyre!("unsupported storage access flags `{bits:#x}`"));
        }
    })
}

fn texture_view_dimension(dimension: naga::ImageDimension, arrayed: bool) -> TextureViewDimension {
    match (dimension, arrayed) {
        (naga::ImageDimension::D1, _) => TextureViewDimension::D1,
        (naga::ImageDimension::D2, false) => TextureViewDimension::D2,
        (naga::ImageDimension::D2, true) => TextureViewDimension::D2Array,
        (naga::ImageDimension::D3, _) => TextureViewDimension::D3,
        (naga::ImageDimension::Cube, false) => TextureViewDimension::Cube,
        (naga::ImageDimension::Cube, true) => TextureViewDimension::CubeArray,
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
