use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use super::{
    ReflectedModule,
    pipeline::types::{
        BindGroupLayoutEntry, BindingType, ComputeState, FragmentState, PipelineDescriptor,
        PushConstantRange, ResourceAccess, ResourceAddressSpace, ShaderStage, VertexState,
    },
    types::{
        DeclarationIdentity, LinkedMemberDescriptor, LinkedTypeDescriptor, LinkedTypeInfo,
        LinkedTypeReference, MemberDescriptor, TypeDescriptor, TypeId, TypeInfo,
    },
};

/// Version 2 adds all bound resources, their address/access metadata, and an
/// explicit local-or-canonical type reference to each binding.
pub const LINKED_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkedSchema {
    pub schema_version: u32,
    pub types: Vec<LinkedTypeInfo>,
    pub modules: Vec<LinkedModule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkedModule {
    pub id: String,
    pub types: Vec<LinkedTypeInfo>,
    pub local_to_canonical: BTreeMap<TypeId, TypeId>,
    pub pipelines: Option<LinkedPipelineDescriptor>,
    pub bindgroups: Vec<LinkedBindGroupLayoutEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkedPipelineDescriptor {
    pub layout: LinkedPipelineLayout,
    pub vertex: Option<VertexState>,
    pub fragment: Option<FragmentState>,
    pub compute: Option<ComputeState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkedPipelineLayout {
    pub bind_group_layouts: Vec<LinkedBindGroupLayout>,
    pub push_constant_ranges: Vec<PushConstantRange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkedBindGroupLayout {
    pub group: u32,
    pub entries: Vec<LinkedBindGroupLayoutEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkedBindGroupLayoutEntry {
    pub binding: u32,
    pub name: String,
    pub visibility: Vec<ShaderStage>,
    pub ty: BindingType,
    pub address_space: ResourceAddressSpace,
    pub access: ResourceAccess,
    pub resource_type: LinkedTypeReference,
    pub count: Option<u32>,
}

#[derive(Debug)]
pub enum LinkError {
    DuplicateModuleId(String),
    IdentityForNonStruct {
        module_id: String,
        type_id: TypeId,
        identity: DeclarationIdentity,
    },
    ConflictingTypeDefinition {
        identity: DeclarationIdentity,
        first_module: String,
        first_type: TypeId,
        conflicting_module: String,
        conflicting_type: TypeId,
    },
}

impl fmt::Display for LinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateModuleId(id) => write!(f, "duplicate reflected module id `{id}`"),
            Self::IdentityForNonStruct {
                module_id,
                type_id,
                identity,
            } => write!(
                f,
                "declaration identity `{}::{}` in module `{module_id}` refers to non-struct type `{type_id}`",
                identity.module_path, identity.name
            ),
            Self::ConflictingTypeDefinition {
                identity,
                first_module,
                first_type,
                conflicting_module,
                conflicting_type,
            } => write!(
                f,
                "conflicting definitions for `{}::{}`: `{first_type}` in module `{first_module}` differs from `{conflicting_type}` in module `{conflicting_module}`",
                identity.module_path, identity.name
            ),
        }
    }
}

impl std::error::Error for LinkError {}

#[derive(Debug, Clone)]
struct CanonicalSource {
    module_index: usize,
    type_id: TypeId,
    id: TypeId,
    identity: Option<DeclarationIdentity>,
}

pub fn link_modules(mut modules: Vec<ReflectedModule>) -> Result<LinkedSchema, LinkError> {
    modules.sort_by(|left, right| left.id.cmp(&right.id));
    for pair in modules.windows(2) {
        if pair[0].id == pair[1].id {
            return Err(LinkError::DuplicateModuleId(pair[0].id.clone()));
        }
    }

    let mut local_to_canonical = vec![BTreeMap::new(); modules.len()];
    let mut canonical_sources = Vec::<CanonicalSource>::new();
    let mut shared_types = HashMap::<TypeDescriptor, TypeId>::new();
    let mut identities = BTreeMap::<DeclarationIdentity, CanonicalSource>::new();

    for (module_index, module) in modules.iter().enumerate() {
        for ty in all_types(&module.bindings) {
            if ty.descriptor.is_shared() {
                let id = if let Some(id) = shared_types.get(&ty.descriptor) {
                    id.clone()
                } else {
                    let id = TypeId(format!("ct{}", canonical_sources.len()));
                    shared_types.insert(ty.descriptor.clone(), id.clone());
                    canonical_sources.push(CanonicalSource {
                        module_index,
                        type_id: ty.id.clone(),
                        id: id.clone(),
                        identity: None,
                    });
                    id
                };
                local_to_canonical[module_index].insert(ty.id.clone(), id);
                continue;
            }

            let Some(identity) = module.identities.get(&ty.id) else {
                continue;
            };
            if !matches!(ty.descriptor, TypeDescriptor::Struct { .. }) {
                return Err(LinkError::IdentityForNonStruct {
                    module_id: module.id.clone(),
                    type_id: ty.id.clone(),
                    identity: identity.clone(),
                });
            }

            if let Some(first) = identities.get(identity) {
                if !equivalent_type(
                    &modules[first.module_index],
                    &first.type_id,
                    module,
                    &ty.id,
                    &mut HashSet::new(),
                ) {
                    return Err(LinkError::ConflictingTypeDefinition {
                        identity: identity.clone(),
                        first_module: modules[first.module_index].id.clone(),
                        first_type: first.type_id.clone(),
                        conflicting_module: module.id.clone(),
                        conflicting_type: ty.id.clone(),
                    });
                }
                local_to_canonical[module_index].insert(ty.id.clone(), first.id.clone());
            } else {
                let id = TypeId(format!("ct{}", canonical_sources.len()));
                let source = CanonicalSource {
                    module_index,
                    type_id: ty.id.clone(),
                    id: id.clone(),
                    identity: Some(identity.clone()),
                };
                identities.insert(identity.clone(), source.clone());
                canonical_sources.push(source);
                local_to_canonical[module_index].insert(ty.id.clone(), id);
            }
        }
    }

    let types = canonical_sources
        .iter()
        .map(|source| {
            let module = &modules[source.module_index];
            let ty = find_type(&module.bindings, &source.type_id)
                .expect("canonical source points at a reflected type");
            LinkedTypeInfo {
                id: source.id.clone(),
                name: source
                    .identity
                    .as_ref()
                    .map_or_else(|| ty.name.clone(), |identity| identity.name.clone()),
                identity: source.identity.clone(),
                descriptor: link_descriptor(
                    &ty.descriptor,
                    module,
                    &local_to_canonical[source.module_index],
                ),
            }
        })
        .collect();

    let linked_modules = modules
        .iter()
        .enumerate()
        .map(|(index, module)| {
            let mut types = all_types(&module.bindings)
                .filter(|ty| !local_to_canonical[index].contains_key(&ty.id))
                .map(|ty| LinkedTypeInfo {
                    id: ty.id.clone(),
                    name: ty.name.clone(),
                    identity: None,
                    descriptor: link_descriptor(&ty.descriptor, module, &local_to_canonical[index]),
                })
                .collect::<Vec<_>>();
            types.sort_by(|left, right| left.id.cmp(&right.id));

            let pipelines =
                module.bindings.pipelines.as_ref().map(|pipeline| {
                    link_pipeline(pipeline, &module.id, &local_to_canonical[index])
                });
            let bindgroups = link_bindgroup_entries(
                &module.bindings.bindgroups,
                &module.id,
                &local_to_canonical[index],
            );

            LinkedModule {
                id: module.id.clone(),
                types,
                local_to_canonical: local_to_canonical[index].clone(),
                pipelines,
                bindgroups,
            }
        })
        .collect();

    Ok(LinkedSchema {
        schema_version: LINKED_SCHEMA_VERSION,
        types,
        modules: linked_modules,
    })
}

fn all_types(bindings: &super::ShaderBindings) -> impl Iterator<Item = &TypeInfo> {
    bindings
        .types
        .iter()
        .chain(bindings.shared.iter().flat_map(|shared| &shared.types))
}

fn find_type<'a>(bindings: &'a super::ShaderBindings, id: &TypeId) -> Option<&'a TypeInfo> {
    all_types(bindings).find(|ty| &ty.id == id)
}

fn equivalent_type(
    left_module: &ReflectedModule,
    left_id: &TypeId,
    right_module: &ReflectedModule,
    right_id: &TypeId,
    seen: &mut HashSet<(TypeId, TypeId)>,
) -> bool {
    if !seen.insert((left_id.clone(), right_id.clone())) {
        return true;
    }
    let (Some(left), Some(right)) = (
        find_type(&left_module.bindings, left_id),
        find_type(&right_module.bindings, right_id),
    ) else {
        return false;
    };

    let left_identity = left_module.identities.get(left_id);
    let right_identity = right_module.identities.get(right_id);
    match (left_identity, right_identity) {
        (Some(left), Some(right)) if left == right => {}
        (None, None) => {}
        _ => return false,
    }

    match (&left.descriptor, &right.descriptor) {
        (TypeDescriptor::Scalar { scalar: a }, TypeDescriptor::Scalar { scalar: b })
        | (TypeDescriptor::Atomic { scalar: a }, TypeDescriptor::Atomic { scalar: b }) => a == b,
        (
            TypeDescriptor::Vector {
                length: al,
                scalar: a,
            },
            TypeDescriptor::Vector {
                length: bl,
                scalar: b,
            },
        ) => al == bl && a == b,
        (
            TypeDescriptor::Matrix {
                columns: ac,
                rows: ar,
                scalar: a,
            },
            TypeDescriptor::Matrix {
                columns: bc,
                rows: br,
                scalar: b,
            },
        ) => ac == bc && ar == br && a == b,
        (
            TypeDescriptor::Array {
                base: a,
                size: asize,
                stride: astride,
            },
            TypeDescriptor::Array {
                base: b,
                size: bsize,
                stride: bstride,
            },
        ) => {
            asize == bsize
                && astride == bstride
                && equivalent_type(left_module, a, right_module, b, seen)
        }
        (
            TypeDescriptor::BindingArray {
                base: a,
                size: asize,
            },
            TypeDescriptor::BindingArray {
                base: b,
                size: bsize,
            },
        ) => asize == bsize && equivalent_type(left_module, a, right_module, b, seen),
        (TypeDescriptor::Pointer { base: a }, TypeDescriptor::Pointer { base: b }) => {
            equivalent_type(left_module, a, right_module, b, seen)
        }
        (
            TypeDescriptor::Struct {
                size: asize,
                alignment: aalign,
                members: a,
            },
            TypeDescriptor::Struct {
                size: bsize,
                alignment: balign,
                members: b,
            },
        ) => {
            asize == bsize
                && aalign == balign
                && a.len() == b.len()
                && a.iter().zip(b).all(|(a, b)| {
                    member_attributes_equal(a, b)
                        && equivalent_type(left_module, &a.ty, right_module, &b.ty, seen)
                })
        }
        (TypeDescriptor::Sampler { comparison: a }, TypeDescriptor::Sampler { comparison: b }) => {
            a == b
        }
        (
            TypeDescriptor::Image {
                dimension: ad,
                arrayed: aa,
                class: ac,
            },
            TypeDescriptor::Image {
                dimension: bd,
                arrayed: ba,
                class: bc,
            },
        ) => ad == bd && aa == ba && ac == bc,
        _ => false,
    }
}

fn member_attributes_equal(left: &MemberDescriptor, right: &MemberDescriptor) -> bool {
    left.name == right.name
        && left.offset == right.offset
        && left.location == right.location
        && left.builtin == right.builtin
        && left.interpolation == right.interpolation
        && left.sampling == right.sampling
        && left.invariant == right.invariant
}

fn link_reference(
    id: &TypeId,
    module_id: &str,
    local_to_canonical: &BTreeMap<TypeId, TypeId>,
) -> LinkedTypeReference {
    local_to_canonical
        .get(id)
        .cloned()
        .map(LinkedTypeReference::Canonical)
        .unwrap_or_else(|| LinkedTypeReference::Local {
            module_id: module_id.to_owned(),
            type_id: id.clone(),
        })
}

fn link_descriptor(
    descriptor: &TypeDescriptor,
    module: &ReflectedModule,
    local_to_canonical: &BTreeMap<TypeId, TypeId>,
) -> LinkedTypeDescriptor {
    let reference = |id: &TypeId| link_reference(id, &module.id, local_to_canonical);
    match descriptor {
        TypeDescriptor::Scalar { scalar } => LinkedTypeDescriptor::Scalar {
            scalar: scalar.clone(),
        },
        TypeDescriptor::Vector { length, scalar } => LinkedTypeDescriptor::Vector {
            length: *length,
            scalar: scalar.clone(),
        },
        TypeDescriptor::Matrix {
            columns,
            rows,
            scalar,
        } => LinkedTypeDescriptor::Matrix {
            columns: *columns,
            rows: *rows,
            scalar: scalar.clone(),
        },
        TypeDescriptor::Atomic { scalar } => LinkedTypeDescriptor::Atomic {
            scalar: scalar.clone(),
        },
        TypeDescriptor::Array { base, size, stride } => LinkedTypeDescriptor::Array {
            base: reference(base),
            size: *size,
            stride: *stride,
        },
        TypeDescriptor::BindingArray { base, size } => LinkedTypeDescriptor::BindingArray {
            base: reference(base),
            size: *size,
        },
        TypeDescriptor::Struct {
            size,
            alignment,
            members,
        } => LinkedTypeDescriptor::Struct {
            size: *size,
            alignment: *alignment,
            members: members
                .iter()
                .map(|member| LinkedMemberDescriptor {
                    name: member.name.clone(),
                    offset: member.offset,
                    ty: reference(&member.ty),
                    location: member.location,
                    builtin: member.builtin.clone(),
                    interpolation: member.interpolation,
                    sampling: member.sampling,
                    invariant: member.invariant,
                })
                .collect(),
        },
        TypeDescriptor::Pointer { base } => LinkedTypeDescriptor::Pointer {
            base: reference(base),
        },
        TypeDescriptor::Sampler { comparison } => LinkedTypeDescriptor::Sampler {
            comparison: *comparison,
        },
        TypeDescriptor::Image {
            dimension,
            arrayed,
            class,
        } => LinkedTypeDescriptor::Image {
            dimension: dimension.clone(),
            arrayed: *arrayed,
            class: class.clone(),
        },
    }
}

fn remap_type_id(id: &mut TypeId, local_to_canonical: &BTreeMap<TypeId, TypeId>) {
    if let Some(canonical) = local_to_canonical.get(id) {
        *id = canonical.clone();
    }
}

fn link_bindgroup_entries(
    entries: &[BindGroupLayoutEntry],
    module_id: &str,
    local_to_canonical: &BTreeMap<TypeId, TypeId>,
) -> Vec<LinkedBindGroupLayoutEntry> {
    entries
        .iter()
        .map(|entry| link_bindgroup_entry(entry, module_id, local_to_canonical))
        .collect()
}

fn link_bindgroup_entry(
    entry: &BindGroupLayoutEntry,
    module_id: &str,
    local_to_canonical: &BTreeMap<TypeId, TypeId>,
) -> LinkedBindGroupLayoutEntry {
    let mut ty = entry.ty.clone();
    if let BindingType::Buffer { ty, .. } = &mut ty
        && let Some(canonical) = local_to_canonical.get(&TypeId(ty.clone()))
    {
        *ty = canonical.to_string();
    }

    LinkedBindGroupLayoutEntry {
        binding: entry.binding,
        name: entry.name.clone(),
        visibility: entry.visibility.clone(),
        ty,
        address_space: entry.address_space,
        access: entry.access,
        resource_type: link_reference(&entry.resource_type, module_id, local_to_canonical),
        count: entry.count,
    }
}

fn link_pipeline(
    pipeline: &PipelineDescriptor,
    module_id: &str,
    local_to_canonical: &BTreeMap<TypeId, TypeId>,
) -> LinkedPipelineDescriptor {
    let mut states = pipeline.clone();
    remap_pipeline_states(&mut states, local_to_canonical);
    LinkedPipelineDescriptor {
        layout: LinkedPipelineLayout {
            bind_group_layouts: pipeline
                .layout
                .bind_group_layouts
                .iter()
                .map(|group| LinkedBindGroupLayout {
                    group: group.group,
                    entries: link_bindgroup_entries(&group.entries, module_id, local_to_canonical),
                })
                .collect(),
            push_constant_ranges: pipeline.layout.push_constant_ranges.clone(),
        },
        vertex: states.vertex,
        fragment: states.fragment,
        compute: states.compute,
    }
}

fn remap_pipeline_states(
    pipeline: &mut PipelineDescriptor,
    local_to_canonical: &BTreeMap<TypeId, TypeId>,
) {
    if let Some(vertex) = &mut pipeline.vertex {
        for input in &mut vertex.inputs {
            remap_type_id(&mut input.ty, local_to_canonical);
        }
        for output in &mut vertex.outputs {
            remap_type_id(&mut output.ty, local_to_canonical);
        }
        for buffer in vertex.buffers.iter_mut().flatten() {
            for attribute in &mut buffer.attributes {
                remap_type_id(&mut attribute.ty, local_to_canonical);
            }
        }
    }
    if let Some(fragment) = &mut pipeline.fragment {
        for variable in fragment.inputs.iter_mut().chain(&mut fragment.outputs) {
            remap_type_id(&mut variable.ty, local_to_canonical);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeclarationIdentity, ShaderMetadata, naga};

    fn reflected(
        module_id: &str,
        source: &str,
        identities: &[(&str, &str, &str)],
    ) -> ReflectedModule {
        let module = naga::front::wgsl::parse_str(source).expect("WGSL should parse");
        let identity_by_handle = module
            .types
            .iter()
            .filter_map(|(handle, ty)| {
                let name = ty.name.as_deref()?;
                let (_, module_path, declaration) = identities
                    .iter()
                    .find(|(type_name, _, _)| *type_name == name)?;
                Some((
                    handle,
                    DeclarationIdentity {
                        module_path: (*module_path).to_owned(),
                        name: (*declaration).to_owned(),
                    },
                ))
            })
            .collect();
        ShaderMetadata::new(&module)
            .expect("Naga module should reflect")
            .generate_reflected_module(module_id.to_owned(), identity_by_handle)
            .expect("reflected module should build")
    }

    fn canonical_named<'a>(schema: &'a LinkedSchema, name: &str) -> Vec<&'a LinkedTypeInfo> {
        schema.types.iter().filter(|ty| ty.name == name).collect()
    }

    const CAMERA: &str = r"
struct Camera {
    view: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;
";

    #[test]
    fn same_declaration_identity_is_linked_once() {
        let identity = [("Camera", "package::common", "Camera")];
        let first = reflected("package::first", CAMERA, &identity);
        let second = reflected("package::second", CAMERA, &identity);
        let first_id = first
            .bindings
            .types
            .iter()
            .find(|ty| ty.name == "Camera")
            .expect("first module should reflect Camera")
            .id
            .clone();
        let second_id = second
            .bindings
            .types
            .iter()
            .find(|ty| ty.name == "Camera")
            .expect("second module should reflect Camera")
            .id
            .clone();
        let schema = link_modules(vec![first, second]).expect("matching definitions should link");

        assert_eq!(canonical_named(&schema, "Camera").len(), 1);
        let first_mapping = &schema.modules[0].local_to_canonical[&first_id];
        let second_mapping = &schema.modules[1].local_to_canonical[&second_id];
        assert_eq!(first_mapping, second_mapping);
    }

    #[test]
    fn one_declaration_identity_with_different_definitions_fails() {
        let first = reflected(
            "package::first",
            CAMERA,
            &[("Camera", "package::common", "Camera")],
        );
        let second = reflected(
            "package::second",
            r"
struct Camera {
    view: mat4x4<f32>,
    position: vec4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;
",
            &[("Camera", "package::common", "Camera")],
        );

        let error = link_modules(vec![first, second]).expect_err("definitions should conflict");
        assert!(matches!(error, LinkError::ConflictingTypeDefinition { .. }));
    }

    #[test]
    fn different_declaration_identities_do_not_merge_equal_layouts() {
        let schema = link_modules(vec![
            reflected(
                "package::first",
                CAMERA,
                &[("Camera", "package::first", "Camera")],
            ),
            reflected(
                "package::second",
                CAMERA,
                &[("Camera", "package::second", "Camera")],
            ),
        ])
        .expect("different declarations should link independently");

        assert_eq!(canonical_named(&schema, "Camera").len(), 2);
    }

    #[test]
    fn nested_structs_and_arrays_keep_module_local_wrapper_types() {
        let source = r"
struct Camera {
    view: mat4x4<f32>,
}
struct Frame {
    cameras: array<Camera, 2>,
}
@group(0) @binding(0) var<uniform> frame: Frame;
";
        let identities = [
            ("Camera", "package::common", "Camera"),
            ("Frame", "package::frames", "Frame"),
        ];
        let schema = link_modules(vec![
            reflected("package::first", source, &identities),
            reflected("package::second", source, &identities),
        ])
        .expect("nested declarations should link");

        assert_eq!(canonical_named(&schema, "Camera").len(), 1);
        assert_eq!(canonical_named(&schema, "Frame").len(), 1);
        let frame = canonical_named(&schema, "Frame")[0];
        let LinkedTypeDescriptor::Struct { members, .. } = &frame.descriptor else {
            panic!("Frame should be a struct");
        };
        let LinkedTypeReference::Local {
            module_id,
            type_id: array_id,
        } = &members[0].ty
        else {
            panic!("the array wrapper should remain module-local");
        };
        let owner = schema
            .modules
            .iter()
            .find(|module| &module.id == module_id)
            .expect("canonical member should identify its local owner");
        let array = owner
            .types
            .iter()
            .find(|ty| &ty.id == array_id)
            .expect("array wrapper should remain in its owner module");
        let LinkedTypeDescriptor::Array { base, .. } = &array.descriptor else {
            panic!("cameras member should be an array");
        };
        assert_eq!(
            base,
            &LinkedTypeReference::Canonical(canonical_named(&schema, "Camera")[0].id.clone())
        );
    }

    #[test]
    fn types_without_identity_remain_module_local() {
        let schema = link_modules(vec![
            reflected("package::first", CAMERA, &[]),
            reflected("package::second", CAMERA, &[]),
        ])
        .expect("unidentified WGSL types should remain valid");

        assert!(canonical_named(&schema, "Camera").is_empty());
        assert!(
            schema
                .modules
                .iter()
                .all(|module| module.types.iter().any(|ty| ty.name == "Camera"))
        );
    }

    #[test]
    fn linked_bindings_reference_canonical_buffers_and_module_local_images() {
        let camera_with_bindings = |texture: &str, sampler: &str, texture_type: &str| {
            format!(
                r"
struct Camera {{ view: mat4x4<f32>, }}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var {texture}: {texture_type};
@group(0) @binding(2) var {sampler}: sampler;
"
            )
        };
        let identity = [("Camera", "package::common", "Camera")];
        let schema = link_modules(vec![
            reflected(
                "package::first",
                &camera_with_bindings("albedo", "linear_sampler", "texture_2d<f32>"),
                &identity,
            ),
            reflected(
                "package::second",
                &camera_with_bindings("environment", "shadow_sampler", "texture_cube<i32>"),
                &identity,
            ),
        ])
        .expect("module resources should link independently");

        assert_eq!(canonical_named(&schema, "Camera").len(), 1);
        for module in &schema.modules {
            let entries = &module
                .pipelines
                .as_ref()
                .expect("reflected pipeline")
                .layout
                .bind_group_layouts[0]
                .entries;
            assert_eq!(entries.len(), 3);
            assert!(matches!(
                &entries[0].resource_type,
                LinkedTypeReference::Canonical(_)
            ));
            for entry in &entries[1..] {
                let LinkedTypeReference::Local { module_id, type_id } = &entry.resource_type else {
                    panic!("image and sampler types should stay module-local");
                };
                assert_eq!(module_id, &module.id);
                assert!(module.types.iter().any(|ty| &ty.id == type_id));
            }
        }
        assert_eq!(
            schema.modules[0]
                .pipelines
                .as_ref()
                .unwrap()
                .layout
                .bind_group_layouts[0]
                .entries[1]
                .name,
            "albedo"
        );
        assert_eq!(
            schema.modules[1]
                .pipelines
                .as_ref()
                .unwrap()
                .layout
                .bind_group_layouts[0]
                .entries[1]
                .name,
            "environment"
        );
    }
}
