use std::collections::HashMap;
use std::fmt::{self, Display};

use naga::{ArraySize, Handle, ScalarKind, Type, TypeInner, VectorSize, proc::Alignment};
use serde::{Deserialize, Serialize};

use crate::shaders::{
    ShaderMetadata,
    image::{ImageClass, ImageDimension, StorageAccess, StorageFormat},
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ScalarInfo {
    pub name: String,
    pub width: u8,
}

impl ScalarInfo {
    pub fn from_kind(kind: ScalarKind, width: u8) -> Self {
        let kind_str = match kind {
            naga::ScalarKind::Sint => "sint",
            naga::ScalarKind::Uint => "uint",
            naga::ScalarKind::Float => "float",
            naga::ScalarKind::Bool => "bool",
            _ => unreachable!(),
        };
        ScalarInfo {
            name: kind_str.to_string(),
            width,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Interpolation {
    Perspective,
    Linear,
    Flat,
    PerVertex,
}

impl From<naga::Interpolation> for Interpolation {
    fn from(value: naga::Interpolation) -> Self {
        match value {
            naga::Interpolation::Perspective => Self::Perspective,
            naga::Interpolation::Linear => Self::Linear,
            naga::Interpolation::Flat => Self::Flat,
            naga::Interpolation::PerVertex => Self::PerVertex,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sampling {
    Center,
    Centroid,
    Sample,
    First,
    Either,
}

impl From<naga::Sampling> for Sampling {
    fn from(value: naga::Sampling) -> Self {
        match value {
            naga::Sampling::Center => Self::Center,
            naga::Sampling::Centroid => Self::Centroid,
            naga::Sampling::Sample => Self::Sample,
            naga::Sampling::First => Self::First,
            naga::Sampling::Either => Self::Either,
        }
    }
}

/// The attributes that describe how a value crosses a shader stage boundary.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct BindingAttributes {
    pub location: Option<u32>,
    pub builtin: Option<String>,
    pub interpolation: Option<Interpolation>,
    pub sampling: Option<Sampling>,
    pub invariant: bool,
}

pub(crate) fn binding_attributes(binding: Option<&naga::Binding>) -> BindingAttributes {
    match binding {
        Some(naga::Binding::BuiltIn(builtin)) => BindingAttributes {
            location: None,
            builtin: Some(builtin_name(builtin)),
            interpolation: None,
            sampling: None,
            invariant: matches!(builtin, naga::BuiltIn::Position { invariant: true }),
        },
        Some(naga::Binding::Location {
            location,
            interpolation,
            sampling,
            ..
        }) => BindingAttributes {
            location: Some(*location),
            builtin: None,
            interpolation: interpolation.map(Into::into),
            sampling: sampling.map(Into::into),
            invariant: false,
        },
        None => BindingAttributes::default(),
    }
}

/// The WGSL spelling of a builtin, e.g. `Position` becomes `position`.
pub(crate) fn builtin_name(builtin: &naga::BuiltIn) -> String {
    let debug = format!("{builtin:?}");
    let variant = debug.split([' ', '{']).next().unwrap_or(debug.as_str());
    let chars: Vec<char> = variant.chars().collect();
    let mut name = String::with_capacity(variant.len() + 4);
    for (index, ch) in chars.iter().enumerate() {
        if ch.is_ascii_uppercase() {
            let previous = index.checked_sub(1).map(|index| chars[index]);
            let next = chars.get(index + 1).copied();
            let boundary = index > 0
                && (previous.is_some_and(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit())
                    || (previous.is_some_and(|ch| ch.is_ascii_uppercase())
                        && next.is_some_and(|ch| ch.is_ascii_lowercase())));
            if boundary {
                name.push('_');
            }
            name.push(ch.to_ascii_lowercase());
        } else {
            name.push(*ch);
        }
    }
    name
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemberDescriptor {
    pub name: String,
    pub offset: u32,
    pub ty: TypeId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub builtin: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interpolation: Option<Interpolation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<Sampling>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub invariant: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeDescriptor {
    Scalar {
        scalar: ScalarInfo,
    },
    Vector {
        length: u8,
        scalar: ScalarInfo,
    },
    Matrix {
        columns: u8,
        rows: u8,
        scalar: ScalarInfo,
    },
    Atomic {
        scalar: ScalarInfo,
    },
    Array {
        base: TypeId,
        size: Option<u32>,
        stride: u32,
    },
    BindingArray {
        base: TypeId,
        size: Option<u32>,
    },
    Struct {
        size: u32,
        alignment: u32,
        members: Vec<MemberDescriptor>,
    },
    Pointer {
        base: TypeId,
    },
    Sampler {
        comparison: bool,
    },
    Image {
        dimension: ImageDimension,
        arrayed: bool,
        class: ImageClass,
    },
}

impl TypeDescriptor {
    /// Whether the type is intrinsic to WGSL itself - a scalar, vector, matrix or
    /// atomic - and therefore shared by every shader that names it, rather than
    /// being declared by one shader's source.
    pub fn is_shared(&self) -> bool {
        matches!(
            self,
            TypeDescriptor::Scalar { .. }
                | TypeDescriptor::Vector { .. }
                | TypeDescriptor::Matrix { .. }
                | TypeDescriptor::Atomic { .. }
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TypeId(pub String);

impl TypeId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for TypeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeInfo {
    #[serde(default)]
    pub id: TypeId,
    pub name: String,
    pub descriptor: TypeDescriptor,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DeclarationIdentity {
    pub module_path: String,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinkedTypeReference {
    Canonical(TypeId),
    Local { module_id: String, type_id: TypeId },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkedMemberDescriptor {
    pub name: String,
    pub offset: u32,
    pub ty: LinkedTypeReference,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub builtin: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interpolation: Option<Interpolation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<Sampling>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub invariant: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinkedTypeDescriptor {
    Scalar {
        scalar: ScalarInfo,
    },
    Vector {
        length: u8,
        scalar: ScalarInfo,
    },
    Matrix {
        columns: u8,
        rows: u8,
        scalar: ScalarInfo,
    },
    Atomic {
        scalar: ScalarInfo,
    },
    Array {
        base: LinkedTypeReference,
        size: Option<u32>,
        stride: u32,
    },
    BindingArray {
        base: LinkedTypeReference,
        size: Option<u32>,
    },
    Struct {
        size: u32,
        alignment: u32,
        members: Vec<LinkedMemberDescriptor>,
    },
    Pointer {
        base: LinkedTypeReference,
    },
    Sampler {
        comparison: bool,
    },
    Image {
        dimension: ImageDimension,
        arrayed: bool,
        class: ImageClass,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkedTypeInfo {
    pub id: TypeId,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity: Option<DeclarationIdentity>,
    pub descriptor: LinkedTypeDescriptor,
}

#[derive(Debug, Clone, Default)]
pub struct TypeInterner {
    types: Vec<TypeInfo>,
    by_key: HashMap<(String, TypeDescriptor), TypeId>,
}

impl TypeInterner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Interns one shader-specific type, giving it an id of its own: `t0`, `t1`,
    /// and so on.
    pub fn intern(
        &mut self,
        name: String,
        build: impl FnOnce(&mut Self) -> TypeDescriptor,
    ) -> TypeId {
        self.intern_prefixed(name, build, "t")
    }

    /// Interns one type that is intrinsic to WGSL itself - a scalar, a vector or
    /// a matrix - which every shader shares, giving it an id of its own that says
    /// so: `st0`, `st1`, and so on.
    pub fn intern_shared(
        &mut self,
        name: String,
        build: impl FnOnce(&mut Self) -> TypeDescriptor,
    ) -> TypeId {
        self.intern_prefixed(name, build, "st")
    }

    fn intern_prefixed(
        &mut self,
        name: String,
        build: impl FnOnce(&mut Self) -> TypeDescriptor,
        prefix: &str,
    ) -> TypeId {
        let descriptor = build(self);
        if let Some(id) = self.by_key.get(&(name.clone(), descriptor.clone())) {
            return id.clone();
        }
        let index = self
            .types
            .iter()
            .filter(|ty| ty.id.as_str().starts_with(prefix))
            .count();
        let id = TypeId(format!("{prefix}{index}"));
        self.by_key
            .insert((name.clone(), descriptor.clone()), id.clone());
        self.types.push(TypeInfo {
            id: id.clone(),
            name,
            descriptor,
        });
        id
    }

    /// The interned types, each defined once, in the order they were reached.
    pub fn finish(self) -> Vec<TypeInfo> {
        self.types
    }
}

impl<'a> ShaderMetadata<'a> {
    fn scalar_name(kind: ScalarKind, width: u8) -> String {
        match (kind, width) {
            (ScalarKind::Sint, 4) => "i32".to_string(),
            (ScalarKind::Uint, 4) => "u32".to_string(),
            (ScalarKind::Float, 4) => "f32".to_string(),
            (ScalarKind::Float, 8) => "f64".to_string(),
            (ScalarKind::Bool, _) => "bool".to_string(),
            (kind, width) => format!("{:?}{}", kind, width * 8),
        }
    }

    fn vec_size_n(size: VectorSize) -> u8 {
        match size {
            VectorSize::Bi => 2,
            VectorSize::Tri => 3,
            VectorSize::Quad => 4,
        }
    }

    /// Interns the type at `handle`, together with everything it is built from.
    ///
    /// `memo` keeps one handle from being walked twice. WGSL has no recursive
    /// types, so interning the children before the parent cannot cycle.
    pub fn type_id(
        &self,
        interner: &mut TypeInterner,
        memo: &mut HashMap<Handle<Type>, TypeId>,
        handle: Handle<Type>,
    ) -> TypeId {
        if let Some(id) = memo.get(&handle) {
            return id.clone();
        }
        let name = self.get_type_name(handle);
        let descriptor = self.type_descriptor(interner, memo, handle);
        // Types intrinsic to WGSL itself - scalars, vectors, matrices - are
        // shared by every shader, so they are interned under their own prefix.
        let id = if descriptor.is_shared() {
            interner.intern_shared(name, |_| descriptor)
        } else {
            interner.intern(name, |_| descriptor)
        };
        memo.insert(handle, id.clone());
        id
    }

    /// Describes the type at `handle`, together with everything it is built from.
    ///
    /// The interner is handed to the built type so that whatever it names is
    /// interned first, which keeps children ahead of the types that name them.
    fn type_descriptor(
        &self,
        interner: &mut TypeInterner,
        memo: &mut HashMap<Handle<Type>, TypeId>,
        handle: Handle<Type>,
    ) -> TypeDescriptor {
        let ty = self.module.types.get_handle(handle).unwrap();
        let layout = match &ty.inner {
            TypeInner::Struct { .. } => Some(self.layouter[handle]),
            _ => None,
        };
        let inner = &ty.inner;
        match inner {
            TypeInner::Scalar(s) => TypeDescriptor::Scalar {
                scalar: ScalarInfo::from_kind(s.kind, s.width),
            },

            TypeInner::Vector { size, scalar } => TypeDescriptor::Vector {
                length: Self::vec_size_n(*size),
                scalar: ScalarInfo::from_kind(scalar.kind, scalar.width),
            },

            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => TypeDescriptor::Matrix {
                columns: Self::vec_size_n(*columns),
                rows: Self::vec_size_n(*rows),
                scalar: ScalarInfo::from_kind(scalar.kind, scalar.width),
            },

            TypeInner::Atomic(s) => TypeDescriptor::Atomic {
                scalar: ScalarInfo::from_kind(s.kind, s.width),
            },

            TypeInner::Array { base, size, stride } => TypeDescriptor::Array {
                base: self.type_id(interner, memo, *base),
                size: match size {
                    naga::ArraySize::Constant(n) => Some(n.get()),
                    _ => None,
                },
                stride: *stride,
            },

            TypeInner::Struct { members, span } => {
                let layout = layout.expect("a struct is laid out when it is reached");
                let members = members
                    .iter()
                    .map(|m| {
                        let attributes = binding_attributes(m.binding.as_ref());
                        MemberDescriptor {
                            name: m.name.clone().unwrap_or_default(),
                            offset: m.offset,
                            ty: self.type_id(interner, memo, m.ty),
                            location: attributes.location,
                            builtin: attributes.builtin,
                            interpolation: attributes.interpolation,
                            sampling: attributes.sampling,
                            invariant: attributes.invariant,
                        }
                    })
                    .collect();
                let len: u32 = match layout.alignment {
                    Alignment::ONE => 1,
                    Alignment::TWO => 2,
                    Alignment::FOUR => 4,
                    Alignment::EIGHT => 8,
                    Alignment::SIXTEEN => 16,
                    Alignment::MIN_UNIFORM => Alignment::MIN_UNIFORM.round_up(1),
                    other => unreachable!("Unrecognized alignment: {other:?}"),
                };

                TypeDescriptor::Struct {
                    size: *span,
                    alignment: len,
                    members,
                }
            }

            TypeInner::Pointer { base, .. } => TypeDescriptor::Pointer {
                base: self.type_id(interner, memo, *base),
            },

            TypeInner::Image {
                dim,
                arrayed,
                class,
            } => TypeDescriptor::Image {
                dimension: match dim {
                    naga::ImageDimension::Cube => ImageDimension::Cube,
                    naga::ImageDimension::D1 => ImageDimension::D1,
                    naga::ImageDimension::D2 => ImageDimension::D2,
                    naga::ImageDimension::D3 => ImageDimension::D3,
                },
                arrayed: *arrayed,
                class: match class {
                    naga::ImageClass::Depth { multi } => ImageClass::Depth { multi: *multi },
                    naga::ImageClass::Sampled { kind, multi } => ImageClass::Sampled {
                        kind: ScalarInfo::from_kind(*kind, 0),
                        multi: *multi,
                    },
                    naga::ImageClass::Storage { format, access } => ImageClass::Storage {
                        format: StorageFormat::from(*format),
                        access: StorageAccess::from(*access),
                    },
                    naga::ImageClass::External => ImageClass::External,
                },
            },

            TypeInner::Sampler { comparison } => TypeDescriptor::Sampler {
                comparison: *comparison,
            },

            TypeInner::BindingArray { base, size } => TypeDescriptor::BindingArray {
                base: self.type_id(interner, memo, *base),
                size: match size {
                    naga::ArraySize::Constant(n) => Some(n.get()),
                    naga::ArraySize::Dynamic => None,
                    other => unimplemented!("{:?}", other),
                },
            },

            #[allow(unreachable_patterns)]
            _ => unreachable!("tipo não coberto"),
        }
    }

    pub fn get_type_name(&self, ty: Handle<Type>) -> String {
        let ty = self.module.types.get_handle(ty).unwrap();
        if let Some(ref name) = ty.name {
            name.clone()
        } else {
            match &ty.inner {
                TypeInner::Scalar(scalar) => Self::scalar_name(scalar.kind, scalar.width),

                TypeInner::Vector { size, scalar } => {
                    format!(
                        "vec{}<{}>",
                        Self::vec_size_n(*size),
                        Self::scalar_name(scalar.kind, scalar.width)
                    )
                }

                TypeInner::Matrix {
                    columns,
                    rows,
                    scalar,
                } => {
                    format!(
                        "mat{}x{}<{}>",
                        Self::vec_size_n(*columns),
                        Self::vec_size_n(*rows),
                        Self::scalar_name(scalar.kind, scalar.width)
                    )
                }

                TypeInner::Atomic(scalar) => {
                    format!("atomic<{}>", Self::scalar_name(scalar.kind, scalar.width))
                }

                TypeInner::Pointer { base, .. } => {
                    format!("ptr<{}>", self.get_type_name(*base))
                }

                TypeInner::ValuePointer { scalar, .. } => {
                    Self::scalar_name(scalar.kind, scalar.width)
                }

                TypeInner::Array { base, size, .. } => {
                    let base_name = self.get_type_name(*base);
                    match size {
                        ArraySize::Constant(n) => format!("array<{}, {}>", base_name, n),
                        ArraySize::Dynamic => format!("array<{}>", base_name),
                        ArraySize::Pending(_) => "array<pending>".to_string(),
                    }
                }

                TypeInner::Struct { members, span } => {
                    // sem nome próprio: deriva dos membros e do tamanho, para que
                    // duas structs anônimas diferentes nunca compartilhem nome
                    let members: Vec<String> =
                        members.iter().map(|m| self.get_type_name(m.ty)).collect();
                    format!("anon_struct<{} /* {} bytes */>", members.join(", "), span)
                }

                TypeInner::Image { .. } => "texture".to_string(),
                TypeInner::Sampler { comparison } => {
                    if *comparison {
                        "sampler_comparison".to_string()
                    } else {
                        "sampler".to_string()
                    }
                }
                TypeInner::BindingArray { base, size } => {
                    let base_name = self.get_type_name(*base);
                    match size {
                        ArraySize::Constant(n) => format!("binding_array<{}, {}>", base_name, n),
                        ArraySize::Dynamic => format!("binding_array<{}>", base_name),
                        ArraySize::Pending(_) => "binding_array<pending>".to_string(),
                    }
                }

                _ => "unknown".to_string(),
            }
        }
    }
}
