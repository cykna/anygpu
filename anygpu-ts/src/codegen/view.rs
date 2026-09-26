use std::collections::BTreeSet;

use anygpu::{MemberDescriptor, TypeDescriptor, TypeInfo};
use color_eyre::eyre::{Result, eyre};

use crate::codegen::registry::Registry;
use crate::codegen::scalar::{self, ScalarLayout};

/// WGSL swizzle letters, in memory order.
const COMPONENTS: [&str; 4] = ["x", "y", "z", "w"];

pub fn component(length: u8, index: usize) -> Result<&'static str> {
    COMPONENTS
        .get(index)
        .copied()
        .ok_or_else(|| eyre!("vector of length {length} is not supported (expected 1..=4)"))
}

/// A resolved view over one occurrence of a schema type.
///
/// `View` is the single source of truth for everything an emitter needs to know
/// about a type: its byte footprint, the typed array that backs it, the
/// TypeScript type that exposes it, and how to read it out of a value. Emitters
/// never re-derive layout themselves, so adding an output style (getters over
/// `buf.subarray(..)`, for instance) only means consuming this model.
///
/// `size`, `alignment`, `offset` and `stride` are not read by the current
/// emitter, which packs values tightly; they are the layout surface the
/// accessors need to slice a backing buffer, and are covered by the tests.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct View {
    pub name: String,
    pub kind: ViewKind,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum ViewKind {
    Scalar(ScalarLayout),
    Vector {
        length: u8,
        scalar: ScalarLayout,
    },
    Matrix {
        columns: u8,
        rows: u8,
        scalar: ScalarLayout,
    },
    Array {
        size: Option<u32>,
        stride: u32,
        element: Box<View>,
    },
    Atomic(ScalarLayout),
    Struct(StructView),
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct StructView {
    pub size: u32,
    pub alignment: u32,
    pub members: Vec<Member>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Member {
    pub name: String,
    pub offset: u32,
    pub view: View,
}

impl View {
    pub fn build(ty: &TypeInfo) -> Result<Self> {
        let kind = match &ty.descriptor {
            TypeDescriptor::Scalar { scalar } => ViewKind::Scalar(scalar::layout(scalar)?),
            TypeDescriptor::Vector { length, scalar } => {
                for index in 0..usize::from(*length) {
                    component(*length, index)?;
                }
                ViewKind::Vector {
                    length: *length,
                    scalar: scalar::layout(scalar)?,
                }
            }
            TypeDescriptor::Matrix {
                columns,
                rows,
                scalar,
            } => {
                for row in 0..usize::from(*rows) {
                    component(*rows, row)?;
                }
                ViewKind::Matrix {
                    columns: *columns,
                    rows: *rows,
                    scalar: scalar::layout(scalar)?,
                }
            }
            TypeDescriptor::Array { base, size, stride } => ViewKind::Array {
                size: *size,
                stride: *stride,
                element: Box::new(View::build(base)?),
            },
            TypeDescriptor::Atomic { scalar } => ViewKind::Atomic(scalar::layout(scalar)?),
            TypeDescriptor::Struct {
                size,
                alignment,
                members,
            } => ViewKind::Struct(StructView {
                size: *size,
                alignment: *alignment,
                members: members
                    .iter()
                    .enumerate()
                    .map(|(index, member)| {
                        Ok(Member {
                            name: member_name(member, index),
                            offset: member.offset,
                            view: View::build(&member.ty)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            }),
            other => {
                return Err(eyre!(
                    "`{}` has descriptor {other:?}, which has no view representation",
                    ty.name
                ));
            }
        };
        Ok(Self {
            name: ty.name.clone(),
            kind,
        })
    }

    pub fn as_struct(&self) -> Option<&StructView> {
        match &self.kind {
            ViewKind::Struct(view) => Some(view),
            _ => None,
        }
    }

    /// Byte footprint, including any trailing padding declared by the layout.
    #[allow(dead_code)]
    pub fn byte_size(&self) -> Result<u32> {
        Ok(match &self.kind {
            ViewKind::Scalar(scalar) | ViewKind::Atomic(scalar) => scalar.byte_size,
            ViewKind::Vector { length, scalar } => u32::from(*length) * scalar.byte_size,
            ViewKind::Matrix {
                columns,
                rows,
                scalar,
            } => u32::from(*columns) * u32::from(*rows) * scalar.byte_size,
            ViewKind::Array { size, stride, .. } => {
                let size =
                    size.ok_or_else(|| eyre!("`{}` has a runtime-sized array", self.name))?;
                stride * size
            }
            ViewKind::Struct(view) => view.size,
        })
    }

    /// The typed array constructor that backs this view's storage.
    pub fn storage(&self) -> Result<&'static str> {
        Ok(self.storage_layout()?.array)
    }

    /// The scalar layout that backs this view's storage.
    pub fn storage_layout(&self) -> Result<ScalarLayout> {
        let mut storage = None;
        self.collect_storage(&mut storage)?;
        storage.ok_or_else(|| eyre!("`{}` contains no scalars to back", self.name))
    }

    /// The width in bytes of one element of the backing typed array.
    pub fn element_width(&self) -> Result<u32> {
        Ok(self.storage_layout()?.byte_size)
    }

    /// The number of elements the backing typed array needs to hold this view.
    pub fn element_count(&self) -> Result<u32> {
        Ok(self.byte_size()? / self.element_width()?)
    }

    /// Converts a byte offset from the schema into an index into the backing
    /// typed array. The schema reports offsets in bytes, but `Float32Array` and
    /// friends are indexed by element, so every offset has to be scaled down by
    /// the element width.
    pub fn element_offset(&self, byte_offset: u32) -> Result<u32> {
        let width = self.element_width()?;
        if byte_offset % width != 0 {
            return Err(eyre!(
                "offset {byte_offset} in `{}` is not a multiple of the {width}-byte element size",
                self.name
            ));
        }
        Ok(byte_offset / width)
    }

    /// The TypeScript type an emitter uses for this view.
    pub fn ts_type(&self) -> Result<String> {
        Ok(match &self.kind {
            ViewKind::Scalar(scalar) | ViewKind::Atomic(scalar) => scalar.ts_type.to_string(),
            ViewKind::Vector { length, scalar } => {
                format!("Vector{length}{}", scalar.suffix)
            }
            ViewKind::Matrix {
                columns,
                rows,
                scalar,
            } => format!("Mat{columns}x{rows}{}", scalar.suffix),
            ViewKind::Array { size, element, .. } => {
                let element = element.ts_type()?;
                match size {
                    Some(size) => format!("Array<{element}, {size}>"),
                    None => format!("Array<{element}>"),
                }
            }
            ViewKind::Struct(_) => identifier(&self.name),
        })
    }

    /// Registers this view and everything nested inside it as a type that needs
    /// to be declared.
    pub fn register(&self, registry: &mut Registry) -> Result<()> {
        match &self.kind {
            ViewKind::Scalar(_) | ViewKind::Atomic(_) => {}
            ViewKind::Vector { length, scalar } => {
                registry.add_vector(*length, *scalar);
            }
            ViewKind::Matrix {
                columns,
                rows,
                scalar,
            } => {
                registry.add_matrix(*columns, *rows, *scalar);
            }
            ViewKind::Array { element, .. } => element.register(registry)?,
            ViewKind::Struct(_) => registry.add_struct(self.clone()),
        }
        Ok(())
    }

    fn collect_storage(&self, storage: &mut Option<ScalarLayout>) -> Result<()> {
        match &self.kind {
            ViewKind::Scalar(scalar) | ViewKind::Atomic(scalar) => {
                merge_storage(storage, *scalar, &self.name)
            }
            ViewKind::Vector { scalar, .. } | ViewKind::Matrix { scalar, .. } => {
                merge_storage(storage, *scalar, &self.name)
            }
            ViewKind::Array { element, .. } => element.collect_storage(storage),
            ViewKind::Struct(view) => {
                for member in &view.members {
                    member.view.collect_storage(storage)?;
                }
                Ok(())
            }
        }
    }

    /// Every scalar width backing this view, used to decide element counts.
    #[allow(dead_code)]
    pub fn scalar_widths(&self) -> BTreeSet<u32> {
        match &self.kind {
            ViewKind::Scalar(scalar) | ViewKind::Atomic(scalar) => {
                BTreeSet::from([scalar.byte_size])
            }
            ViewKind::Vector { scalar, .. } | ViewKind::Matrix { scalar, .. } => {
                BTreeSet::from([scalar.byte_size])
            }
            ViewKind::Array { element, .. } => element.scalar_widths(),
            ViewKind::Struct(view) => view
                .members
                .iter()
                .flat_map(|member| member.view.scalar_widths())
                .collect(),
        }
    }
}

fn merge_storage(
    storage: &mut Option<ScalarLayout>,
    scalar: ScalarLayout,
    name: &str,
) -> Result<()> {
    match storage {
        Some(existing) if *existing != scalar => Err(eyre!(
            "`{name}` mixes `{existing:?}` and `{scalar:?}` storage, which cannot share one buffer"
        )),
        Some(_) => Ok(()),
        None => {
            *storage = Some(scalar);
            Ok(())
        }
    }
}

fn member_name(member: &MemberDescriptor, index: usize) -> String {
    if member.name.is_empty() {
        return format!("m{index}");
    }
    identifier(&member.name)
}

/// Turns a schema name into a legal TypeScript identifier.
pub fn identifier(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 1);
    for (index, ch) in raw.chars().enumerate() {
        let valid = if index == 0 {
            ch.is_ascii_alphabetic() || ch == '_' || ch == '$'
        } else {
            ch.is_ascii_alphanumeric() || ch == '_' || ch == '$'
        };
        out.push(if valid { ch } else { '_' });
    }
    if out.is_empty() || out.starts_with(|ch: char| ch.is_ascii_digit()) {
        out.insert(0, '_');
    }
    out
}
