use color_eyre::eyre::{Result, eyre};

use crate::codegen::registry::Registry;
use crate::codegen::scalar::ScalarLayout;
use crate::codegen::view::{Member, View, ViewKind, component};

const INDENT: &str = "  ";

/// Indentation-aware text builder.
#[derive(Debug, Default, Clone)]
pub struct CodeBuilder {
    out: String,
    depth: usize,
}

impl CodeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn blank(&mut self) {
        if !self.out.is_empty() {
            self.out.push('\n');
        }
    }

    pub fn line(&mut self, text: &str) {
        for (index, part) in text.split('\n').enumerate() {
            if index > 0 {
                self.out.push('\n');
            }
            if !part.is_empty() {
                self.out.push_str(&INDENT.repeat(self.depth));
                self.out.push_str(part);
            }
        }
        self.out.push('\n');
    }

    pub fn indented(&mut self, body: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        self.depth += 1;
        let result = body(self);
        self.depth -= 1;
        result
    }

    pub fn build(self) -> String {
        self.out
    }

    pub fn emit_views(&mut self, registry: &Registry) -> Result<()> {
        emit_views(self, registry)
    }

    pub fn emit_struct(&mut self, view: &View) -> Result<()> {
        emit_struct(self, view)
    }
}

/// Emits a `get`/`set` pair for one component of a view's own buffer.
///
/// Vectors index by component, matrices by `column * rows + row` so the layout
/// stays column-major, matching WGSL.
fn emit_component(code: &mut CodeBuilder, scalar: ScalarLayout, accessor: &str, index: u32) {
    code.line(&format!(
        "get {accessor}(): {} {{ return this.buffer[{index}]; }}",
        scalar.ts_type
    ));
    code.line(&format!(
        "set {accessor}(v: {}) {{ this.buffer[{index}] = v; }}",
        scalar.ts_type
    ));
}

fn emit_vector(code: &mut CodeBuilder, length: u8, scalar: ScalarLayout) -> Result<()> {
    let accessors = (0..usize::from(length))
        .map(|index| component(length, index))
        .collect::<Result<Vec<_>>>()?;

    code.line(&format!("export class Vector{length}{} {{", scalar.suffix));
    code.indented(|code| {
        code.line(&format!("buffer: {};", scalar.array));
        code.line(&format!(
            "constructor(buffer: {}) {{ this.buffer = buffer; }}",
            scalar.array
        ));
        for (index, accessor) in accessors.iter().enumerate() {
            emit_component(code, scalar, accessor, index as u32);
        }
        Ok(())
    })?;
    code.line("}");
    Ok(())
}

fn emit_matrix(code: &mut CodeBuilder, columns: u8, rows: u8, scalar: ScalarLayout) -> Result<()> {
    code.line(&format!(
        "export class Mat{columns}x{rows}{} {{",
        scalar.suffix
    ));
    code.indented(|code| {
        code.line(&format!("buffer: {};", scalar.array));
        code.line(&format!(
            "constructor(buffer: {}) {{ this.buffer = buffer; }}",
            scalar.array
        ));
        for column in 0..columns {
            for row in 0..rows {
                let index = u32::from(column) * u32::from(rows) + u32::from(row);
                emit_component(code, scalar, &format!("m{column}{row}"), index);
            }
        }
        Ok(())
    })?;
    code.line("}");
    Ok(())
}

fn emit_views(code: &mut CodeBuilder, registry: &Registry) -> Result<()> {
    for (length, scalar) in registry.vectors() {
        code.blank();
        emit_vector(code, length, scalar)?;
    }
    for (columns, rows, scalar) in registry.matrices() {
        code.blank();
        emit_matrix(code, columns, rows, scalar)?;
    }
    Ok(())
}

/// How one struct member reads from, and writes into, the struct's own buffer.
enum Access {
    /// A single element, e.g. `get intensity(): number { return this.buffer[3]; }`.
    Element { index: u32 },
    /// A nested view class over `[start, end)`, e.g. `get position(): Vector4f32`.
    View { start: u32, end: u32 },
}

/// Resolves a member against the struct that owns it.
///
/// The schema reports `offset` in bytes, so the offset is converted into an
/// index into the owning struct's backing array; the member's own `byte_size` is
/// converted the same way to get the `end` of a view.
fn member_access(owner: &View, member: &Member) -> Result<Access> {
    let start = owner.element_offset(member.offset)?;
    Ok(match &member.view.kind {
        ViewKind::Scalar(_) | ViewKind::Atomic(_) => Access::Element { index: start },
        ViewKind::Vector { .. } | ViewKind::Matrix { .. } | ViewKind::Struct(_) => Access::View {
            start,
            end: start + member.view.element_count()?,
        },
        ViewKind::Array { .. } => {
            return Err(eyre!(
                "member `{}` is an array, which cannot be exposed as a view yet",
                member.name
            ));
        }
    })
}

fn emit_struct(code: &mut CodeBuilder, view: &View) -> Result<()> {
    let name = view.ts_type()?;
    let members = &view
        .as_struct()
        .ok_or_else(|| eyre!("`{name}` is not a struct"))?
        .members;

    let array = view.storage()?;
    let width = view.element_width()?;
    let total_bytes = view.byte_size()?;

    code.blank();
    code.line(&format!("export class {name} {{"));
    code.indented(|code| {
        code.line(&format!("buffer: {array};"));

        for member in members {
            let ty = member.view.ts_type()?;
            let body = match member_access(view, member)? {
                Access::Element { index } => {
                    format!("return this.buffer[{index}];")
                }
                Access::View { start, end } => {
                    format!("return new {ty}(this.buffer.subarray({start}, {end}));")
                }
            };
            code.line(&format!(
                "get {}(): {ty} {{ {body} }}",
                member.name
            ));
        }

        let params = members
            .iter()
            .map(|member| {
                Ok(format!(
                    "{}: {}",
                    member.name,
                    member.view.ts_type().unwrap_or_default()
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        code.line(&format!("constructor({}) {{", params.join(", ")));
        code.indented(|code| {
            code.line(&format!(
                "this.buffer = new {array}({}); // {total_bytes} bytes / {width}",
                total_bytes / width
            ));
            for member in members {
                match member_access(view, member)? {
                    Access::Element { index } => {
                        code.line(&format!("this.buffer[{index}] = {};", member.name));
                    }
                    Access::View { start, .. } => code.line(&format!(
                        "this.buffer.set({}.buffer, {start}); // offset {} bytes / {width} = {start}",
                        member.name, member.offset
                    )),
                }
            }
            Ok(())
        })?;
        code.line("}");
        Ok(())
    })?;
    code.line("}");
    Ok(())
}
