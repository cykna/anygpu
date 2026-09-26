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

/// Emits a factory that adopts an existing buffer as a view.
///
/// Struct classes have to take their members and allocate a buffer, so they
/// cannot be built with `new X(subarray)` the way a `Vector` can. Every
/// buffer-wrapping class gets the same static entry point instead, so "a view
/// over someone else's buffer is always made with `.view(..)`" holds everywhere.
fn emit_view_factory(code: &mut CodeBuilder, name: &str, array: &str) {
    code.line(&format!(
        "static view(buffer: {array}): {name} {{ const self = Object.create({name}.prototype) as {name}; self.buffer = buffer; return self; }}"
    ));
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
    let name = format!("Vector{length}{}", scalar.suffix);
    let accessors = (0..usize::from(length))
        .map(|index| component(length, index))
        .collect::<Result<Vec<_>>>()?;

    code.line(&format!("export class {name} {{"));
    code.indented(|code| {
        code.line(&format!("buffer: {};", scalar.array));
        code.line(&format!(
            "constructor(buffer: {}) {{ this.buffer = buffer; }}",
            scalar.array
        ));
        emit_view_factory(code, &name, scalar.array);
        for (index, accessor) in accessors.iter().enumerate() {
            emit_component(code, scalar, accessor, index as u32);
        }
        Ok(())
    })?;
    code.line("}");
    Ok(())
}

fn emit_matrix(code: &mut CodeBuilder, columns: u8, rows: u8, scalar: ScalarLayout) -> Result<()> {
    let name = format!("Mat{columns}x{rows}{}", scalar.suffix);
    code.line(&format!("export class {name} {{"));
    code.indented(|code| {
        code.line(&format!("buffer: {};", scalar.array));
        code.line(&format!(
            "constructor(buffer: {}) {{ this.buffer = buffer; }}",
            scalar.array
        ));
        emit_view_factory(code, &name, scalar.array);
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

/// Emits the accessor class for an array of `base`.
///
/// The count and the stride are deliberately *not* baked into the class: the
/// stride is a constructor argument and the count is derived from the buffer, so
/// one class serves every `array<Base, N>` in the schema.
fn emit_array(code: &mut CodeBuilder, name: &str, base: &View) -> Result<()> {
    let base_name = base.ts_type()?;
    let array = base.storage()?;

    code.line(&format!("export class {name} {{"));
    code.indented(|code| {
        code.line(&format!("buffer: {array};"));
        code.line("stride: number; // em elementos");
        code.line(&format!(
            "constructor(buffer: {array}, stride: number) {{ this.buffer = buffer; this.stride = stride; }}"
        ));
        emit_view_factory(code, name, array);
        code.line(&format!(
            "get(index: number): {base_name} {{ const start = index * this.stride; return {base_name}.view(this.buffer.subarray(start, start + this.stride)); }}"
        ));
        code.line("get length(): number { return this.buffer.length / this.stride; }");
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
    for (name, base) in registry.arrays() {
        code.blank();
        emit_array(code, name, base)?;
    }
    Ok(())
}

/// How one struct member reads from, and writes into, the struct's own buffer.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Access {
    /// A single element, e.g. `get intensity(): number { return this.buffer[3]; }`.
    Element { index: u32 },
    /// A nested view class over `[start, end)`, e.g. `get position(): Vector4f32`.
    View { start: u32, end: u32 },
    /// An array view, e.g. `get items(): Array<Item> { return this.buffer[4..]; }`.
    Array {
        start: u32,
        stride: u32,
        count: Option<usize>,
    },
}

/// Resolves a member against the struct that owns it.
///
/// The schema reports `offset` in bytes, so the offset is converted into an
/// index into the owning struct's backing array; the member's own `byte_size` is
/// converted the same way to get the `end` of a view.
pub(crate) fn member_access(owner: &View, member: &Member) -> Result<Access> {
    let start = owner.element_offset(member.offset)?;
    Ok(match &member.view.kind {
        ViewKind::Scalar(_) | ViewKind::Atomic(_) => Access::Element { index: start },
        ViewKind::Vector { .. } | ViewKind::Matrix { .. } | ViewKind::Struct(_) => Access::View {
            start,
            end: start + member.view.element_count()?,
        },
        ViewKind::Array { stride, size, .. } => Access::Array {
            start,
            stride: owner.element_offset(*stride)?, // mesma conversão bytes->elementos que offset usa
            count: size.map(|n| n as usize),
        },
        ViewKind::Opaque { .. } => {
            return Err(eyre!(
                "member `{}` is an {:?}, which cannot be exposed as a view yet",
                member.name,
                member.view.name
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
        code.line(&format!("public buffer: {array};"));
        emit_view_factory(code, &name, array);

        for member in members {
            let ty = member.view.ts_type()?;
            let body = match member_access(view, member)? {
                Access::Element { index } => {
                    format!("return this.buffer[{index}];")
                }
                Access::View { start, end } => {
                    format!("return {ty}.view(this.buffer.subarray({start}, {end}));")
                }
                // A runtime-sized array runs to the end of the struct's buffer, so
                // it is left unbounded; a fixed one is clipped to its own count.
                Access::Array {
                    start,
                    stride,
                    count,
                } => {
                    let window = match count {
                        Some(count) => format!("{start}, {}", start + stride * count as u32),
                        None => format!("{start}"),
                    };
                    format!("return new {ty}(this.buffer.subarray({window}), {stride});")
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
            // A runtime-sized array has no length in the schema: it runs to the end
            // of the buffer, so the allocation has to grow by whatever the caller
            // passed in. `Math.max` keeps the struct's declared span, which naga
            // measures including one padding element for the runtime array, from
            // showing up as a phantom element ahead of an array that starts at 0.
            let mut elements = (total_bytes / width).to_string();
            if let Some(dynamic) = members
                .iter()
                .find(|member| matches!(member.view.kind, ViewKind::Array { size: None, .. }))
            {
                let start = view.element_offset(dynamic.offset)?;
                let supplied = if start == 0 {
                    format!("{}.buffer.length", dynamic.name)
                } else {
                    format!("{start} + {}.buffer.length", dynamic.name)
                };
                elements = format!("Math.max({elements}, {supplied})");
            }
            code.line(&format!(
                "this.buffer = new {array}({elements}); // {total_bytes} bytes / {width}"
            ));
            for member in members {
                match member_access(view, member)? {
                    Access::Element { index } => {
                        code.line(&format!("this.buffer[{index}] = {};", member.name));
                    }
                    Access::View { start, .. } | Access::Array { start, .. } => code.line(
                        &format!(
                            "this.buffer.set({}.buffer, {start}); // offset {} bytes / {width} = {start}",
                            member.name, member.offset
                        ),
                    ),
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
