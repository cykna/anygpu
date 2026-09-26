use anygpu::ScalarInfo;
use color_eyre::eyre::Result;

use crate::codegen::registry::Registry;
use crate::codegen::scalar::ScalarLayout;
use crate::codegen::view::{View, ViewKind, component};

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

    pub fn indented(&mut self, body: impl FnOnce(&mut Self)) {
        self.depth += 1;
        body(self);
        self.depth -= 1;
    }

    pub fn build(self) -> String {
        self.out
    }

    pub fn emit_views(&mut self, registry: &Registry) {
        emit_views(self, registry);
    }
    pub fn emit_struct(&mut self, view: &View) -> color_eyre::Result<()> {
        emit_struct(self, view)
    }
}

fn emit_views(code: &mut CodeBuilder, registry: &Registry) {
    for (length, scalar) in registry.vectors() {
        let fields = (0..usize::from(length))
            .map(|index| {
                let component = component(length, index).unwrap_or("x");
                format!("{component}: {}", scalar.ts_type)
            })
            .collect::<Vec<_>>()
            .join(", ");
        code.line(&format!(
            "export interface Vector{length}{} {{ {fields} }}",
            scalar.suffix
        ));
    }

    for (columns, rows, scalar) in registry.matrices() {
        code.line(&format!(
            "export interface Mat{columns}x{rows}{} {{",
            scalar.suffix
        ));
        code.indented(|code| {
            for column in 0..columns {
                let fields = (0..rows)
                    .map(|row| format!("m{column}{row}: {}", scalar.ts_type))
                    .collect::<Vec<_>>()
                    .join("; ");
                code.line(&format!("{fields};"));
            }
        });
        code.line("}");
    }
}

fn emit_getter_function_body(target_class: &str, memoffset: u32, memsize: u32) -> String {
    format!(
        "return new {target_class}(this.buffer.subarray({memoffset}, {}));",
        memoffset + memsize
    )
}

fn emit_struct(code: &mut CodeBuilder, view: &View) -> Result<()> {
    let name = view.ts_type()?;
    let view = view
        .as_struct()
        .ok_or_else(|| color_eyre::eyre::eyre!("`{name}` is not a struct"))?;

    code.blank();
    code.line(&format!("export class {name} {{"));
    code.indented(|code| {
        code.line("buffer: Float32Array;");

        for member in &view.members {
            let getter_function = {
                let name = &member.name;
                let body = match &member.view.kind {
                    ViewKind::Vector { length, scalar } => {
                        let classname = format!("Vector{}{}", length, scalar.suffix);
                        emit_getter_function_body(
                            &classname,
                            member.offset,
                            (*length as u32) * scalar.byte_size,
                        )
                    }
                    ViewKind::Matrix {
                        columns,
                        rows,
                        scalar,
                    } => {
                        let classname = format!("Mat{columns}x{rows}{}", scalar.suffix);
                        emit_getter_function_body(
                            &classname,
                            member.offset,
                            (columns * rows) as u32 * scalar.byte_size,
                        )
                    }
                    other => unimplemented!("{other:?}"),
                };
                format!("get {name}(){{{}}}", body)
            };
            code.line(&getter_function);
        }
        let params = view
            .members
            .iter()
            .map(|member| {
                format!(
                    "{}: {}",
                    member.name,
                    member.view.ts_type().unwrap_or_default()
                )
            })
            .collect::<Vec<_>>();
        code.line(&format!("constructor({}) {{", params.join(", ")));
        code.indented(|code| {
            for member in &view.members {
                match member.view.leaf_accessors(&member.name) {
                    Ok(leaves) => code.line(&format!(
                        "this.{} = new {}([{}]);",
                        member.name,
                        member.view.storage().unwrap_or("Float32Array"),
                        leaves.join(", ")
                    )),
                    Err(err) => code.line(&format!("throw new Error(\"anygpu-gen-ts: {err}\");")),
                }
            }
        });
        code.line("}");
    });
    code.line("}");
    Ok(())
}
