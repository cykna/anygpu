use color_eyre::eyre::{Result, eyre};

use crate::codegen::descriptors::{Access, Accessor, ClassDescriptor, Getter, Method};
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

/// Renders a [`ClassDescriptor`] as a TypeScript class.
///
/// This is the only place that writes a class, so the shape of the generated
/// code cannot drift between kinds. Members are emitted in one canonical
/// order: the buffer field, extra fields, the constructor, the view factory,
/// accessors, methods and finally getters.
fn emit_class(code: &mut CodeBuilder, class: &ClassDescriptor) -> Result<()> {
    let name = &class.name;

    code.blank();
    code.line(&format!("export class {name} {{"));
    code.indented(|code| {
        code.line(&format!("public buffer: {};", class.buffer));
        for field in &class.fields {
            code.line(field);
        }

        code.line(&format!("constructor({}) {{", class.params.join(", ")));
        code.indented(|code| {
            for statement in &class.init {
                code.line(statement);
            }
            Ok(())
        })?;
        code.line("}");

        // Every class is a view over a buffer, so they all share this factory.
        // A struct has to allocate from its members instead, which means
        // `new X(subarray)` is not available for it, so `.view(..)` is the one
        // way to adopt someone else's buffer.
        code.line(&format!(
            "static view(buffer: {array}): {name} {{ const self = Object.create({name}.prototype) as {name}; self.buffer = buffer; return self; }}",
            array = class.buffer
        ));

        for accessor in &class.accessors {
            code.line(&format!(
                "get {name}(): {ty} {{ return this.buffer[{index}]; }}",
                name = accessor.name,
                ty = accessor.ty,
                index = accessor.index
            ));
            code.line(&format!(
                "set {name}(v: {ty}) {{ this.buffer[{index}] = v; }}",
                name = accessor.name,
                ty = accessor.ty,
                index = accessor.index
            ));
        }

        for method in &class.methods {
            let returns = method.ty.as_ref().map_or(String::new(), |ty| format!(": {ty}"));
            code.line(&format!("{}({}){returns} {{", method.name, method.params));
            code.indented(|code| {
                for statement in &method.body {
                    code.line(statement);
                }
                Ok(())
            })?;
            code.line("}");
        }

        for getter in &class.getters {
            code.line(&format!(
                "get {name}(): {ty} {{ return {expr}; }}",
                name = getter.name,
                ty = getter.ty,
                expr = getter.expr
            ));
        }
        Ok(())
    })?;
    code.line("}");
    Ok(())
}

/// `Vector3f32`: one accessor per component.
fn vector_class(length: u8, scalar: ScalarLayout) -> Result<ClassDescriptor> {
    let accessors = (0..usize::from(length))
        .map(|index| {
            Ok(Accessor {
                name: component(length, index)?.to_string(),
                ty: scalar.ts_type.to_string(),
                index: index as u32,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut class =
        ClassDescriptor::adopting(format!("Vector{length}{}", scalar.suffix), scalar.array);
    class.accessors = accessors;
    Ok(class)
}

/// `Mat4x4f32`: one accessor per component, indexed column-major to match WGSL.
fn matrix_class(columns: u8, rows: u8, scalar: ScalarLayout) -> ClassDescriptor {
    let accessors = (0..columns)
        .flat_map(|column| {
            (0..rows).map(move |row| Accessor {
                name: format!("m{column}{row}"),
                ty: scalar.ts_type.to_string(),
                index: u32::from(column) * u32::from(rows) + u32::from(row),
            })
        })
        .collect();

    let mut class = ClassDescriptor::adopting(
        format!("Mat{columns}x{rows}{}", scalar.suffix),
        scalar.array,
    );
    class.accessors = accessors;
    class
}

/// The accessor class for an array of `base`.
///
/// The count and the stride are deliberately *not* baked into the class: the
/// stride is a constructor argument and the count is derived from the buffer, so
/// one class serves every `array<Base, N>` in the schema.
fn array_class(name: &str, base: &View) -> Result<ClassDescriptor> {
    let base_name = base.ts_type()?;
    let mut class = ClassDescriptor::adopting(name.to_string(), base.storage()?);
    class.params.push("stride: number".to_string());
    class
        .fields
        .push("stride: number; // em elementos".to_string());
    class.init.push("this.stride = stride;".to_string());
    class.methods.push(Method {
        name: "get".to_string(),
        ty: Some(base_name.clone()),
        params: "index: number".to_string(),
        body: vec![
            "const start = index * this.stride;".to_string(),
            format!("return {base_name}.view(this.buffer.subarray(start, start + this.stride));"),
        ],
    });
    class.getters.push(Getter {
        name: "length".to_string(),
        ty: "number".to_string(),
        expr: "this.buffer.length / this.stride".to_string(),
    });
    Ok(class)
}

/// A struct: one getter and one copy per member, and a buffer it allocates
/// itself.
fn struct_class(view: &View) -> Result<ClassDescriptor> {
    let name = view.ts_type()?;
    let members = &view
        .as_struct()
        .ok_or_else(|| eyre!("`{name}` is not a struct"))?
        .members;

    let array = view.storage()?;
    let width = view.element_width()?;
    let total_bytes = view.byte_size()?;

    let mut class = ClassDescriptor {
        name: name.clone(),
        buffer: array.to_string(),
        init: vec![format!(
            "this.buffer = new {array}({}); // {total_bytes} bytes / {width}",
            allocation(view, members, width, total_bytes)?
        )],
        ..Default::default()
    };
    for member in members {
        let ty = member.view.ts_type()?;
        let access = member_access(view, member)?;
        let expr = access.getter(&ty);
        class.params.push(format!("{}: {ty}", member.name));
        class.getters.push(Getter {
            name: member.name.clone(),
            ty,
            expr,
        });
        class
            .init
            .push(access.copy(&member.name, member.offset, width));
    }
    Ok(class)
}

/// The element count a struct constructor allocates.
///
/// A runtime-sized array has no length in the schema: it runs to the end of the
/// buffer, so the allocation has to grow by whatever the caller passed in.
/// `Math.max` keeps the struct's declared span, which naga measures including one
/// padding element for the runtime array, from showing up as a phantom element
/// ahead of an array that starts at 0.
fn allocation(view: &View, members: &[Member], width: u32, total_bytes: u32) -> Result<String> {
    let elements = (total_bytes / width).to_string();
    let Some(dynamic) = members
        .iter()
        .find(|member| matches!(member.view.kind, ViewKind::Array { size: None, .. }))
    else {
        return Ok(elements);
    };

    let start = view.element_offset(dynamic.offset)?;
    let supplied = if start == 0 {
        format!("{}.buffer.length", dynamic.name)
    } else {
        format!("{start} + {}.buffer.length", dynamic.name)
    };
    Ok(format!("Math.max({elements}, {supplied})"))
}

fn emit_views(code: &mut CodeBuilder, registry: &Registry) -> Result<()> {
    for (length, scalar) in registry.vectors() {
        emit_class(code, &vector_class(length, scalar)?)?;
    }
    for (columns, rows, scalar) in registry.matrices() {
        emit_class(code, &matrix_class(columns, rows, scalar))?;
    }
    for (name, base) in registry.arrays() {
        emit_class(code, &array_class(name, base)?)?;
    }
    Ok(())
}

fn emit_struct(code: &mut CodeBuilder, view: &View) -> Result<()> {
    emit_class(code, &struct_class(view)?)
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
