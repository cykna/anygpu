use std::collections::BTreeSet;
use std::fmt::Write;

use anygpu::{MemberDescriptor, ShaderBindings, TypeDescriptor, TypeInfo};
use color_eyre::eyre::{Result, eyre};

const VECTOR_COMPONENTS: [&str; 4] = ["x", "y", "z", "w"];

#[derive(Default)]
struct Generator {
    vectors: BTreeSet<u8>,
    matrices: Vec<(u8, u8)>,
}

pub fn generate_typescript(schema: &ShaderBindings) -> Result<String> {
    let mut generator = Generator::default();
    for ty in &schema.types {
        generator.collect(ty)?;
    }

    let mut out = String::new();
    generator.write_vector_interfaces(&mut out)?;
    generator.write_matrix_interfaces(&mut out)?;

    for ty in &schema.types {
        if let TypeDescriptor::Struct { members, .. } = &ty.descriptor {
            generator.write_struct_class(&mut out, &ty.name, members)?;
        }
    }
    Ok(out)
}

impl Generator {
    fn collect(&mut self, ty: &TypeInfo) -> Result<()> {
        match &ty.descriptor {
            TypeDescriptor::Scalar { .. } => Ok(()),
            TypeDescriptor::Vector { length, .. } => {
                for index in 0..usize::from(*length) {
                    vector_component(*length, index)?;
                }
                self.vectors.insert(*length);
                Ok(())
            }
            TypeDescriptor::Matrix { columns, rows, .. } => {
                let shape = (*columns, *rows);
                if !self.matrices.contains(&shape) {
                    self.matrices.push(shape);
                }
                Ok(())
            }
            TypeDescriptor::Struct { members, .. } => {
                for member in members {
                    self.collect(&member.ty)?;
                }
                Ok(())
            }
            other => unimplemented!("Not supported yet {:?}", other),
        }
    }

    fn write_vector_interfaces(&self, out: &mut String) -> Result<()> {
        for length in &self.vectors {
            let mut fields = Vec::with_capacity(usize::from(*length));
            for index in 0..usize::from(*length) {
                fields.push(format!("{}: number", vector_component(*length, index)?));
            }
            writeln!(
                out,
                "export interface Vector{length} {{ {} }}",
                fields.join(", ")
            )?;
        }
        Ok(())
    }

    fn write_matrix_interfaces(&self, out: &mut String) -> Result<()> {
        for (columns, rows) in &self.matrices {
            writeln!(out, "export interface Mat{columns}x{rows}<T> {{")?;
            for column in 0..*columns {
                let fields: Vec<String> =
                    (0..*rows).map(|row| format!("m{column}{row}: T")).collect();
                writeln!(out, "  {};", fields.join("; "))?;
            }
            writeln!(out, "}}")?;
        }
        Ok(())
    }

    fn write_struct_class(
        &self,
        out: &mut String,
        name: &str,
        members: &[MemberDescriptor],
    ) -> Result<()> {
        if !out.is_empty() {
            out.push('\n');
        }
        writeln!(out, "export class {name} {{")?;
        for member in members {
            writeln!(out, "  {}: Float32Array;", member.name)?;
        }

        let mut params = Vec::with_capacity(members.len());
        for member in members {
            params.push(format!("{}: {}", member.name, input_type(&member.ty)?));
        }
        writeln!(out, "  constructor({}) {{", params.join(", "))?;

        for member in members {
            match flattened_fields(&member.ty, &member.name) {
                Ok(fields) => writeln!(
                    out,
                    "    this.{name} = new Float32Array([{fields}]);",
                    name = member.name,
                    fields = fields.join(", ")
                )?,
                Err(err) => writeln!(out, "    throw new Error(\"anygpu-gen-ts: {err}\");")?,
            }
        }
        writeln!(out, "  }}")?;
        writeln!(out, "}}")?;
        Ok(())
    }
}

fn vector_component(length: u8, index: usize) -> Result<&'static str> {
    VECTOR_COMPONENTS
        .get(index)
        .copied()
        .ok_or_else(|| eyre!("vector length {length} is not supported (expected 2..=4)"))
}

fn input_type(ty: &TypeInfo) -> Result<String> {
    Ok(match &ty.descriptor {
        TypeDescriptor::Scalar { .. } => "number".to_string(),
        TypeDescriptor::Vector { length, .. } => format!("Vector{length}"),
        TypeDescriptor::Matrix { columns, rows, .. } => format!("Mat{columns}x{rows}<number>"),
        TypeDescriptor::Struct { .. } => ty.name.clone(),
        other => unimplemented!("Not implemented yet {other:?}"),
    })
}

fn flattened_fields(ty: &TypeInfo, param: &str) -> Result<Vec<String>> {
    match &ty.descriptor {
        TypeDescriptor::Scalar { .. } => Ok(vec![param.to_string()]),
        TypeDescriptor::Vector { length, .. } => (0..usize::from(*length))
            .map(|index| vector_component(*length, index).map(|c| format!("{param}.{c}")))
            .collect(),
        TypeDescriptor::Matrix { columns, rows, .. } => {
            let mut fields = Vec::with_capacity(usize::from(*columns) * usize::from(*rows));
            for column in 0..*columns {
                for row in 0..*rows {
                    fields.push(format!("{param}.m{column}{row}"));
                }
            }
            Ok(fields)
        }
        TypeDescriptor::Struct { .. } => Err(eyre!(
            "member `{param}` has nested struct type `{}`, which is not supported yet",
            ty.name
        )),
        other => unimplemented!("Not implemented yet {other:?}",),
    }
}
