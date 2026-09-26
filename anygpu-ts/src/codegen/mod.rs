mod emit;
mod registry;
mod scalar;
mod view;

use color_eyre::eyre::{Result, eyre};

use crate::codegen::emit::CodeBuilder;
use crate::codegen::registry::Registry;
use crate::codegen::view::View;
use anygpu::ShaderBindings;

pub fn generate_typescript(schema: &ShaderBindings) -> Result<String> {
    TypeScriptBuilder::new().schema(schema).build()
}

#[derive(Debug, Default)]
pub struct TypeScriptBuilder<'a> {
    schema: Option<&'a ShaderBindings>,
}

impl<'a> TypeScriptBuilder<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn schema(mut self, schema: &'a ShaderBindings) -> Self {
        self.schema = Some(schema);
        self
    }

    pub fn build(self) -> Result<String> {
        let schema = self.schema.ok_or_else(|| eyre!("no schema was provided"))?;
        let registry = collect(schema)?;

        let mut code = CodeBuilder::new();
        code.emit_views(&registry)?;
        for view in registry.structs() {
            code.emit_struct(view)?;
        }
        Ok(code.build())
    }
}

/// Walks every type in the schema and registers the accessor types it needs.
fn collect(schema: &ShaderBindings) -> Result<Registry> {
    let mut registry = Registry::default();
    for ty in &schema.types {
        View::build(ty)?.register(&mut registry)?;
    }
    Ok(registry)
}

#[cfg(test)]
mod tests {
    use anygpu::{
        MemberDescriptor, ScalarInfo, ShaderBindings, TypeDescriptor, TypeInfo, serde_json,
    };

    use super::*;

    fn f32() -> ScalarInfo {
        ScalarInfo {
            name: "float".to_string(),
            width: 4,
        }
    }

    fn i32() -> ScalarInfo {
        ScalarInfo {
            name: "sint".to_string(),
            width: 4,
        }
    }

    fn camera_schema() -> ShaderBindings {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../fixtures/camera.json");
        let json = std::fs::read_to_string(path).expect("fixtures/camera.json should be readable");
        serde_json::from_str(&json).expect("fixtures/camera.json should parse")
    }

    fn struct_view(schema: &ShaderBindings, name: &str) -> View {
        schema
            .types
            .iter()
            .map(|ty| View::build(ty).expect("every type should build a view"))
            .find(|view| view.name == name)
            .unwrap_or_else(|| panic!("schema should contain a `{name}` view"))
    }

    #[test]
    fn builder_requires_a_schema() {
        let err = TypeScriptBuilder::new().build().unwrap_err();
        assert!(err.to_string().contains("no schema"), "{err}");
    }

    #[test]
    fn byte_offsets_become_element_indices() {
        let camera = struct_view(&camera_schema(), "Camera");
        assert_eq!(camera.storage().unwrap(), "Float32Array");
        assert_eq!(camera.element_width().unwrap(), 4);
        assert_eq!(camera.byte_size().unwrap(), 80);
        assert_eq!(camera.element_count().unwrap(), 20);
        assert_eq!(camera.element_offset(0).unwrap(), 0);
        assert_eq!(camera.element_offset(16).unwrap(), 4);

        let members = &camera.as_struct().unwrap().members;
        assert_eq!(members[0].view.byte_size().unwrap(), 16);
        assert_eq!(members[0].view.element_count().unwrap(), 4);
        assert_eq!(members[1].view.byte_size().unwrap(), 64);
        assert_eq!(members[1].view.element_count().unwrap(), 16);
    }

    #[test]
    fn element_width_follows_the_scalar_kind() {
        let ty = TypeInfo {
            name: "vec2<f64>".to_string(),
            descriptor: TypeDescriptor::Vector {
                length: 2,
                scalar: ScalarInfo {
                    name: "float".to_string(),
                    width: 8,
                },
            },
        };
        let view = View::build(&ty).unwrap();
        assert_eq!(view.storage().unwrap(), "Float64Array");
        assert_eq!(view.element_width().unwrap(), 8);
        assert_eq!(view.element_offset(8).unwrap(), 1);
    }

    #[test]
    fn offsets_that_are_not_whole_elements_are_rejected() {
        let camera = struct_view(&camera_schema(), "Camera");
        let err = camera.element_offset(2).unwrap_err().to_string();
        assert!(
            err.contains("not a multiple of the 4-byte element"),
            "{err}"
        );
    }

    #[test]
    fn mixed_scalar_storage_is_rejected() {
        let view = View::build(&mixed_scalar_struct()).unwrap();
        let err = view.storage().unwrap_err().to_string();
        assert!(err.contains("cannot share one buffer"), "{err}");
    }

    #[test]
    fn array_members_report_an_error_instead_of_panicking() {
        let view = View::build(&array_member_struct()).unwrap();
        let typescript = TypeScriptBuilder::new()
            .schema(&ShaderBindings {
                types: vec![array_member_struct()],
            })
            .build()
            .unwrap_err()
            .to_string();
        assert!(
            typescript.contains("cannot be exposed as a view yet"),
            "{typescript}"
        );
        let _ = view;
    }

    #[test]
    fn unsupported_descriptors_error_instead_of_panicking() {
        let ty = TypeInfo {
            name: "texture".to_string(),
            descriptor: TypeDescriptor::Image,
        };
        let err = View::build(&ty).unwrap_err().to_string();
        assert!(err.contains("no view representation"), "{err}");
    }

    #[test]
    fn names_are_sanitized_into_identifiers() {
        let ty = TypeInfo {
            name: "anon_struct<f32, i32>".to_string(),
            descriptor: TypeDescriptor::Struct {
                size: 8,
                alignment: 4,
                members: vec![],
            },
        };
        let view = View::build(&ty).unwrap();
        assert_eq!(view.ts_type().unwrap(), "anon_struct_f32__i32_");
    }

    #[test]
    fn output_is_stable_and_scalar_keyed() {
        let schema = camera_schema();
        let typescript = TypeScriptBuilder::new().schema(&schema).build().unwrap();

        let expected = concat!(
            "export class Vector4f32 {\n",
            "  buffer: Float32Array;\n",
            "  constructor(buffer: Float32Array) { this.buffer = buffer; }\n",
            "  get x(): number { return this.buffer[0]; }\n",
            "  set x(v: number) { this.buffer[0] = v; }\n",
            "  get y(): number { return this.buffer[1]; }\n",
            "  set y(v: number) { this.buffer[1] = v; }\n",
            "  get z(): number { return this.buffer[2]; }\n",
            "  set z(v: number) { this.buffer[2] = v; }\n",
            "  get w(): number { return this.buffer[3]; }\n",
            "  set w(v: number) { this.buffer[3] = v; }\n",
            "}\n",
            "\n",
            "export class Mat4x4f32 {\n",
            "  buffer: Float32Array;\n",
            "  constructor(buffer: Float32Array) { this.buffer = buffer; }\n",
            "  get m00(): number { return this.buffer[0]; }\n",
            "  set m00(v: number) { this.buffer[0] = v; }\n",
            "  get m01(): number { return this.buffer[1]; }\n",
            "  set m01(v: number) { this.buffer[1] = v; }\n",
            "  get m02(): number { return this.buffer[2]; }\n",
            "  set m02(v: number) { this.buffer[2] = v; }\n",
            "  get m03(): number { return this.buffer[3]; }\n",
            "  set m03(v: number) { this.buffer[3] = v; }\n",
            "  get m10(): number { return this.buffer[4]; }\n",
            "  set m10(v: number) { this.buffer[4] = v; }\n",
            "  get m11(): number { return this.buffer[5]; }\n",
            "  set m11(v: number) { this.buffer[5] = v; }\n",
            "  get m12(): number { return this.buffer[6]; }\n",
            "  set m12(v: number) { this.buffer[6] = v; }\n",
            "  get m13(): number { return this.buffer[7]; }\n",
            "  set m13(v: number) { this.buffer[7] = v; }\n",
            "  get m20(): number { return this.buffer[8]; }\n",
            "  set m20(v: number) { this.buffer[8] = v; }\n",
            "  get m21(): number { return this.buffer[9]; }\n",
            "  set m21(v: number) { this.buffer[9] = v; }\n",
            "  get m22(): number { return this.buffer[10]; }\n",
            "  set m22(v: number) { this.buffer[10] = v; }\n",
            "  get m23(): number { return this.buffer[11]; }\n",
            "  set m23(v: number) { this.buffer[11] = v; }\n",
            "  get m30(): number { return this.buffer[12]; }\n",
            "  set m30(v: number) { this.buffer[12] = v; }\n",
            "  get m31(): number { return this.buffer[13]; }\n",
            "  set m31(v: number) { this.buffer[13] = v; }\n",
            "  get m32(): number { return this.buffer[14]; }\n",
            "  set m32(v: number) { this.buffer[14] = v; }\n",
            "  get m33(): number { return this.buffer[15]; }\n",
            "  set m33(v: number) { this.buffer[15] = v; }\n",
            "}\n",
            "\n",
            "export class Camera {\n",
            "  buffer: Float32Array;\n",
            "  get position(): Vector4f32 { return new Vector4f32(this.buffer.subarray(0, 4)); }\n",
            "  get rot(): Mat4x4f32 { return new Mat4x4f32(this.buffer.subarray(4, 20)); }\n",
            "  constructor(position: Vector4f32, rot: Mat4x4f32) {\n",
            "    this.buffer = new Float32Array(20); // 80 bytes / 4\n",
            "    this.buffer.set(position.buffer, 0); // offset 0 bytes / 4 = 0\n",
            "    this.buffer.set(rot.buffer, 4); // offset 16 bytes / 4 = 4\n",
            "  }\n",
            "}\n",
        );
        assert_eq!(typescript, expected);
    }

    fn mixed_scalar_struct() -> TypeInfo {
        scalar_struct("Mixed", vec![("a", 0, f32()), ("b", 4, i32())])
    }

    fn array_member_struct() -> TypeInfo {
        TypeInfo {
            name: "WithArray".to_string(),
            descriptor: TypeDescriptor::Struct {
                size: 16,
                alignment: 16,
                members: vec![MemberDescriptor {
                    name: "items".to_string(),
                    offset: 0,
                    ty: Box::new(TypeInfo {
                        name: "array<vec4<f32>, 1>".to_string(),
                        descriptor: TypeDescriptor::Array {
                            base: Box::new(TypeInfo {
                                name: "vec4<f32>".to_string(),
                                descriptor: TypeDescriptor::Vector {
                                    length: 4,
                                    scalar: f32(),
                                },
                            }),
                            size: Some(1),
                            stride: 16,
                        },
                    }),
                }],
            },
        }
    }

    fn scalar_struct(name: &str, members: Vec<(&str, u32, ScalarInfo)>) -> TypeInfo {
        TypeInfo {
            name: name.to_string(),
            descriptor: TypeDescriptor::Struct {
                size: 16,
                alignment: 16,
                members: members
                    .into_iter()
                    .map(|(name, offset, scalar)| MemberDescriptor {
                        name: name.to_string(),
                        offset,
                        ty: Box::new(TypeInfo {
                            name: scalar.name.clone(),
                            descriptor: TypeDescriptor::Scalar { scalar },
                        }),
                    })
                    .collect(),
            },
        }
    }
}
