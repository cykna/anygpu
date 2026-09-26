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
        code.emit_views(&registry);
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

    fn example_schema() -> ShaderBindings {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../examples/init.json");
        let json = std::fs::read_to_string(path).expect("examples/init.json should be readable");
        serde_json::from_str(&json).expect("examples/init.json should parse")
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
    fn views_carry_layout_for_buffer_slicing() {
        let schema = example_schema();
        let camera = struct_view(&schema, "Camera");
        assert_eq!(camera.ts_type().unwrap(), "Camera");
        assert_eq!(camera.byte_size().unwrap(), 80);
        assert_eq!(camera.storage().unwrap(), "Float32Array");

        let members = camera.as_struct().unwrap().members.clone();
        let ranges: Vec<_> = members
            .iter()
            .map(|member| {
                (
                    member.name.as_str(),
                    member.offset,
                    member.offset + member.view.byte_size().unwrap(),
                )
            })
            .collect();
        assert_eq!(
            ranges,
            vec![("position", 0, 16), ("rot", 16, 80)],
            "each member should resolve to its own byte range"
        );
        assert_eq!(members[0].view.ts_type().unwrap(), "Vector4f32");
        assert_eq!(members[1].view.ts_type().unwrap(), "Mat4x4f32");
    }

    #[test]
    fn views_read_out_leaves_in_memory_order() {
        let schema = example_schema();
        let camera = struct_view(&schema, "Camera");
        let members = &camera.as_struct().unwrap().members;
        assert_eq!(
            members[0].view.leaf_accessors("position").unwrap(),
            ["position.x", "position.y", "position.z", "position.w"]
        );
        assert_eq!(
            members[1].view.leaf_accessors("rot").unwrap()[..3],
            ["rot.m00", "rot.m01", "rot.m02"]
        );
    }

    #[test]
    fn nested_structs_and_arrays_recurse() {
        let inner = TypeInfo {
            name: "Inner".to_string(),
            descriptor: TypeDescriptor::Struct {
                size: 8,
                alignment: 4,
                members: vec![
                    MemberDescriptor {
                        name: "a".to_string(),
                        offset: 0,
                        ty: Box::new(TypeInfo {
                            name: "f32".to_string(),
                            descriptor: TypeDescriptor::Scalar { scalar: f32() },
                        }),
                    },
                    MemberDescriptor {
                        name: "b".to_string(),
                        offset: 4,
                        ty: Box::new(TypeInfo {
                            name: "i32".to_string(),
                            descriptor: TypeDescriptor::Scalar { scalar: i32() },
                        }),
                    },
                ],
            },
        };
        let outer = TypeInfo {
            name: "Outer".to_string(),
            descriptor: TypeDescriptor::Struct {
                size: 32,
                alignment: 16,
                members: vec![MemberDescriptor {
                    name: "items".to_string(),
                    offset: 0,
                    ty: Box::new(TypeInfo {
                        name: "array<Inner, 2>".to_string(),
                        descriptor: TypeDescriptor::Array {
                            base: Box::new(inner),
                            size: Some(2),
                            stride: 16,
                        },
                    }),
                }],
            },
        };

        let view = View::build(&outer).unwrap();
        let items = &view.as_struct().unwrap().members[0].view;
        assert_eq!(items.ts_type().unwrap(), "Array<Inner, 2>");
        assert_eq!(items.byte_size().unwrap(), 32);
        assert_eq!(
            items.leaf_accessors("items").unwrap(),
            ["items[0].a", "items[0].b", "items[1].a", "items[1].b"]
        );

        let err = items.storage().unwrap_err();
        assert!(err.to_string().contains("cannot share one buffer"), "{err}");
        assert_eq!(view.scalar_widths().into_iter().collect::<Vec<_>>(), [4]);
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
        let schema = example_schema();
        let typescript = TypeScriptBuilder::new().schema(&schema).build().unwrap();
        assert_eq!(
            typescript,
            concat!(
                "export interface Vector4f32 { x: number, y: number, z: number, w: number }\n",
                "export interface Mat4x4f32 {\n",
                "  m00: number; m01: number; m02: number; m03: number;\n",
                "  m10: number; m11: number; m12: number; m13: number;\n",
                "  m20: number; m21: number; m22: number; m23: number;\n",
                "  m30: number; m31: number; m32: number; m33: number;\n",
                "}\n",
                "\n",
                "export class Camera {\n",
                "  position: Float32Array;\n",
                "  rot: Float32Array;\n",
                "  constructor(position: Vector4f32, rot: Mat4x4f32) {\n",
                "    this.position = new Float32Array([position.x, position.y, position.z, position.w]);\n",
                "    this.rot = new Float32Array([rot.m00, rot.m01, rot.m02, rot.m03, rot.m10, rot.m11, rot.m12, rot.m13, rot.m20, rot.m21, rot.m22, rot.m23, rot.m30, rot.m31, rot.m32, rot.m33]);\n",
                "  }\n",
                "}\n",
            )
        );
    }
}
