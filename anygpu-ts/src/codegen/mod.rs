pub mod descriptors;
mod emit;
mod pipeline;
mod registry;
mod scalar;
mod view;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use color_eyre::eyre::{Result, eyre};

use crate::codegen::descriptors::ClassDescriptor;
use crate::codegen::emit::{
    CodeBuilder, builtin_classes, emit_class, emit_class_all, shader_classes,
};
use crate::codegen::registry::Registry;
use crate::codegen::view::{Views, builtin_module};
use anygpu::ShaderBindings;

/// The file the support section takes inside the output folder.
const SUPPORT_PATH: &str = "support.ts";

/// The folder the shared accessor classes are written to.
const BUILTINS_DIR: &str = "builtins";

/// Generates the TypeScript for one shader.
///
/// `name` is the shader's own name, taken from the file it came from: it is what
/// the pipeline helper is exported under, since the schema itself has nowhere to
/// record a name.
///
/// The result is self-contained: it carries the support section and every shared
/// class inline, so a file written from it compiles on its own. Generating a
/// whole folder instead spreads the same declarations over `support.ts` and
/// `builtins/`, which is what [`generate_batch`] does.
pub fn generate_typescript(schema: &ShaderBindings, name: &str) -> Result<String> {
    TypeScriptBuilder::new().schema(schema).shader(name).build()
}

/// One file a batch produces, named by the path it takes inside the output
/// folder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchFile {
    /// Relative to the output folder, e.g. `support.ts`, `builtins/mat4x4f32.ts`
    /// or `camera.ts`.
    pub path: PathBuf,
    pub contents: String,
}

/// Generates every file one run over a folder of schemas produces.
///
/// A batch is one execution of the command over a whole folder, and the output
/// is split three ways:
///
/// - `support.ts`, once per batch. It does not depend on any schema, so it is
///   written from the one checked-in copy of the support section.
/// - `builtins/<class>.ts`, one per shared accessor class, written once per
///   batch however many shaders use it. A class is shareable when nothing about
///   it is shader-specific: every `vec3<f32>` in the folder wants the same
///   `Vector3f32`.
/// - `<shader>.ts`, one per shader exactly as before, carrying only what is
///   specific to it: its own structs, the arrays over those structs, and its
///   pipeline instance. Everything shared is imported instead.
///
/// Every run rebuilds the whole set from scratch, so pointing a second run at the
/// same folder overwrites what the first wrote and changes nothing else.
pub fn generate_batch<'a>(
    shaders: impl IntoIterator<Item = (&'a str, &'a ShaderBindings)>,
) -> Result<Vec<BatchFile>> {
    let shaders: Vec<(&str, &ShaderBindings)> = shaders.into_iter().collect();

    // The shared classes are collected across the whole batch before anything is
    // written, so each is written once and a shader can import exactly what the
    // batch needs rather than what it happened to be processed before.
    let mut builtins: BTreeMap<String, ClassDescriptor> = BTreeMap::new();
    let mut collected = Vec::with_capacity(shaders.len());
    for (name, schema) in &shaders {
        let registry = collect(schema)?;
        for class in builtin_classes(&registry)? {
            // Keyed by name, which is enough: a vector, a matrix and an array of
            // primitives are each fully determined by the class name they are
            // given, so the same name is always the same class.
            builtins.entry(class.name.clone()).or_insert(class);
        }
        collected.push((*name, schema, registry));
    }
    let shared: BTreeSet<&str> = builtins.keys().map(String::as_str).collect();

    // A folder with no schema in it still gets the support section: it does not
    // depend on any schema, and a caller asking for a batch of nothing is asking
    // for the part that holds for any batch.
    let mut files = vec![BatchFile {
        path: PathBuf::from(SUPPORT_PATH),
        contents: pipeline::SUPPORT.to_string(),
    }];
    for class in builtins.values() {
        files.push(BatchFile {
            path: PathBuf::from(BUILTINS_DIR).join(format!("{}.ts", builtin_module(&class.name))),
            contents: builtin_file(class, &shared)?,
        });
    }
    for (name, schema, registry) in collected {
        files.push(BatchFile {
            path: PathBuf::from(format!("{name}.ts")),
            contents: shader_file(name, schema, &registry, &shared)?,
        });
    }
    Ok(files)
}

/// A shared class's own file, with the siblings of `builtins/` it names.
///
/// Every file the batch writes sits at most one folder deep and imports only
/// from its own folder, so a specifier is always `./<module>`.
fn builtin_file(class: &ClassDescriptor, shared: &BTreeSet<&str>) -> Result<String> {
    let mut code = CodeBuilder::new();
    for name in shared_dependencies(std::slice::from_ref(class), shared) {
        code.line(&format!(
            "import {{ {name} }} from \"./{}\";",
            builtin_module(name)
        ));
    }
    emit_class(&mut code, class)?;
    Ok(code.build())
}

/// One shader's file: its own structs, the arrays over them, and the pipeline
/// instance its schema described — importing the support symbols and the shared
/// classes that content names.
fn shader_file(
    name: &str,
    schema: &ShaderBindings,
    registry: &Registry,
    shared: &BTreeSet<&str>,
) -> Result<String> {
    let classes = shader_classes(registry)?;
    let mut code = CodeBuilder::new();

    // Collected into a set of whole import lines, so the same folder always
    // generates the same files in the same order.
    let mut modules: BTreeSet<String> = BTreeSet::new();
    for name in shared_dependencies(&classes, shared) {
        modules.insert(format!(
            "import {{ {name} }} from \"./builtins/{}\";",
            builtin_module(name)
        ));
    }
    if let Some(pipeline) = &schema.pipelines {
        let names: Vec<&str> = pipeline::support_symbols(pipeline).into_iter().collect();
        modules.insert(format!(
            "import {{ {} }} from \"./support\";",
            names.join(", ")
        ));
    }
    // The blank line after the imports is left to whatever comes next: both the
    // pipeline instance and every class open with one, and `CodeBuilder` drops it
    // on a file with nothing in it yet.
    for line in &modules {
        code.line(line);
    }

    if let Some(pipeline) = &schema.pipelines {
        pipeline::emit_shader(&mut code, name, pipeline)?;
    }
    emit_class_all(&mut code, &classes)?;
    Ok(code.build())
}

/// The shared classes a group of classes names from another file.
///
/// A name outside `shared` is declared in the same file — a struct the shader
/// declares for itself, or a scalar — and needs no import. A class never names
/// itself.
fn shared_dependencies<'a>(
    classes: &'a [ClassDescriptor],
    shared: &BTreeSet<&str>,
) -> Vec<&'a str> {
    let mut names: BTreeSet<&'a str> = BTreeSet::new();
    for class in classes {
        for dep in &class.deps {
            if dep != &class.name && shared.contains(dep.as_str()) {
                names.insert(dep.as_str());
            }
        }
    }
    names.into_iter().collect()
}

#[derive(Debug, Default)]
pub struct TypeScriptBuilder<'a> {
    schema: Option<&'a ShaderBindings>,
    shader: Option<&'a str>,
}

impl<'a> TypeScriptBuilder<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn schema(mut self, schema: &'a ShaderBindings) -> Self {
        self.schema = Some(schema);
        self
    }

    /// The shader's name, needed only to name the pipeline helper.
    pub fn shader(mut self, name: &'a str) -> Self {
        self.shader = Some(name);
        self
    }

    pub fn build(self) -> Result<String> {
        let schema = self.schema.ok_or_else(|| eyre!("no schema was provided"))?;
        let registry = collect(schema)?;

        let mut code = CodeBuilder::new();
        // The pipeline comes first, as one block: the WebGPU types a descriptor is
        // built from, the class that builds it, and the one instance that belongs
        // to this shader. It is emitted for every file, including one whose schema
        // has no pipeline at all — then only the class is there to be shared.
        code.emit_pipeline_support()?;
        // A schema without a pipeline is not an error, it just has no pipeline to
        // describe; a named helper is the whole point of a shader that has one.
        if let Some(pipelines) = &schema.pipelines {
            let name = self
                .shader
                .ok_or_else(|| eyre!("no shader name was provided for its pipeline helper"))?;
            code.emit_shader_pipeline(name, pipelines)?;
        }
        code.emit_accessor_classes(&registry)?;
        for view in registry.structs() {
            code.emit_struct(view)?;
        }
        Ok(code.build())
    }
}

/// Walks every type in the schema and registers the accessor types it needs.
///
/// Each type is resolved once, by id, and registered from there: a type used by
/// several others is visited a single time no matter how often it is named.
fn collect(schema: &ShaderBindings) -> Result<Registry> {
    let mut registry = Registry::default();
    let mut views = Views::new(&schema.types);
    for ty in &schema.types {
        views.get(ty.id.as_str())?.register(&mut registry)?;
    }
    Ok(registry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::descriptors::Access;
    use crate::codegen::emit::member_access;
    use crate::codegen::view::{View, ViewKind};
    use anygpu::{
        ImageClass, ImageDimension, MemberDescriptor, PipelineDescriptor, PipelineLayout,
        ScalarInfo, ShaderBindings, ShaderMetadata, TypeDescriptor, TypeId, TypeInfo, TypeInterner,
        naga,
    };
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

    /// A camera uniform: `vec4` at offset 0, `mat4x4` at 16, 80 bytes in total.
    ///
    /// Kept inline and run through the real pipeline (naga, then the binding
    /// schema) so the test exercises the same path a `.wgsl` input takes, instead
    /// of depending on a checked-in schema that would have to be regenerated by
    /// hand whenever the layout changes.
    const CAMERA_WGSL: &str = r"
struct Camera {
    position: vec4<f32>,
    rot: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: Camera;
";

    fn camera_schema() -> ShaderBindings {
        let module = naga::front::wgsl::parse_str(CAMERA_WGSL).expect("the camera should parse");
        ShaderMetadata::new(&module)
            .expect("the camera should expose bindings")
            .generate_bindings()
    }

    /// Wraps a type list in the rest of a `ShaderBindings`, with no pipeline:
    /// the types are worth generating on their own.
    fn schema(types: Vec<TypeInfo>) -> ShaderBindings {
        ShaderBindings {
            types,
            pipelines: None,
            bindgroups: vec![],
        }
    }

    /// The same, for a schema that does describe a pipeline.
    fn schema_with_pipeline(types: Vec<TypeInfo>, pipelines: PipelineDescriptor) -> ShaderBindings {
        ShaderBindings {
            types,
            pipelines: Some(pipelines),
            bindgroups: vec![],
        }
    }

    /// A pipeline with nothing in it but its layout, for tests that only care
    /// about the types.
    fn empty_pipeline() -> PipelineDescriptor {
        PipelineDescriptor {
            layout: PipelineLayout {
                bind_group_layouts: vec![],
                push_constant_ranges: vec![],
            },
            vertex: None,
            fragment: None,
            compute: None,
        }
    }

    /// Interns a fixture the way the real pipeline does, so a test that names a
    /// type twice gets one id back, exactly as a schema read from disk would.
    fn types_of(build: impl FnOnce(&mut TypeInterner)) -> Vec<TypeInfo> {
        let mut interner = TypeInterner::new();
        build(&mut interner);
        interner.finish()
    }

    fn scalar_ty(interner: &mut TypeInterner, name: &str, scalar: ScalarInfo) -> TypeId {
        interner.intern(name.to_string(), |_| TypeDescriptor::Scalar {
            scalar: scalar.clone(),
        })
    }

    fn vector_ty(
        interner: &mut TypeInterner,
        name: &str,
        length: u8,
        scalar: ScalarInfo,
    ) -> TypeId {
        interner.intern(name.to_string(), |_| TypeDescriptor::Vector {
            length,
            scalar: scalar.clone(),
        })
    }

    fn array_ty(
        interner: &mut TypeInterner,
        name: &str,
        base: &TypeId,
        size: Option<u32>,
        stride: u32,
    ) -> TypeId {
        let base = base.clone();
        interner.intern(name.to_string(), |_| TypeDescriptor::Array {
            base: base.clone(),
            size,
            stride,
        })
    }

    fn struct_ty(
        interner: &mut TypeInterner,
        name: &str,
        size: u32,
        alignment: u32,
        members: &[(&str, u32, TypeId)],
    ) -> TypeId {
        let members = members
            .iter()
            .map(|(name, offset, ty)| MemberDescriptor {
                name: name.to_string(),
                offset: *offset,
                ty: ty.clone(),
            })
            .collect();
        interner.intern(name.to_string(), |_| TypeDescriptor::Struct {
            size,
            alignment,
            members,
        })
    }

    /// `struct Light { color: vec3<f32>, intensity: f32 }`
    fn light_ty(interner: &mut TypeInterner) -> TypeId {
        let color = vector_ty(interner, "vec3<f32>", 3, f32());
        let intensity = scalar_ty(interner, "f32", f32());
        struct_ty(
            interner,
            "Light",
            16,
            16,
            &[("color", 0, color), ("intensity", 12, intensity)],
        )
    }

    /// `struct Lights { items: array<Light, 4> }`
    fn array_member_types() -> Vec<TypeInfo> {
        types_of(|interner| {
            let light = light_ty(interner);
            let items = array_ty(interner, "array<Light, 4>", &light, Some(4), 16);
            struct_ty(interner, "Lights", 64, 16, &[("items", 0, items)]);
        })
    }

    /// `struct Particles { data: array<vec4<f32>> }`, sized at runtime.
    fn runtime_array_types() -> Vec<TypeInfo> {
        types_of(|interner| {
            let vec4 = vector_ty(interner, "vec4<f32>", 4, f32());
            let data = array_ty(interner, "array<vec4<f32>>", &vec4, None, 16);
            struct_ty(interner, "Particles", 16, 16, &[("data", 0, data)]);
        })
    }

    /// The view of the type the schema calls `name`.
    fn struct_view(types: &[TypeInfo], name: &str) -> View {
        let mut views = Views::new(types);
        types
            .iter()
            .find(|ty| ty.name == name)
            .map(|ty| {
                views
                    .get(ty.id.as_str())
                    .expect("every type should build a view")
            })
            .unwrap_or_else(|| panic!("schema should contain a `{name}` view"))
    }

    #[test]
    fn builder_requires_a_schema() {
        let err = TypeScriptBuilder::new().build().unwrap_err();
        assert!(err.to_string().contains("no schema"), "{err}");
    }

    #[test]
    fn a_type_used_twice_is_defined_once() {
        // `vec3<f32>` is a member of both `Light` and `Scene`, and `Light` is
        // reached again through `Scene`. Each is defined once and referred to by
        // id, so the schema never nests a second copy of a type.
        let types = types_of(|interner| {
            let light = light_ty(interner);
            // Already interned as a member of `Light`, so this is the same type.
            let vec3 = vector_ty(interner, "vec3<f32>", 3, f32());
            struct_ty(
                interner,
                "Scene",
                32,
                16,
                &[("ambient", 0, vec3), ("light", 16, light.clone())],
            );
            // Naming either of them again must not define anything new.
            assert_eq!(light_ty(interner), light);
        });

        let names: Vec<&str> = types.iter().map(|ty| ty.name.as_str()).collect();
        assert_eq!(names, ["vec3<f32>", "f32", "Light", "Scene"]);

        // Ids are unique, and every reference points at one that exists.
        let ids: Vec<&str> = types.iter().map(|ty| ty.id.as_str()).collect();
        assert_eq!(ids, ["t0", "t1", "t2", "t3"]);
        for ty in &types {
            let TypeDescriptor::Struct { members, .. } = &ty.descriptor else {
                continue;
            };
            for member in members {
                assert!(
                    ids.contains(&member.ty.as_str()),
                    "`{}` refers to `{}`, which is not in the schema",
                    ty.name,
                    member.ty
                );
            }
        }
    }

    #[test]
    fn a_view_is_built_once_per_type_and_shared() {
        let types = array_member_types();
        let mut views = Views::new(&types);

        let lights = types
            .iter()
            .find(|ty| ty.name == "Lights")
            .expect("`Lights` should be in the schema");
        let first = views.get(lights.id.as_str()).unwrap();
        let second = views.get(lights.id.as_str()).unwrap();

        // The cache hands back the same view, and the nested `array<Light, 4>`
        // and `Light` inside it are the shared ones rather than fresh copies.
        assert_eq!(first.name, "Lights");
        assert_eq!(
            format!("{first:?}"),
            format!("{second:?}"),
            "a cached view should be identical to the one it cached"
        );
        let items = &first.as_struct().unwrap().members[0].view;
        let again = views
            .get(
                types
                    .iter()
                    .find(|ty| ty.name == "array<Light, 4>")
                    .unwrap()
                    .id
                    .as_str(),
            )
            .unwrap();
        assert_eq!(
            format!("{items:?}"),
            format!("{again:?}"),
            "a nested type should be the same view as the one named by its id"
        );
    }

    #[test]
    fn every_type_in_a_schema_resolves() {
        let types = array_member_types();
        let mut views = Views::new(&types);
        for ty in &types {
            let view = views.get(ty.id.as_str()).unwrap();
            assert_eq!(ty.id.as_str(), view_name_id(&types, &view.name), "{ty:?}");
        }
    }

    fn view_name_id<'a>(types: &'a [TypeInfo], name: &str) -> &'a str {
        types
            .iter()
            .find(|ty| ty.name == name)
            .map(|ty| ty.id.as_str())
            .expect("the view should name a type in the schema")
    }

    #[test]
    fn an_unknown_type_id_is_reported() {
        let types = array_member_types();
        let mut views = Views::new(&types);
        let err = views.get("t99").unwrap_err().to_string();
        assert!(err.contains("the schema has no type `t99`"), "{err}");
    }

    #[test]
    fn byte_offsets_become_element_indices() {
        let camera = struct_view(&camera_schema().types, "Camera");
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
        let types = types_of(|interner| {
            vector_ty(
                interner,
                "vec2<f64>",
                2,
                ScalarInfo {
                    name: "float".to_string(),
                    width: 8,
                },
            );
        });
        let view = struct_view(&types, "vec2<f64>");
        assert_eq!(view.storage().unwrap(), "Float64Array");
        assert_eq!(view.element_width().unwrap(), 8);
        assert_eq!(view.element_offset(8).unwrap(), 1);
    }

    #[test]
    fn offsets_that_are_not_whole_elements_are_rejected() {
        let camera = struct_view(&camera_schema().types, "Camera");
        let err = camera.element_offset(2).unwrap_err().to_string();
        assert!(
            err.contains("not a multiple of the 4-byte element"),
            "{err}"
        );
    }

    #[test]
    fn mixed_scalar_storage_is_rejected() {
        let types = types_of(|interner| {
            let a = scalar_ty(interner, "float", f32());
            let b = scalar_ty(interner, "sint", i32());
            struct_ty(interner, "Mixed", 16, 16, &[("a", 0, a), ("b", 4, b)]);
        });
        let view = struct_view(&types, "Mixed");
        let err = view.storage().unwrap_err().to_string();
        assert!(err.contains("cannot share one buffer"), "{err}");
    }

    #[test]
    fn array_members_get_a_deduplicated_accessor_class() {
        let schema = schema(array_member_types());
        let view = struct_view(&schema.types, "Lights");
        let members = &view.as_struct().unwrap().members;

        // The stride reaches TypeScript in elements, not bytes.
        let access = member_access(&view, &members[0]).unwrap();
        assert_eq!(
            access,
            Access::Array {
                start: 0,
                stride: 4,
                count: Some(4)
            }
        );
        assert_eq!(members[0].view.ts_type().unwrap(), "LightArray");

        let typescript = TypeScriptBuilder::new().schema(&schema).build().unwrap();
        assert_eq!(
            typescript.matches("export class LightArray").count(),
            1,
            "{typescript}"
        );
        assert!(
            typescript.contains("stride: number; // em elementos"),
            "{typescript}"
        );
        assert!(
            typescript
                .contains("get length(): number { return this.buffer.length / this.stride; }")
        );
        // A fixed array is clipped to its own count.
        assert!(
            typescript.contains(
                "get items(): LightArray { return new LightArray(this.buffer.subarray(0, 16), 4); }"
            ),
            "{typescript}"
        );
        assert!(
            typescript.contains("this.buffer = new Float32Array(16); // 64 bytes / 4"),
            "{typescript}"
        );
        assert!(
            typescript.contains("this.buffer.set(items.buffer, 0);"),
            "{typescript}"
        );
        // An array of structs needs the `.view` factory: a struct cannot be built
        // from a buffer with `new`, since it takes its members.
        assert!(
            typescript
                .contains("return Light.view(this.buffer.subarray(start, start + this.stride));"),
            "{typescript}"
        );
    }

    #[test]
    fn runtime_sized_arrays_extend_the_allocation() {
        let schema = schema(runtime_array_types());
        let view = struct_view(&schema.types, "Particles");
        let members = &view.as_struct().unwrap().members;

        // No count in the schema, so the view is left unbounded.
        assert_eq!(
            member_access(&view, &members[0]).unwrap(),
            Access::Array {
                start: 0,
                stride: 4,
                count: None
            }
        );

        let typescript = TypeScriptBuilder::new().schema(&schema).build().unwrap();
        assert!(
            typescript.contains("get data(): Vector4f32Array { return new Vector4f32Array(this.buffer.subarray(0), 4); }"),
            "{typescript}"
        );
        assert!(
            typescript.contains("this.buffer = new Float32Array(Math.max(4, data.buffer.length));"),
            "{typescript}"
        );
    }

    #[test]
    fn structs_can_be_built_over_a_foreign_buffer() {
        let typescript = TypeScriptBuilder::new()
            .schema(&camera_schema())
            .shader("camera")
            .build()
            .unwrap();
        // A struct getter used to emit `new X(subarray)`, which cannot compile:
        // the struct constructor allocates from its members instead.
        assert!(
            typescript.contains(
                "get position(): Vector4f32 { return Vector4f32.view(this.buffer.subarray(0, 4)); }"
            ),
            "{typescript}"
        );
        assert!(
            typescript.contains("static view(buffer: Float32Array): Vector4f32 {"),
            "{typescript}"
        );
    }

    #[test]
    fn opaque_types_cannot_be_struct_members() {
        let types = types_of(|interner| {
            // A scalar member is needed so the struct resolves its backing
            // storage and the failure lands on the opaque member instead.
            let count = scalar_ty(interner, "f32", f32());
            let tex = interner.intern("sampler".to_string(), |_| TypeDescriptor::Sampler {
                comparison: false,
            });
            struct_ty(
                interner,
                "Bad",
                8,
                4,
                &[("count", 0, count), ("tex", 4, tex)],
            );
        });
        let typescript = TypeScriptBuilder::new()
            .schema(&schema(types))
            .build()
            .unwrap_err()
            .to_string();
        assert!(
            typescript.contains("cannot be exposed as a view yet"),
            "{typescript}"
        );
    }

    #[test]
    fn opaque_descriptors_build_as_opaque_views() {
        // Images and samplers have no buffer representation yet, but they are
        // recognised so that using one *inside a struct* reports a clear error
        // instead of panicking.
        for (descriptor, tsname) in [
            (
                TypeDescriptor::Image {
                    dimension: ImageDimension::D2,
                    arrayed: false,
                    class: ImageClass::External,
                },
                "GPUImage",
            ),
            (TypeDescriptor::Sampler { comparison: false }, "GPUSampler"),
        ] {
            let types = types_of(|interner| {
                interner.intern("texture".to_string(), |_| descriptor.clone());
            });
            let view = struct_view(&types, "texture");
            assert!(matches!(view.kind, ViewKind::Opaque { .. }), "{view:?}");
            assert_eq!(view.ts_type().unwrap(), tsname);
        }
    }

    #[test]
    fn names_are_sanitized_into_identifiers() {
        let types = types_of(|interner| {
            struct_ty(interner, "anon_struct<f32, i32>", 8, 4, &[]);
        });
        let view = struct_view(&types, "anon_struct<f32, i32>");
        assert_eq!(view.ts_type().unwrap(), "anon_struct_f32__i32_");
    }

    #[test]
    fn a_pipeline_needs_a_name_to_be_exported_under() {
        // The schema has nowhere to record the shader's name, so generating its
        // helper without one would have to invent a class name.
        let schema = schema_with_pipeline(array_member_types(), empty_pipeline());
        let err = TypeScriptBuilder::new()
            .schema(&schema)
            .build()
            .unwrap_err()
            .to_string();
        assert!(err.contains("no shader name"), "{err}");

        // With one, the file carries both the support section and the instance.
        let typescript = TypeScriptBuilder::new()
            .schema(&schema)
            .shader("lights")
            .build()
            .unwrap();
        assert!(
            typescript.contains("export const LightsPipelineHelper = new PipelineHelper({"),
            "{typescript}"
        );
    }

    #[test]
    fn the_class_section_is_stable_and_scalar_keyed() {
        let schema = camera_schema();

        let typescript = TypeScriptBuilder::new()
            .schema(&schema)
            .shader("camera")
            .build()
            .unwrap();

        let expected = concat!(
            "export class Vector4f32 {\n",
            "  public buffer: Float32Array;\n",
            "  constructor(buffer: Float32Array) {\n",
            "    this.buffer = buffer;\n",
            "  }\n",
            "  static view(buffer: Float32Array): Vector4f32 { const self = Object.create(Vector4f32.prototype) as Vector4f32; self.buffer = buffer; return self; }\n",
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
            "  public buffer: Float32Array;\n",
            "  constructor(buffer: Float32Array) {\n",
            "    this.buffer = buffer;\n",
            "  }\n",
            "  static view(buffer: Float32Array): Mat4x4f32 { const self = Object.create(Mat4x4f32.prototype) as Mat4x4f32; self.buffer = buffer; return self; }\n",
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
            "  public buffer: Float32Array;\n",
            "  constructor(position: Vector4f32, rot: Mat4x4f32) {\n",
            "    this.buffer = new Float32Array(20); // 80 bytes / 4\n",
            "    this.buffer.set(position.buffer, 0); // offset 0 bytes / 4 = 0\n",
            "    this.buffer.set(rot.buffer, 4); // offset 16 bytes / 4 = 4\n",
            "  }\n",
            "  static view(buffer: Float32Array): Camera { const self = Object.create(Camera.prototype) as Camera; self.buffer = buffer; return self; }\n",
            "  get position(): Vector4f32 { return Vector4f32.view(this.buffer.subarray(0, 4)); }\n",
            "  get rot(): Mat4x4f32 { return Mat4x4f32.view(this.buffer.subarray(4, 20)); }\n",
            "}\n",
        );
        // The pipeline block sits in front of the classes, so the type classes keep
        // comparing exactly as they did.
        assert!(typescript.ends_with(expected), "{typescript}");
    }
}

/// The batch half of the generator: the same declarations, spread over a folder.
///
/// A single schema generates one self-contained file. A folder generates a
/// `support.ts`, one file per shared accessor class, and one file per shader
/// that imports the first two instead of repeating them.
#[cfg(test)]
mod batch {
    use std::path::Path;

    use anygpu::{ShaderMetadata, naga};

    use super::*;

    /// `struct Light { color: vec3<f32>, intensity: f32 }`
    const LIGHTS_WGSL: &str = r"
struct Light {
    color: vec3<f32>,
    intensity: f32,
}
struct Lights {
    items: array<Light, 4>,
}
@group(0) @binding(0) var<storage, read> lights: Lights;
";

    /// `struct Camera { position: vec4<f32>, rot: mat4x4<f32> }`
    const CAMERA_WGSL: &str = r"
struct Camera {
    position: vec4<f32>,
    rot: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;
";

    fn schema(wgsl: &str) -> ShaderBindings {
        let module = naga::front::wgsl::parse_str(wgsl).expect("the shader should parse");
        ShaderMetadata::new(&module)
            .expect("the shader should expose bindings")
            .generate_bindings()
    }

    /// The batch a set of shaders generates, keyed by its output path.
    fn batch(named: &[(&str, &str)]) -> BTreeMap<String, String> {
        let schemas: Vec<(&str, ShaderBindings)> = named
            .iter()
            .map(|(name, wgsl)| (*name, schema(wgsl)))
            .collect();
        generate_batch(schemas.iter().map(|(name, schema)| (*name, schema)))
            .expect("the batch should generate")
            .into_iter()
            .map(|file| (file.path.display().to_string(), file.contents))
            .collect()
    }

    fn get<'a>(files: &'a BTreeMap<String, String>, path: &str) -> &'a str {
        files
            .get(path)
            .unwrap_or_else(|| panic!("`{path}` should be in the batch: {:?}", files.keys()))
    }

    #[test]
    fn the_support_section_is_written_once_for_the_whole_folder() {
        let files = batch(&[("a", LIGHTS_WGSL), ("b", CAMERA_WGSL)]);

        assert_eq!(
            get(&files, "support.ts"),
            crate::codegen::pipeline::SUPPORT,
            "the batch writes the checked-in support section verbatim"
        );
        // Once, not once per shader.
        assert_eq!(files.keys().filter(|path| *path == "support.ts").count(), 1);
        // And no shader file repeats it.
        for (path, contents) in &files {
            if path.ends_with(".ts") && path != "support.ts" {
                assert!(
                    !contents.contains("export class PipelineHelper"),
                    "`{path}` should import the support section, not repeat it"
                );
            }
        }
    }

    #[test]
    fn a_shared_class_is_written_once_however_many_shaders_use_it() {
        // Both shaders use `vec3<f32>` and `vec4<f32>`, and only one uses a matrix.
        let files = batch(&[("a", LIGHTS_WGSL), ("b", CAMERA_WGSL), ("c", LIGHTS_WGSL)]);

        assert_eq!(
            get(&files, "builtins/vector3f32.ts")
                .matches("export class Vector3f32")
                .count(),
            1
        );
        // `vec4<f32>` is named by `Camera` here, but a matrix pulls in no vector.
        assert!(files.contains_key("builtins/vector4f32.ts"));
        assert!(files.contains_key("builtins/mat4x4f32.ts"));
        // Three shaders, still one file each.
        for path in ["builtins/vector3f32.ts", "builtins/vector4f32.ts"] {
            assert_eq!(
                files.keys().filter(|p| *p == path).count(),
                1,
                "`{path}` should be written once"
            );
        }
        // No shader file declares a shared class any more.
        for (path, contents) in &files {
            if path == "a.ts" || path == "b.ts" || path == "c.ts" {
                for shared in ["Vector3f32", "Vector4f32", "Mat4x4f32"] {
                    assert!(
                        !contents.contains(&format!("export class {shared}")),
                        "`{path}` should not declare `{shared}`: {contents}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_shader_file_imports_what_it_uses() {
        let files = batch(&[("a", LIGHTS_WGSL), ("b", CAMERA_WGSL)]);

        let a = get(&files, "a.ts");
        assert!(
            a.contains("import { Vector3f32 } from \"./builtins/vector3f32\";"),
            "{a}"
        );
        // `a` has no matrix, so it does not import one.
        assert!(!a.contains("mat4x4f32"), "{a}");
        // Its own structs stay with it.
        assert!(a.contains("export class Light {"), "{a}");
        assert!(a.contains("export class Lights {"), "{a}");

        let b = get(&files, "b.ts");
        assert!(
            b.contains("import { Vector4f32 } from \"./builtins/vector4f32\";"),
            "{b}"
        );
        assert!(
            b.contains("import { Mat4x4f32 } from \"./builtins/mat4x4f32\";"),
            "{b}"
        );
        assert!(b.contains("export class Camera {"), "{b}");
    }

    #[test]
    fn a_shared_array_imports_the_element_class() {
        // `Particles.data: array<vec4<f32>>` makes a `Vector4f32Array` whose
        // `get` returns a `Vector4f32` — a sibling, not the shader's own file.
        let files = batch(&[("a", RUNTIME_ARRAY_WGSL)]);
        let array = get(&files, "builtins/vector4f32_array.ts");

        assert!(
            array.contains("import { Vector4f32 } from \"./vector4f32\";"),
            "{array}"
        );
        assert!(array.contains("export class Vector4f32Array {"), "{array}");
        assert!(
            array.contains("get(index: number): Vector4f32 {"),
            "{array}"
        );
    }

    const RUNTIME_ARRAY_WGSL: &str = r"
struct Particles {
    data: array<vec4<f32>>,
}
@group(0) @binding(0) var<storage, read> particles: Particles;
";

    #[test]
    fn an_array_of_structs_stays_with_the_shader_that_declared_it() {
        // `LightArray` is named after a struct, and a struct is per shader. The
        // class and the type it names have to land in the same file, so neither
        // one can be shared.
        let files = batch(&[("a", LIGHTS_WGSL)]);

        assert!(
            !files.contains_key("builtins/light_array.ts"),
            "{:?}",
            files.keys()
        );
        let a = get(&files, "a.ts");
        assert!(a.contains("export class LightArray {"), "{a}");
        assert!(a.contains("get(index: number): Light {"), "{a}");
        // `Light` is declared in this same file, so it is not imported.
        assert!(!a.contains("light_array"), "{a}");
    }

    #[test]
    fn two_shaders_may_declare_the_same_struct_name_differently() {
        // Each shader wants a `LightArray`, but over its own `Light`. Sharing one
        // would type one of them against the other's struct, so neither does.
        let files = batch(&[("a", LIGHT_A_WGSL), ("b", LIGHT_B_WGSL)]);

        assert!(
            !files.contains_key("builtins/light_array.ts"),
            "{:?}",
            files.keys()
        );
        for name in ["a.ts", "b.ts"] {
            let file = get(&files, name);
            assert!(file.contains("export class LightArray {"), "{name}: {file}");
            assert!(file.contains("export class Light {"), "{name}: {file}");
        }
        // The two `Light` classes are still each shader's own.
        let a = get(&files, "a.ts");
        let b = get(&files, "b.ts");
        assert!(
            a.contains("constructor(color: Vector3f32, intensity: number)"),
            "{a}"
        );
        assert!(
            b.contains("constructor(pos: Vector4f32, tint: Vector2f32)"),
            "{b}"
        );
        // What the two have in common — a `vec4<f32>` — is still written once.
        assert!(files.contains_key("builtins/vector4f32.ts"));
    }

    const LIGHT_A_WGSL: &str = r"
struct Light {
    color: vec3<f32>,
    intensity: f32,
}
struct LightsA {
    items: array<Light, 4>,
}
@group(0) @binding(0) var<uniform> lights: LightsA;
";

    const LIGHT_B_WGSL: &str = r"
struct Light {
    pos: vec4<f32>,
    tint: vec2<f32>,
}
struct LightsB {
    items: array<Light, 8>,
}
@group(0) @binding(0) var<uniform> lights: LightsB;
";

    #[test]
    fn every_import_points_at_a_file_the_batch_wrote() {
        let files = batch(&[("a", LIGHTS_WGSL), ("b", CAMERA_WGSL)]);

        for (path, contents) in &files {
            for line in contents.lines().filter(|line| line.starts_with("import ")) {
                let specifier = line
                    .rsplit_once("from \"")
                    .and_then(|(_, rest)| rest.split_once('"'))
                    .map(|(specifier, _)| specifier)
                    .unwrap_or_else(|| panic!("`{path}` has an unparsable import: {line}"));
                // Every file is at most one folder deep, so the specifier is
                // relative to the importing file's own folder. `Path::join` does
                // not resolve the `./` the generator writes, so it goes first.
                let relative = specifier.strip_prefix("./").unwrap_or(specifier);
                let folder = Path::new(path).parent().unwrap_or(Path::new(""));
                let target = folder.join(format!("{relative}.ts")).display().to_string();
                assert!(
                    files.contains_key(&target),
                    "`{path}` imports `{specifier}`, which the batch never wrote: {:?}",
                    files.keys()
                );
            }
        }
    }

    #[test]
    fn a_shader_file_names_the_support_it_uses() {
        // The pipeline instance names `PipelineHelper`, so it is imported.
        let files = batch(&[("a", LIGHTS_WGSL)]);
        let a = get(&files, "a.ts");
        assert!(
            a.contains("export const APipelineHelper = new PipelineHelper({"),
            "{a}"
        );
        assert!(
            a.contains("import { PipelineHelper } from \"./support\";"),
            "{a}"
        );
        // `LIGHTS_WGSL` resolves no stage, so every mask is a bare `0` and the
        // stage flags are never named.
        assert!(a.contains("visibility: 0"), "{a}");
        assert!(!a.contains("AnygpuShaderStage"), "{a}");
    }

    #[test]
    fn a_schema_with_no_pipeline_imports_no_helper() {
        // `generate_bindings` always describes a pipeline, even for a shader with
        // no entry points, so the pipeline is taken away to reach this case.
        let mut bindings = schema(CAMERA_WGSL);
        bindings.pipelines = None;
        let files = generate_batch([("a", &bindings)]).expect("it should generate");
        let a = &files
            .iter()
            .find(|file| file.path.ends_with("a.ts"))
            .expect("`a.ts` should be written")
            .contents;

        assert!(!a.contains("PipelineHelper"), "{a}");
        assert!(!a.contains("from \"./support\""), "{a}");
        // It still has types, so it still imports the shared ones.
        assert!(a.contains("export class Camera {"), "{a}");
    }

    #[test]
    fn a_folder_of_no_schemas_still_gets_the_support_section() {
        // The support section depends on no schema, so it is true of any batch.
        let files = generate_batch(std::iter::empty()).expect("an empty batch should generate");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, PathBuf::from("support.ts"));
        assert_eq!(files[0].contents, crate::codegen::pipeline::SUPPORT);
    }

    /// The text of `export class <name> { .. }` in a generated file.
    ///
    /// A generated class is a top-level declaration, so it ends at the first
    /// line that is nothing but the closing brace. Whatever follows it — a blank
    /// line, another class, the end of the file — is not part of it.
    fn class_of(typescript: &str, name: &str) -> String {
        let open = format!("export class {name} {{");
        let start = typescript
            .find(&open)
            .unwrap_or_else(|| panic!("`{name}` should be declared in:\n{typescript}"));
        let body = &typescript[start..];
        let end = body
            .lines()
            .skip(1)
            .position(|line| line == "}")
            .map(|lines| lines + 2)
            .unwrap_or_else(|| panic!("`{name}` should be closed in:\n{typescript}"));
        body.lines().take(end).collect::<Vec<_>>().join("\n")
    }

    #[test]
    fn splitting_the_output_does_not_change_what_a_class_says() {
        // The declarations are the same either way. Only the file each one lands
        // in changes, so a class read out of a shared file has to match the same
        // class read out of that shader's self-contained output, character for
        // character.
        let files = batch(&[("a", LIGHTS_WGSL), ("b", CAMERA_WGSL)]);
        // Every class, and which file of the batch it moved to.
        let expected: &[(&str, &str, &[&str])] = &[
            (
                "a",
                LIGHTS_WGSL,
                &["Vector3f32", "Light", "Lights", "LightArray"],
            ),
            ("b", CAMERA_WGSL, &["Vector4f32", "Mat4x4f32", "Camera"]),
        ];

        for (name, wgsl, classes) in expected {
            let one = generate_typescript(&schema(wgsl), name).expect("it should generate");
            let shader_file = get(&files, &format!("{name}.ts"));
            for class in *classes {
                let shared = files.contains_key(&format!(
                    "builtins/{}.ts",
                    super::view::builtin_module(class)
                ));
                // A class is either shared or the shader's own, never both and
                // never neither.
                assert_eq!(
                    shared,
                    !shader_file.contains(&format!("export class {class} {{")),
                    "`{class}` should be in exactly one place"
                );
                let home = if shared {
                    let path = format!("builtins/{}.ts", super::view::builtin_module(class));
                    get(&files, &path)
                } else {
                    shader_file
                };
                assert_eq!(
                    class_of(home, class),
                    class_of(&one, class),
                    "`{class}` should be declared exactly as it was inline"
                );
            }
        }
    }
}
