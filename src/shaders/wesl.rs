use std::path::{Component, Path};

use wesl::{
    CompileOptions, EscapeMangler, Mangler, SourceMap, Wesl,
    syntax::{ModulePath, PathOrigin},
};

use super::{
    ReflectedModule, ShaderMetadata,
    pipeline::types::{BindGroupLayoutEntry, PipelineDescriptor},
    types::DeclarationIdentity,
};

pub fn compile_wesl_shader(
    base: impl AsRef<Path>,
    root: impl AsRef<Path>,
    module_id: impl Into<String>,
) -> color_eyre::Result<ReflectedModule> {
    let root = root.as_ref();
    let root_path = module_path(root)?;
    let mut compiler = Wesl::new(base);
    compiler.set_options(CompileOptions {
        mangle_root: true,
        ..CompileOptions::default()
    });
    let compiled = compiler.compile(&root_path)?;
    let source_map = compiled
        .sourcemap
        .as_ref()
        .ok_or_else(|| color_eyre::eyre::eyre!("WESL compilation did not produce a source map"))?;
    let wgsl = compiled.to_string();
    let module = naga::front::wgsl::parse_str(&wgsl)?;

    let mangler = EscapeMangler;
    let mut identities = std::collections::HashMap::new();
    for (handle, ty) in module.types.iter() {
        if !matches!(ty.inner, naga::TypeInner::Struct { .. }) {
            continue;
        }
        let Some(name) = ty.name.as_deref() else {
            continue;
        };
        let Some((path, declaration)) = source_map.get_decl(name) else {
            continue;
        };
        identities.insert(
            handle,
            DeclarationIdentity {
                module_path: path.to_string(),
                name: declaration.to_owned(),
            },
        );
    }

    let mut reflected =
        ShaderMetadata::new(&module)?.generate_reflected_module(module_id.into(), identities)?;
    restore_names(&mut reflected.bindings, source_map, &mangler);
    for ty in &mut reflected.bindings.types {
        if let Some(identity) = reflected.identities.get(&ty.id) {
            ty.name = identity.name.clone();
        }
    }
    Ok(reflected)
}

fn module_path(root: &Path) -> color_eyre::Result<ModulePath> {
    use color_eyre::eyre::{WrapErr, eyre};

    if root.is_absolute() {
        return Err(eyre!(
            "WESL root path must be relative to its source directory"
        ));
    }
    if root.extension().and_then(|extension| extension.to_str()) != Some("wesl") {
        return Err(eyre!("WESL root file must have a `.wesl` extension"));
    }

    let mut components = Vec::new();
    for component in root.with_extension("").components() {
        match component {
            Component::Normal(name) => components.push(
                name.to_str()
                    .ok_or_else(|| eyre!("WESL module path is not valid UTF-8"))?
                    .to_owned(),
            ),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(eyre!(
                    "WESL root path must stay inside its source directory"
                ));
            }
        }
    }
    if components.is_empty() {
        return Err(eyre!("WESL root path must name a module"));
    }

    let mut module = ModulePath::new(PathOrigin::Absolute, Vec::new());
    module.components = components;
    module.components.iter().try_for_each(|component| {
        let parsed = format!("package::{component}").parse::<ModulePath>();
        parsed
            .map(|_| ())
            .wrap_err_with(|| format!("invalid WESL module name `{component}`"))
    })?;
    Ok(module)
}

fn restore_name(name: &mut String, source_map: &impl SourceMap, mangler: &EscapeMangler) {
    if let Some((_, declaration)) = source_map.get_decl(name) {
        *name = declaration.to_owned();
    } else if let Some((_, declaration)) = mangler.unmangle(name) {
        *name = declaration;
    }
}

fn restore_bindgroups(
    bindgroups: &mut [BindGroupLayoutEntry],
    source_map: &impl SourceMap,
    mangler: &EscapeMangler,
) {
    for binding in bindgroups {
        restore_name(&mut binding.name, source_map, mangler);
    }
}

fn restore_names(
    bindings: &mut super::ShaderBindings,
    source_map: &impl SourceMap,
    mangler: &EscapeMangler,
) {
    restore_bindgroups(&mut bindings.bindgroups, source_map, mangler);
    if let Some(pipeline) = &mut bindings.pipelines {
        restore_pipeline(pipeline, source_map, mangler);
    }
}

fn restore_pipeline(
    pipeline: &mut PipelineDescriptor,
    source_map: &impl SourceMap,
    mangler: &EscapeMangler,
) {
    for group in &mut pipeline.layout.bind_group_layouts {
        restore_bindgroups(&mut group.entries, source_map, mangler);
    }
    if let Some(vertex) = &mut pipeline.vertex {
        restore_name(&mut vertex.entry_point, source_map, mangler);
    }
    if let Some(fragment) = &mut pipeline.fragment {
        restore_name(&mut fragment.entry_point, source_map, mangler);
    }
    if let Some(compute) = &mut pipeline.compute {
        restore_name(&mut compute.entry_point, source_map, mangler);
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::{LinkedTypeInfo, link_modules};

    struct TempDirectory(std::path::PathBuf);

    impl TempDirectory {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock should be after the Unix epoch")
                .as_nanos();
            let path =
                std::env::temp_dir().join(format!("anygpu-wesl-{}-{nonce}", std::process::id()));
            std::fs::create_dir_all(&path).expect("temporary WESL directory should be created");
            Self(path)
        }
    }

    impl Drop for TempDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_shader(base: &Path, name: &str, body: &str) {
        std::fs::write(base.join(name), body).expect("WESL fixture should be written");
    }

    fn camera_type(schema: &crate::LinkedSchema) -> &LinkedTypeInfo {
        schema
            .types
            .iter()
            .find(|ty| {
                ty.identity
                    .as_ref()
                    .is_some_and(|identity| identity.name == "Camera")
            })
            .expect("the linked schema should contain Camera")
    }

    #[test]
    fn shared_imports_link_and_root_names_are_restored() {
        let temp = TempDirectory::new();
        write_shader(
            &temp.0,
            "common.wesl",
            "struct Camera { view: mat4x4<f32>, }",
        );
        write_shader(
            &temp.0,
            "first.wesl",
            r"
import package::common::Camera;
@group(0) @binding(0) var<uniform> camera: Camera;
@compute @workgroup_size(1)
fn run_first() { let view = camera.view; }
",
        );
        write_shader(
            &temp.0,
            "second.wesl",
            r"
import package::common::Camera;
@group(0) @binding(0) var<uniform> camera: Camera;
@compute @workgroup_size(1)
fn run_second() { let view = camera.view; }
",
        );

        let first = compile_wesl_shader(&temp.0, Path::new("first.wesl"), "package::first")
            .expect("first WESL root should compile");
        let second = compile_wesl_shader(&temp.0, Path::new("second.wesl"), "package::second")
            .expect("second WESL root should compile");
        assert_eq!(
            first
                .bindings
                .pipelines
                .as_ref()
                .unwrap()
                .compute
                .as_ref()
                .unwrap()
                .entry_point,
            "run_first"
        );
        assert_eq!(
            first
                .bindings
                .pipelines
                .as_ref()
                .unwrap()
                .layout
                .bind_group_layouts[0]
                .entries[0]
                .name,
            "camera"
        );

        let linked = link_modules(vec![first, second]).expect("shared WESL type should link");
        let camera = camera_type(&linked);
        assert_eq!(
            camera.identity.as_ref().unwrap().module_path,
            "package::common"
        );
        assert_eq!(
            linked
                .types
                .iter()
                .filter(|ty| ty
                    .identity
                    .as_ref()
                    .is_some_and(|identity| identity.name == "Camera"))
                .count(),
            1
        );
    }

    #[test]
    fn root_declarations_receive_wesl_identity() {
        let temp = TempDirectory::new();
        write_shader(
            &temp.0,
            "root.wesl",
            r"
struct Camera { view: mat4x4<f32>, }
@group(0) @binding(0) var<uniform> camera: Camera;
@compute @workgroup_size(1)
fn run_root() { let view = camera.view; }
",
        );

        let reflected = compile_wesl_shader(&temp.0, Path::new("root.wesl"), "package::root")
            .expect("root declarations should compile");
        let identity = reflected
            .identities
            .values()
            .find(|identity| identity.name == "Camera")
            .expect("root Camera should have an identity");
        assert_eq!(identity.module_path, "package::root");
    }
}
