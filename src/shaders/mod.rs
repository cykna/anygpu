pub mod image;
pub mod pipeline;
pub mod types;
use naga::{Handle, Module, Type, TypeInner, proc::Layouter};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::shaders::{
    pipeline::types::{BindGroupLayoutEntry, PipelineDescriptor},
    types::{TypeInfo, TypeInterner},
};
#[derive(Debug, Serialize, Deserialize)]
pub struct SharedTypes {
    #[serde(default)]
    pub types: Vec<TypeInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ShaderBindings {
    #[serde(default)]
    pub types: Vec<TypeInfo>,
    #[serde(default)]
    pub shared: Option<SharedTypes>,
    /// The pipeline this shader needs, or `None` when the schema carries types
    /// only. A generator emits whatever a shader's consumers need, so a missing
    /// pipeline is not an error: it only means there is nothing to describe one
    /// with.
    #[serde(default, alias = "pipeline_descriptor")]
    pub pipelines: Option<PipelineDescriptor>,
    #[serde(default)]
    pub bindgroups: Vec<BindGroupLayoutEntry>,
}
#[derive(Debug)]
pub struct ShaderMetadata<'a> {
    external_types: HashSet<Handle<Type>>,
    module: &'a Module,
    layouter: Layouter,
}
impl<'a> ShaderMetadata<'a> {
    pub fn get_external_types(module: &Module) -> HashSet<Handle<Type>> {
        let mut types = HashSet::new();
        let mut stack = Vec::new();
        for (_, var) in module.global_variables.iter() {
            if let Some(_) = var.binding {
                stack.push(var.ty);
            }
        }
        for entry_point in &module.entry_points {
            for arg in &entry_point.function.arguments {
                stack.push(arg.ty);
            }
        }

        while let Some(handle) = stack.pop() {
            if !types.insert(handle) {
                continue; // já visitado
            }
            match &module.types[handle].inner {
                TypeInner::Struct { members, .. } => {
                    for m in members {
                        stack.push(m.ty);
                    }
                }
                TypeInner::Array { base, .. }
                | TypeInner::Pointer { base, .. }
                | TypeInner::BindingArray { base, .. } => {
                    stack.push(*base);
                }
                _ => {} // escalares, vetores, matrizes: sem filhos
            }
        }
        types
    }

    pub fn new(module: &'a Module) -> color_eyre::Result<Self> {
        let mut layouter = Layouter::default();
        layouter.update(module.to_ctx())?;
        Ok(Self {
            external_types: Self::get_external_types(&module),
            module,
            layouter,
        })
    }

    /// Every type reachable from a binding or an entry point argument, defined
    /// once each and in dependency order: a type is always listed after
    /// everything it is built from.
    pub fn get_types(&self) -> Vec<TypeInfo> {
        let mut interner = TypeInterner::new();
        let mut memo = HashMap::new();
        // `external_types` is a set, so it is walked in handle order to keep the
        // ids — and therefore the schema — identical from one run to the next.
        let mut roots: Vec<Handle<Type>> = self.external_types.iter().copied().collect();
        roots.sort_by_key(|handle| handle.index());
        for handle in roots {
            self.type_id(&mut interner, &mut memo, handle);
        }
        interner.finish()
    }

    pub fn generate_bindings(self) -> ShaderBindings {
        let types = self.get_types();
        // The types intrinsic to WGSL itself - vectors, scalars, matrices - are
        // shared by every shader, so they are described once, under the shared
        // section, while the rest are the types this shader declares for itself.
        let mut shared_types = Vec::new();
        let mut shader_types = Vec::new();
        for ty in types {
            if ty.descriptor.is_shared() {
                shared_types.push(ty);
            } else {
                shader_types.push(ty);
            }
        }
        let pipelines = self.get_pipeline();
        ShaderBindings {
            types: shader_types,
            shared: Some(SharedTypes {
                types: shared_types,
            }),
            pipelines: Some(pipelines),
            bindgroups: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shaders::types::TypeDescriptor;

    const CAMERA_WGSL: &str = r"
struct Camera {
    position: vec4<f32>,
    rot: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
";

    #[test]
    fn wgsl_intrinsics_go_to_the_shared_section() {
        let module = naga::front::wgsl::parse_str(CAMERA_WGSL).expect("the shader should parse");
        let bindings = ShaderMetadata::new(&module)
            .expect("the shader should expose bindings")
            .generate_bindings();

        // The vectors and matrices WGSL itself defines are shared by every shader,
        // so they are described once, under the shared section, as `st0`, `st1`, and
        // so on.
        let shared = bindings.shared.expect("the intrinsics should be shared");
        let shared_types: Vec<&TypeInfo> = shared.types.iter().collect();
        assert_eq!(shared_types.len(), 2, "{shared_types:?}");
        assert_eq!(shared_types[0].id.as_str(), "st0");
        assert_eq!(shared_types[0].name, "vec4<f32>");
        assert_eq!(shared_types[1].id.as_str(), "st1");
        assert_eq!(shared_types[1].name, "mat4x4<f32>");

        // The struct the shader declares is its own, and its members point at the
        // shared types they are built from.
        let types = &bindings.types;
        assert_eq!(types.len(), 1, "{types:?}");
        let TypeDescriptor::Struct { members, .. } = &types[0].descriptor else {
            panic!("the type should be a struct: {types:?}");
        };
        assert_eq!(members[0].ty.as_str(), "st0");
        assert_eq!(members[1].ty.as_str(), "st1");
    }
}
