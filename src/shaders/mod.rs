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
pub struct ShaderBindings {
    #[serde(default)]
    pub types: Vec<TypeInfo>,
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
        let pipelines = self.get_pipeline();
        ShaderBindings {
            types,
            pipelines: Some(pipelines),
            bindgroups: Vec::new(),
        }
    }
}
