pub mod types;
use naga::{Handle, Module, Type, TypeInner, proc::Layouter};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::shaders::types::TypeInfo;
#[derive(Debug, Serialize, Deserialize)]
pub struct ShaderBindings {
    types: Vec<TypeInfo>,
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

    pub fn get_types(&self) -> impl Iterator<Item = TypeInfo> {
        self.external_types.iter().map(|ty| self.get_type_info(*ty))
    }

    pub fn generate_bindings(self) -> ShaderBindings {
        ShaderBindings {
            types: self.get_types().collect(),
        }
    }
}
