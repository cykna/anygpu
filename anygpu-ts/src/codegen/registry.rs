use std::collections::{BTreeMap, BTreeSet};

use crate::codegen::scalar::ScalarLayout;
use crate::codegen::view::{View, ViewKind, identifier};

/// The set of accessor types that need to be declared, deduplicated and in a
/// stable order regardless of the order the schema listed them in.
#[derive(Debug, Default, Clone)]
pub struct Registry {
    vectors: BTreeSet<(u8, ScalarLayout)>,
    matrices: BTreeSet<(u8, u8, ScalarLayout)>,
    arrays: BTreeMap<String, View>,
    structs: BTreeMap<String, View>,
}

impl Registry {
    pub fn add_vector(&mut self, length: u8, scalar: ScalarLayout) {
        self.vectors.insert((length, scalar));
    }

    pub fn add_matrix(&mut self, columns: u8, rows: u8, scalar: ScalarLayout) {
        self.matrices.insert((columns, rows, scalar));
    }

    /// Registers an array accessor class under its generated name, e.g.
    /// `LightArray` for `array<Light, 4>`. The name does not encode the element
    /// count or the stride — both are constructor arguments — so the same class
    /// serves every `array<Light, N>` in the schema.
    pub fn add_array(&mut self, name: String, element: View) {
        self.arrays.entry(name).or_insert(element);
    }

    pub fn add_struct(&mut self, view: View) {
        self.structs.entry(identifier(&view.name)).or_insert(view);
    }

    pub fn vectors(&self) -> impl Iterator<Item = (u8, ScalarLayout)> + '_ {
        self.vectors.iter().copied()
    }

    pub fn matrices(&self) -> impl Iterator<Item = (u8, u8, ScalarLayout)> + '_ {
        self.matrices.iter().copied()
    }

    /// Each registered array accessor class, paired with a view of its element.
    pub fn arrays(&self) -> impl Iterator<Item = (&String, &View)> {
        self.arrays.iter()
    }

    /// The array accessor classes a batch can share, paired with a view of the
    /// element.
    ///
    /// An array is shareable when its element is not a struct. The class is then
    /// named after the element's own accessor type and holds nothing but a
    /// stride, so two shaders that both hold `array<vec3<f32>>` want the same
    /// class, character for character. An array *of structs* is named after a
    /// type each shader declares for itself, so it is not shareable: two shaders
    /// can each have a differently shaped `Light` and still both want a
    /// `LightArray` over it.
    pub fn builtin_arrays(&self) -> impl Iterator<Item = (&String, &View)> {
        self.arrays
            .iter()
            .filter(|(_, element)| !matches!(element.kind, ViewKind::Struct(_)))
    }

    /// The array accessor classes that stay in the shader that declared them,
    /// each paired with a view of its struct element.
    pub fn struct_arrays(&self) -> impl Iterator<Item = (&String, &View)> {
        self.arrays
            .iter()
            .filter(|(_, element)| matches!(element.kind, ViewKind::Struct(_)))
    }

    pub fn structs(&self) -> impl Iterator<Item = &View> {
        self.structs.values()
    }
}
