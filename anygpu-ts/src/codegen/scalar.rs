use anygpu::ScalarInfo;
use color_eyre::eyre::{Result, eyre};

/// How one WGSL scalar is represented in TypeScript.
///
/// Every scalar becomes a plain `number` (or `bigint` for 64-bit integers) when
/// read, but each one has its own storage width, typed array constructor and
/// generated-name suffix, so the mapping lives in exactly one place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScalarLayout {
    pub ts_type: &'static str,
    pub array: &'static str,
    pub byte_size: u32,
    pub suffix: &'static str,
}

const F32: ScalarLayout = ScalarLayout {
    ts_type: "number",
    array: "Float32Array",
    byte_size: 4,
    suffix: "f32",
};

const F64: ScalarLayout = ScalarLayout {
    ts_type: "number",
    array: "Float64Array",
    byte_size: 8,
    suffix: "f64",
};

const I32: ScalarLayout = ScalarLayout {
    ts_type: "number",
    array: "Int32Array",
    byte_size: 4,
    suffix: "i32",
};

const U32: ScalarLayout = ScalarLayout {
    ts_type: "number",
    array: "Uint32Array",
    byte_size: 4,
    suffix: "u32",
};

const I64: ScalarLayout = ScalarLayout {
    ts_type: "bigint",
    array: "BigInt64Array",
    byte_size: 8,
    suffix: "i64",
};

const U64: ScalarLayout = ScalarLayout {
    ts_type: "bigint",
    array: "BigUint64Array",
    byte_size: 8,
    suffix: "u64",
};

const BOOL: ScalarLayout = ScalarLayout {
    ts_type: "number",
    array: "Uint32Array",
    byte_size: 4,
    suffix: "bool",
};

pub fn layout(scalar: &ScalarInfo) -> Result<ScalarLayout> {
    match (scalar.name.as_str(), scalar.width) {
        ("float", 4) => Ok(F32),
        ("float", 8) => Ok(F64),
        ("sint", 4) => Ok(I32),
        ("sint", 8) => Ok(I64),
        ("uint", 4) => Ok(U32),
        ("uint", 8) => Ok(U64),
        ("bool", 4) => Ok(BOOL),
        (name, width) => Err(eyre!(
            "scalar `{name}` with width {width} has no TypeScript layout"
        )),
    }
}
