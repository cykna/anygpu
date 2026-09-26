use serde::{Deserialize, Serialize};

use crate::shaders::types::ScalarInfo;

#[derive(Debug, Serialize, Deserialize)]
pub enum StorageFormat {
    R8Unorm,
    R8Snorm,
    R8Uint,
    R8Sint,
    R16Uint,
    R16Sint,
    R16Float,
    Rg8Unorm,
    Rg8Snorm,
    Rg8Uint,
    Rg8Sint,
    R32Uint,
    R32Sint,
    R32Float,
    Rg16Uint,
    Rg16Sint,
    Rg16Float,
    Rgba8Unorm,
    Rgba8Snorm,
    Rgba8Uint,
    Rgba8Sint,
    Bgra8Unorm,
    Rgb10a2Uint,
    Rgb10a2Unorm,
    Rg11b10Ufloat,
    R64Uint,
    Rg32Uint,
    Rg32Sint,
    Rg32Float,
    Rgba16Uint,
    Rgba16Sint,
    Rgba16Float,
    Rgba32Uint,
    Rgba32Sint,
    Rgba32Float,
    R16Unorm,
    R16Snorm,
    Rg16Unorm,
    Rg16Snorm,
    Rgba16Unorm,
    Rgba16Snorm,
}

impl From<naga::StorageFormat> for StorageFormat {
    fn from(value: naga::StorageFormat) -> Self {
        match value {
            naga::StorageFormat::Bgra8Unorm => Self::Bgra8Unorm,
            naga::StorageFormat::R16Float => Self::R16Float,
            naga::StorageFormat::Rg16Float => Self::Rg16Float,
            naga::StorageFormat::Rgba16Float => Self::Rgba16Float,
            naga::StorageFormat::Rgba32Float => Self::Rgba32Float,
            naga::StorageFormat::R16Unorm => Self::R16Unorm,
            naga::StorageFormat::Rg16Unorm => Self::Rg16Unorm,
            naga::StorageFormat::Rgba16Unorm => Self::Rgba16Unorm,
            naga::StorageFormat::R16Snorm => Self::R16Snorm,
            naga::StorageFormat::Rg16Snorm => Self::Rg16Snorm,
            naga::StorageFormat::Rgba16Snorm => Self::Rgba16Snorm,
            naga::StorageFormat::R16Sint => Self::R16Sint,
            naga::StorageFormat::Rg16Sint => Self::Rg16Sint,
            naga::StorageFormat::Rgba16Sint => Self::Rgba16Sint,
            naga::StorageFormat::R32Float => Self::R32Float,
            naga::StorageFormat::R32Sint => Self::R32Sint,
            naga::StorageFormat::R64Uint => Self::R64Uint,
            naga::StorageFormat::Rg32Uint => Self::Rg32Uint,
            naga::StorageFormat::Rg32Sint => Self::Rg32Sint,
            naga::StorageFormat::Rg32Float => Self::Rg32Float,
            naga::StorageFormat::Rgba32Uint => Self::Rgba32Uint,
            naga::StorageFormat::Rgba32Sint => Self::Rgba32Sint,
            naga::StorageFormat::R16Uint => Self::R16Uint,
            naga::StorageFormat::Rg16Uint => Self::Rg16Uint,
            naga::StorageFormat::Rgba16Uint => Self::Rgba16Uint,
            naga::StorageFormat::R8Unorm => Self::R8Unorm,
            naga::StorageFormat::R8Snorm => Self::R8Snorm,
            naga::StorageFormat::R8Uint => Self::R8Uint,
            naga::StorageFormat::R8Sint => Self::R8Sint,
            naga::StorageFormat::Rg8Snorm => Self::R16Unorm,
            naga::StorageFormat::Rg8Unorm => Self::Rg16Unorm,
            naga::StorageFormat::Rg8Uint => Self::Rgba16Unorm,
            naga::StorageFormat::Rg8Sint => Self::R16Snorm,
            naga::StorageFormat::R32Uint => Self::R32Uint,
            naga::StorageFormat::Rgba8Snorm => Self::Rgba8Snorm,
            naga::StorageFormat::Rgba8Unorm => Self::Rgba8Unorm,
            naga::StorageFormat::Rgba8Uint => Self::Rgba8Uint,
            naga::StorageFormat::Rgba8Sint => Self::Rgba8Sint,
            naga::StorageFormat::Rgb10a2Uint => Self::Rgb10a2Uint,
            naga::StorageFormat::Rgb10a2Unorm => Self::Rgb10a2Unorm,
            naga::StorageFormat::Rg11b10Ufloat => Self::Rg11b10Ufloat,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum StorageAccess {
    Load,
    Store,
    Atomic,
}

impl From<naga::StorageAccess> for StorageAccess {
    fn from(value: naga::StorageAccess) -> Self {
        match value {
            naga::StorageAccess::LOAD => Self::Load,
            naga::StorageAccess::STORE => Self::Store,
            naga::StorageAccess::ATOMIC => Self::Atomic,
            other => unreachable!("{other:?}"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ImageDimension {
    D1,
    D2,
    D3,
    Cube,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ImageClass {
    Sampled {
        kind: ScalarInfo,
        multi: bool,
    },
    Depth {
        multi: bool,
    },
    Storage {
        format: StorageFormat,
        access: StorageAccess,
    },
    External,
}
