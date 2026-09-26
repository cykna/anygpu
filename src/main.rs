use crate::shaders::{ShaderMetadata, types::TypeInfo};

mod shaders;

pub fn compile_wgsl(
    wgsl_source: &str,
) -> color_eyre::Result<(naga::Module, naga::valid::ModuleInfo)> {
    let module = naga::front::wgsl::parse_str(wgsl_source)?;
    let module_info: naga::valid::ModuleInfo = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .subgroup_stages(naga::valid::ShaderStages::all())
    .subgroup_operations(naga::valid::SubgroupOperationSet::all())
    .validate(&module)?;

    Ok((module, module_info))
}

fn main() -> color_eyre::Result<()> {
    let wgsl_source = include_str!("../examples/init.wgsl");
    let module = naga::front::wgsl::parse_str(wgsl_source)?;
    let shader = ShaderMetadata::new(&module)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&shader.generate_bindings()).unwrap()
    );

    Ok(())
}
