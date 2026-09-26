use std::{
    ffi::OsStr,
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
};

use clap::Parser;

use crate::shaders::ShaderMetadata;

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

#[derive(clap::Parser)]
#[command(version, about)]
pub struct Args {
    #[arg(value_name = "SCHEMA")]
    input: PathBuf,

    /// Write the generated TypeScript to this file instead of stdout
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,
    #[arg(long)]
    validate: bool,
}

fn compile_entry(args: &Args, entry: &Path) -> color_eyre::Result<String> {
    let wgsl_source = std::fs::read_to_string(&entry)?;
    let module = naga::front::wgsl::parse_str(&wgsl_source)?;
    if args.validate {
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .subgroup_stages(naga::valid::ShaderStages::all())
        .subgroup_operations(naga::valid::SubgroupOperationSet::all())
        .validate(&module)?;
    }
    let shader = ShaderMetadata::new(&module)?;
    let output_content = serde_json::to_string_pretty(&shader.generate_bindings())?;
    Ok(output_content)
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let args = Args::parse();
    if let Ok(entry) = args.input.read_dir() {
        let output_dir = args.output.as_ref().unwrap_or(&args.input);
        std::fs::create_dir_all(output_dir)?;
        for entry in entry {
            let entry = entry?;
            if entry.path().extension() != Some(OsStr::from_bytes(b"wgsl")) {
                continue;
            }
            let output_content = compile_entry(&args, &entry.path())?;
            let output_dir = output_dir
                .join(entry.path().file_name().unwrap())
                .with_extension("json");

            std::fs::write(output_dir, output_content)?;
        }
    } else {
        let output_content = compile_entry(&args, &args.input)?;
        if let Some(output) = args.output {
            std::fs::write(output, output_content)?;
        } else {
            println!("{}", output_content);
        }
    }

    Ok(())
}
