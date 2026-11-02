use std::{
    ffi::OsStr,
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
};

use clap::Parser;

use anygpu::{ShaderMetadata, compile_wesl_shader, link_modules};

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
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Write the generated JSON schema to this file instead of stdout
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,
    #[arg(long)]
    validate: bool,
    /// Compile WESL roots and link their reflected types
    #[arg(long)]
    wesl: bool,
}

fn compile_entry(args: &Args, entry: &Path) -> color_eyre::Result<String> {
    let wgsl_source = std::fs::read_to_string(entry)?;
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
    let output_content = serde_json::to_string_pretty(&shader.generate_bindings()?)?;
    Ok(output_content)
}

fn compile_wesl_entry(base: &Path, entry: &Path) -> color_eyre::Result<anygpu::ReflectedModule> {
    let relative = match entry.strip_prefix(base) {
        Ok(relative) => relative.to_path_buf(),
        Err(_) => PathBuf::from(
            entry
                .file_name()
                .ok_or_else(|| color_eyre::eyre::eyre!("WESL root path has no file name"))?,
        ),
    };
    let module_id = module_id(&relative)?;
    compile_wesl_shader(base, &relative, module_id)
}

fn module_id(path: &Path) -> color_eyre::Result<String> {
    use color_eyre::eyre::eyre;

    let path = path.with_extension("");
    let mut components = Vec::new();
    for component in path.components() {
        let std::path::Component::Normal(name) = component else {
            return Err(eyre!(
                "WESL root path must be relative to its source directory"
            ));
        };
        components.push(
            name.to_str()
                .ok_or_else(|| eyre!("WESL module path is not valid UTF-8"))?
                .to_owned(),
        );
    }
    if components.is_empty() {
        return Err(eyre!("WESL root path must name a module"));
    }
    Ok(format!("package::{}", components.join("::")))
}

fn compile_wesl_input(args: &Args) -> color_eyre::Result<()> {
    let mut modules = if args.input.is_dir() {
        let mut roots = std::fs::read_dir(&args.input)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<Vec<_>, _>>()?;
        roots.retain(|path| path.extension() == Some(OsStr::new("wesl")));
        roots.sort();
        roots
            .iter()
            .map(|root| compile_wesl_entry(&args.input, root))
            .collect::<color_eyre::Result<Vec<_>>>()?
    } else {
        let base = args
            .input
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        vec![compile_wesl_entry(base, &args.input)?]
    };
    if args.input.is_dir() {
        modules.retain(|module| {
            module.bindings.pipelines.as_ref().is_some_and(|pipeline| {
                pipeline.vertex.is_some()
                    || pipeline.fragment.is_some()
                    || pipeline.compute.is_some()
            })
        });
        if modules.is_empty() {
            color_eyre::eyre::bail!("no WESL shader roots with entry points were found");
        }
    }
    let linked = link_modules(modules)?;
    let json = serde_json::to_string_pretty(&linked)?;

    if args.input.is_dir() {
        let output = args.output.as_ref().unwrap_or(&args.input);
        let output_file = if output.extension() == Some(OsStr::new("json")) {
            output.clone()
        } else {
            std::fs::create_dir_all(output)?;
            output.join("linked.json")
        };
        std::fs::write(output_file, json)?;
    } else if let Some(output) = &args.output {
        std::fs::write(output, json)?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let args = Args::parse();
    if args.wesl {
        return compile_wesl_input(&args);
    }
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
