use std::ffi::OsStr;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use clap::Parser;
use color_eyre::{Result, eyre::WrapErr};

use anygpu_ts::codegen::generate_batch;
use anygpu_ts::codegen::generate_typescript;

const JSON: &str = "json";

/// The name a schema read from stdin is known by. A file always has a name of its
/// own, so this is only reached when there is nothing to take one from.
const UNNAMED: &str = "shader";

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Schema JSON file, or a folder of them, defaults to stdin
    #[arg(value_name = "SCHEMA")]
    input: Option<PathBuf>,

    /// Write to this file, or to this folder when the input is a folder
    #[arg(short, long, value_name = "PATH")]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let cli = Cli::parse();
    // A folder is a batch: one TypeScript file per schema inside it, which is the
    // inverse of `anygpu shaders -o temp/`.
    if let Some(input) = cli.input.as_deref().filter(|path| path.is_dir()) {
        return generate_dir(input, cli.output.as_deref());
    }

    let typescript = generate(
        &read_input(cli.input.as_deref())?,
        &shader_name(cli.input.as_deref()),
    )?;
    write_output(cli.output.as_deref(), &typescript)
}

/// Generates one TypeScript file per schema in `dir`.
///
/// Like `anygpu`, a folder is not searched recursively and the results go
/// back into the input folder when no `-o` is given, each schema keeping its
/// own name. A folder is also a batch: the parts of the output that do not
/// depend on any one schema — the support section, and the accessor classes
/// several shaders share — are written once for the whole folder instead of
/// once per shader, and each shader's file imports them.
fn generate_dir(dir: &Path, output: Option<&Path>) -> Result<()> {
    let output_dir = output.unwrap_or(dir);
    create_dir(output_dir)?;

    // Every schema is read before anything is generated, so the batch knows the
    // full set of shared classes before it writes the first file.
    let schemas = schemas_in(dir)?;
    let shaders = schemas
        .iter()
        .map(|path| {
            let schema: anygpu::ShaderBindings = read_input(Some(path))
                .and_then(|source| Ok(anygpu::serde_json::from_str(&source)?))
                .wrap_err_with(|| format!("failed to read `{}`", path.display()))?;
            Ok((shader_name(Some(path)), schema))
        })
        .collect::<Result<Vec<_>>>()?;

    let files = generate_batch(shaders.iter().map(|(name, schema)| (name.as_str(), schema)))?;
    for file in files {
        let destination = output_dir.join(&file.path);
        if let Some(parent) = destination.parent() {
            create_dir(parent)?;
        }
        std::fs::write(&destination, &file.contents)
            .wrap_err_with(|| format!("failed to write `{}`", destination.display()))?;
        println!("{}", destination.display());
    }
    Ok(())
}

fn create_dir(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path).wrap_err_with(|| format!("failed to create `{}`", path.display()))
}

/// The name a schema is known by: the file it came from, without its extension.
///
/// The schema does not record a name of its own, so the file name is what a
/// generated pipeline helper is exported under.
fn shader_name(path: Option<&Path>) -> String {
    path.and_then(Path::file_stem)
        .and_then(|stem| stem.to_str())
        .unwrap_or(UNNAMED)
        .to_string()
}

/// The schemas directly inside `dir`, in a stable order.
fn schemas_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut schemas = Vec::new();
    for entry in
        std::fs::read_dir(dir).wrap_err_with(|| format!("failed to read `{}`", dir.display()))?
    {
        let path = entry?.path();
        if path.extension() == Some(OsStr::new(JSON)) {
            schemas.push(path);
        }
    }
    schemas.sort();
    Ok(schemas)
}

fn read_input(path: Option<&Path>) -> Result<String> {
    if let Some(path) = path {
        std::fs::read_to_string(path)
            .wrap_err_with(|| format!("failed to read `{}`", path.display()))
    } else {
        let mut out = String::new();
        io::stdin().read_to_string(&mut out)?;
        Ok(out)
    }
}

fn generate(input: &str, name: &str) -> Result<String> {
    let schema: anygpu::ShaderBindings = anygpu::serde_json::from_str(input)?;
    generate_typescript(&schema, name)
}

fn write_output(path: Option<&Path>, typescript: &str) -> Result<()> {
    match path {
        Some(path) => std::fs::write(path, typescript)
            .wrap_err_with(|| format!("failed to write `{}`", path.display())),
        None => io::stdout()
            .write_all(typescript.as_bytes())
            .wrap_err("failed to write stdout"),
    }
}
