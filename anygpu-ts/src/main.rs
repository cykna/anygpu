use std::ffi::OsStr;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use clap::Parser;
use color_eyre::{Result, eyre::WrapErr};

use anygpu_gen_ts::codegen::generate_typescript;

const JSON: &str = "json";
const TS: &str = "ts";

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
/// own name.
fn generate_dir(dir: &Path, output: Option<&Path>) -> Result<()> {
    let output_dir = output.unwrap_or(dir);
    std::fs::create_dir_all(output_dir)
        .wrap_err_with(|| format!("failed to create `{}`", output_dir.display()))?;

    for schema in schemas_in(dir)? {
        let typescript = generate_file(&schema)?;
        let name = schema
            .file_name()
            .expect("a directory entry should have a file name");
        let destination = output_dir.join(name).with_extension(TS);
        std::fs::write(&destination, typescript)
            .wrap_err_with(|| format!("failed to write `{}`", destination.display()))?;
        println!("{} -> {}", schema.display(), destination.display());
    }
    Ok(())
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

/// Generates from a named file, so a failure in a batch says which one.
fn generate_file(path: &Path) -> Result<String> {
    let name = shader_name(Some(path));
    generate(&read_input(Some(path))?, &name)
        .wrap_err_with(|| format!("failed to generate from `{}`", path.display()))
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
