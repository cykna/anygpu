use std::io::{self, Write};
use std::path::{Path, PathBuf};

use clap::Parser;
use color_eyre::{Result, eyre::WrapErr};

use anygpu_gen_ts::codegen::generate_typescript;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Schema JSON file to read, defaults to stdin
    #[arg(value_name = "SCHEMA")]
    input: PathBuf,

    /// Write the generated TypeScript to this file instead of stdout
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let cli = Cli::parse();
    let input = read_input(&cli.input)?;
    let schema: anygpu::ShaderBindings = anygpu::serde_json::from_str(&input)?;

    let typescript = generate_typescript(&schema)?;
    write_output(cli.output.as_deref(), &typescript)
}

fn read_input(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).wrap_err_with(|| format!("failed to read `{}`", path.display()))
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
