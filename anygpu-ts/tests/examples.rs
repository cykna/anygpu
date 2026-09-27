//! Runs the generator over every example input in the repository and reports
//! one line per file.
//!
//! `.wgsl` files go through the whole pipeline (naga parses them, the schema is
//! derived, then TypeScript is generated); `.json` files are already schemas and
//! go straight to the generator. Any other file in those folders is ignored.
//!
//! Run with `cargo test -p anygpu-gen-ts --test examples -- --nocapture` to see
//! the per-file lines; without `--nocapture` the harness swallows the output of
//! passing tests.

use std::path::{Path, PathBuf};

use anygpu::{ShaderBindings, ShaderMetadata, naga, serde_json};
use anygpu_gen_ts::codegen::generate_typescript;
use color_eyre::eyre::{Result, eyre};

/// Folders, relative to the workspace root, holding generator inputs.
const INPUT_DIRS: &[&str] = &["examples"];

const WGSL: &str = "wgsl";
const JSON: &str = "json";

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("anygpu-ts should live inside the workspace")
        .to_path_buf()
}

fn is_input(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some(WGSL | JSON)
    )
}

/// Reads one input and generates TypeScript from it.
fn generate(path: &Path) -> Result<String> {
    let source = std::fs::read_to_string(path)?;
    let schema: ShaderBindings = match path.extension().and_then(|ext| ext.to_str()) {
        Some(WGSL) => {
            let module = naga::front::wgsl::parse_str(&source)?;
            ShaderMetadata::new(&module)?.generate_bindings()
        }
        Some(JSON) => serde_json::from_str(&source)?,
        other => return Err(eyre!("`{other:?}` is not a generator input")),
    };

    // A generated pipeline helper is named after the file it was generated from,
    // so the name has to travel with the input rather than with the schema.
    let name = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .expect("an input file should have a name");
    let typescript = generate_typescript(&schema, name)?;
    if typescript.trim().is_empty() {
        return Err(eyre!("generated no TypeScript"));
    }
    Ok(typescript)
}

fn inputs() -> Vec<PathBuf> {
    let root = workspace_root();
    let mut paths: Vec<PathBuf> = INPUT_DIRS
        .iter()
        .flat_map(|dir| {
            let dir = root.join(dir);
            let entries = std::fs::read_dir(&dir)
                .unwrap_or_else(|err| panic!("`{}` should be readable: {err}", dir.display()));
            entries
                .map(|entry| entry.expect("directory entry should be readable").path())
                .filter(|path| path.is_file() && is_input(path))
                .collect::<Vec<_>>()
        })
        .collect();
    paths.sort();
    paths
}

#[test]
fn every_example_generates_typescript() {
    let root = workspace_root();
    let paths = inputs();
    assert!(
        !paths.is_empty(),
        "no `.wgsl` or `.json` inputs found in {INPUT_DIRS:?}"
    );

    let mut failures = Vec::new();
    for path in &paths {
        let name = path
            .strip_prefix(&root)
            .unwrap_or(path.as_path())
            .display()
            .to_string();

        match generate(path) {
            Ok(_) => println!("SUCCESS: {name}"),
            Err(err) => {
                let reason = format!("{err}");
                println!("ERROR: {name}; ERROR REASON: {reason}");
                failures.push(format!("{name}: {reason}"));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} example(s) failed to generate:\n  - {}",
        failures.len(),
        paths.len(),
        failures.join("\n  - ")
    );
}
