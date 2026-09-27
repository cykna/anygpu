//! Exercises the command line, in particular the folder mode that mirrors
//! `anygpu shaders -o temp/`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use anygpu::{ShaderMetadata, naga, serde_json};

const BIN: &str = env!("CARGO_BIN_EXE_anygpu-gen-ts");

/// An empty scratch folder, private to one test.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the scratch folder should be creatable");
    dir
}

/// The schema `anygpu` would derive for a one-field uniform of the given name.
fn schema(name: &str) -> String {
    let wgsl = format!(
        "struct {name} {{ position: vec4<f32> }}\n\
         @group(0) @binding(0) var<uniform> value: {name};\n"
    );
    let module = naga::front::wgsl::parse_str(&wgsl).expect("the shader should parse");
    let bindings = ShaderMetadata::new(&module)
        .expect("the shader should expose bindings")
        .generate_bindings();
    serde_json::to_string_pretty(&bindings).expect("the schema should serialise")
}

fn run(args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .output()
        .expect("anygpu-gen-ts should run")
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "{output:?}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn generated_typescript(path: &Path) -> String {
    assert!(
        path.is_file(),
        "{} should have been written",
        path.display()
    );
    let typescript = std::fs::read_to_string(path).expect("the output should be readable");
    assert!(
        typescript.contains("export class"),
        "{}: {typescript}",
        path.display()
    );
    typescript
}

#[test]
fn a_folder_produces_one_typescript_file_per_schema() {
    let schemas = scratch("cli-schemas");
    let bindings = scratch("cli-bindings");
    std::fs::write(schemas.join("camera.json"), schema("Camera")).unwrap();
    std::fs::write(schemas.join("lights.json"), schema("Lights")).unwrap();
    std::fs::write(schemas.join("notes.txt"), "not a schema").unwrap();

    let output = run(&[schemas.to_str().unwrap(), "-o", bindings.to_str().unwrap()]);
    assert_success(&output);

    // Each schema keeps its own name, and the stray file is left alone. The
    // pipeline helper is named after the file too, since a schema has nowhere to
    // record a name of its own.
    let camera = generated_typescript(&bindings.join("camera.ts"));
    assert!(camera.contains("Camera"));
    assert!(camera.contains("CameraPipelineHelper"), "{camera}");
    let lights = generated_typescript(&bindings.join("lights.ts"));
    assert!(lights.contains("Lights"));
    assert!(lights.contains("LightsPipelineHelper"), "{lights}");
    assert!(!bindings.join("notes.ts").exists());
}

#[test]
fn a_folder_without_output_writes_beside_its_schemas() {
    let schemas = scratch("cli-in-place");
    std::fs::write(schemas.join("camera.json"), schema("Camera")).unwrap();

    let output = run(&[schemas.to_str().unwrap()]);
    assert_success(&output);

    generated_typescript(&schemas.join("camera.ts"));
}

#[test]
fn a_single_schema_still_goes_to_stdout() {
    let dir = scratch("cli-single");
    let path = dir.join("camera.json");
    std::fs::write(&path, schema("Camera")).unwrap();

    let output = run(&[path.to_str().unwrap()]);
    assert_success(&output);

    let typescript = String::from_utf8_lossy(&output.stdout);
    assert!(typescript.contains("export class"), "{typescript}");
    assert!(typescript.contains("CameraPipelineHelper"), "{typescript}");
}

#[test]
fn a_schema_from_stdin_is_named_after_nothing_in_particular() {
    // Nothing to take a name from, so the helper gets a default one rather than
    // being left unnamed.
    let dir = scratch("cli-stdin");
    let path = dir.join("camera.json");
    std::fs::write(&path, schema("Camera")).unwrap();

    let mut child = Command::new(BIN)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("anygpu-gen-ts should run");
    let source = std::fs::read(&path).expect("the schema should be readable");
    child
        .stdin
        .as_mut()
        .expect("stdin should be piped")
        .write_all(&source)
        .expect("the schema should be written");
    let output = child
        .wait_with_output()
        .expect("anygpu-gen-ts should finish");
    assert_success(&output);

    let typescript = String::from_utf8_lossy(&output.stdout);
    assert!(typescript.contains("ShaderPipelineHelper"), "{typescript}");
}
