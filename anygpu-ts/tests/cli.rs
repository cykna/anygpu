//! Exercises the command line, in particular the folder mode that mirrors
//! `anygpu shaders -o temp/`.

use std::collections::BTreeMap;
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
fn a_folder_writes_the_shared_pieces_once() {
    let schemas = scratch("cli-shared-schemas");
    let bindings = scratch("cli-shared-bindings");
    // Both schemas use `vec4<f32>`, so both want the same `Vector4f32`.
    std::fs::write(schemas.join("camera.json"), schema("Camera")).unwrap();
    std::fs::write(schemas.join("lights.json"), schema("Lights")).unwrap();

    let output = run(&[schemas.to_str().unwrap(), "-o", bindings.to_str().unwrap()]);
    assert_success(&output);

    // The support section does not depend on a schema, so it is written once for
    // the folder rather than once per shader.
    let support = std::fs::read_to_string(bindings.join("support.ts")).expect("support.ts");
    assert!(
        support.contains("export class PipelineHelper {"),
        "{support}"
    );

    // The classes that several shaders share are written once, under `builtins/`,
    // and no shader file repeats them.
    let vector = std::fs::read_to_string(bindings.join("builtins/vector4f32.ts"))
        .expect("builtins/vector4f32.ts");
    assert!(vector.contains("export class Vector4f32 {"), "{vector}");

    for name in ["camera.ts", "lights.ts"] {
        let typescript = generated_typescript(&bindings.join(name));
        assert!(
            !typescript.contains("export class Vector4f32"),
            "{name} should import `Vector4f32`, not redeclare it: {typescript}"
        );
        assert!(
            !typescript.contains("export class PipelineHelper"),
            "{name} should import the support section, not redeclare it: {typescript}"
        );
    }
    let camera = generated_typescript(&bindings.join("camera.ts"));
    assert!(
        camera.contains("import { Vector4f32 } from \"./builtins/vector4f32\";"),
        "{camera}"
    );
    assert!(
        camera.contains("import { PipelineHelper } from \"./support\";"),
        "{camera}"
    );
}

#[test]
fn writing_a_folder_twice_changes_nothing() {
    // A second run over the same folder regenerates the same files rather than
    // accumulating anything on top of the first.
    let schemas = scratch("cli-idempotent-schemas");
    let bindings = scratch("cli-idempotent-bindings");
    std::fs::write(schemas.join("camera.json"), schema("Camera")).unwrap();

    let first = run(&[schemas.to_str().unwrap(), "-o", bindings.to_str().unwrap()]);
    assert_success(&first);
    let before = tree(&bindings);
    assert!(before.contains_key("support.ts"));

    let second = run(&[schemas.to_str().unwrap(), "-o", bindings.to_str().unwrap()]);
    assert_success(&second);
    assert_eq!(before, tree(&bindings));
}

#[test]
fn stale_output_from_an_earlier_run_is_overwritten() {
    // A reused output folder is not a fresh one, and the batch does not ask
    // before writing over what is there.
    let schemas = scratch("cli-stale-schemas");
    let bindings = scratch("cli-stale-bindings");
    std::fs::write(schemas.join("camera.json"), schema("Camera")).unwrap();

    let run_once = run(&[schemas.to_str().unwrap(), "-o", bindings.to_str().unwrap()]);
    assert_success(&run_once);
    let before = tree(&bindings);

    std::fs::write(bindings.join("support.ts"), "stale").unwrap();
    std::fs::write(bindings.join("builtins/vector4f32.ts"), "stale").unwrap();
    std::fs::write(bindings.join("camera.ts"), "stale").unwrap();
    // A file this batch does not write is neither regenerated nor removed.
    std::fs::write(bindings.join("leftover.ts"), "stale").unwrap();

    let run_again = run(&[schemas.to_str().unwrap(), "-o", bindings.to_str().unwrap()]);
    assert_success(&run_again);
    let after = tree(&bindings);

    for (path, contents) in &before {
        assert_eq!(
            after.get(path),
            Some(contents),
            "`{path}` should have been regenerated"
        );
    }
    assert_eq!(after.get("leftover.ts").map(String::as_str), Some("stale"));
}

/// Every file under `dir`, keyed by its path relative to `dir`.
fn tree(dir: &Path) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current).expect("the output folder should be readable") {
            let path = entry.expect("a directory entry should be readable").path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let relative = path
                .strip_prefix(dir)
                .expect("a file should be under the output folder")
                .display()
                .to_string();
            out.insert(relative, std::fs::read_to_string(&path).unwrap_or_default());
        }
    }
    out
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
