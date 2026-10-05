//! Build provenance for the footer (as the X_eTaL live demo shows it):
//! the build host, this repo's short commit, the build time, and the
//! pinned X_eTaL commit (XETAL_COMMIT). Also embeds this
//! repo's libraries (libs/<Name>/src/<Name>.xtl) and those used
//! from X_eTaL-libraries (work/libs) for `libs`, so a page's
//! program can import them with `u_se<`.

use std::process::Command;

fn run(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn pinned() -> String {
    let text = std::fs::read_to_string("../../XETAL_COMMIT").unwrap_or_default();
    match text.trim() {
        "" => "unknown".into(),
        sha => sha.chars().take(7).collect(),
    }
}

/// `$OUT_DIR/libs.rs`: every libs/<Name>/src/<Name>.xtl as (file, source).
fn libraries() {
    let here = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    // This repo's libraries, then those used from X_eTaL-libraries.
    let roots = [here.join("../../libs"), here.join("../../work/libs")];
    let mut found: Vec<(String, std::path::PathBuf)> = roots
        .iter()
        .inspect(|r| println!("cargo:rerun-if-changed={}", r.display()))
        .flat_map(|r| std::fs::read_dir(r).map(|d| d.filter_map(Result::ok).map(|e| e.path()).collect::<Vec<_>>()).unwrap_or_default())
        .filter_map(|dir| {
            let name = dir.file_name()?.to_str()?.to_string();
            let src = dir.join("src").join(format!("{name}.xtl"));
            src.is_file().then(|| (format!("{name}.xtl"), src))
        })
        .collect();
    found.sort();
    let mut code = String::from("pub const LIBRARIES: &[(&str, &str)] = &[\n");
    for (file, src) in &found {
        println!("cargo:rerun-if-changed={}", src.display());
        code += &format!("    ({file:?}, include_str!({:?})),\n", src.canonicalize().unwrap());
    }
    code += "];\n";
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("libs.rs");
    std::fs::write(out, code).unwrap();
}

fn main() {
    libraries();
    println!("cargo:rustc-env=BUILD_SHA={}", run("git", &["rev-parse", "--short", "HEAD"]));
    println!("cargo:rustc-env=BUILD_HOST={}", run("hostname", &["-s"]));
    println!("cargo:rustc-env=BUILD_TIMESTAMP={}", run("date", &["-u", "+%Y%m%dT%H%M%S"]));
    println!("cargo:rustc-env=XETAL_SHA={}", pinned());
    println!("cargo:rerun-if-changed=../../XETAL_COMMIT");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
}
