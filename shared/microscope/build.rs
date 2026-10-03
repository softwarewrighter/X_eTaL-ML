//! Build provenance for the footer (as the X_eTaL live demo shows it):
//! the build host, this repo's short commit, the build time, and the
//! vendored X_eTaL commit (vendor/xetal/VENDORED). Also embeds this
//! repo's libraries (libs/<Name>/src/<Name>.xtl) for `libs`, so a page's
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

fn vendored() -> String {
    let text = std::fs::read_to_string("../../vendor/xetal/VENDORED").unwrap_or_default();
    text.lines()
        .find_map(|l| l.strip_prefix("commit = \""))
        .map(|c| c.chars().take(7).collect())
        .unwrap_or_else(|| "unknown".into())
}

/// `$OUT_DIR/libs.rs`: every libs/<Name>/src/<Name>.xtl as (file, source).
fn libraries() {
    let libs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../libs");
    println!("cargo:rerun-if-changed={}", libs.display());
    let mut found: Vec<(String, std::path::PathBuf)> = std::fs::read_dir(&libs)
        .map(|d| d.filter_map(Result::ok).map(|e| e.path()).collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
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
    println!("cargo:rustc-env=XETAL_SHA={}", vendored());
    println!("cargo:rerun-if-changed=../../vendor/xetal/VENDORED");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
}
