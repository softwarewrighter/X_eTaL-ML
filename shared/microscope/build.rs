//! Build provenance for the footer (as the X_eTaL live demo shows it):
//! the build host, this repo's short commit, the build time, and the
//! pinned X_eTaL commit (XETAL_COMMIT). Also embeds this
//! repo's libraries (libs/<Name>/src/<Name>.xtl) and those used
//! from X_eTaL-libraries (work/libs) for `libs`, so a page's
//! program can import them with `u_se<`; then those under the roots a
//! dependent names in MICROSCOPE_LIBS (X_eTaL-fine-tuning's ask FM1):
//! directories separated by `:`, each a root of libraries
//! (<Name>/src/<Name>.xtl) or one library's own directory, absolute (or
//! made so by `[env] MICROSCOPE_LIBS = { value = "...", relative = true }`
//! in the dependent's .cargo/config.toml). Later roots win a name both
//! have.

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

/// `$OUT_DIR/libs.rs`: every libs/<Name>/src/<Name>.xtl and .xtlm as (file, source).
fn libraries() {
    let here = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    // This repo's libraries, then those used from X_eTaL-libraries.
    // Then the dependent's roots (FM1).
    println!("cargo:rerun-if-env-changed=MICROSCOPE_LIBS");
    let extra = std::env::var("MICROSCOPE_LIBS").unwrap_or_default();
    let roots: Vec<std::path::PathBuf> = [here.join("../../libs"), here.join("../../work/libs")]
        .into_iter()
        .chain(extra.split(':').filter(|r| !r.trim().is_empty()).map(|r| here.join(r.trim())))
        .collect();
    // A library's own directory counts as a root holding just it.
    let is_library = |d: &std::path::Path| {
        let name = d.file_name().and_then(|n| n.to_str()).unwrap_or("");
        ["xtl", "xtlm"].iter().any(|ext| d.join("src").join(format!("{name}.{ext}")).is_file())
    };
    let found: Vec<(String, std::path::PathBuf)> = roots
        .iter()
        .inspect(|r| println!("cargo:rerun-if-changed={}", r.display()))
        .flat_map(|r| match is_library(r) {
            true => vec![r.clone()],
            false => std::fs::read_dir(r).map(|d| d.filter_map(Result::ok).map(|e| e.path()).collect::<Vec<_>>()).unwrap_or_default(),
        })
        .flat_map(|dir| {
            // A library's functions (Name.xtl) and its macros (Name.xtlm).
            let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
            ["xtl", "xtlm"].into_iter().filter_map(move |ext| {
                let src = dir.join("src").join(format!("{name}.{ext}"));
                src.is_file().then(|| (format!("{name}.{ext}"), src))
            })
        })
        .collect();
    // Each name once: the last root's (a dependent's library replaces
    // one of the same name), sorted by name.
    let mut by_name = std::collections::BTreeMap::new();
    for (file, src) in found {
        by_name.insert(file, src);
    }
    let mut code = String::from("pub const LIBRARIES: &[(&str, &str)] = &[\n");
    for (file, src) in &by_name {
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
