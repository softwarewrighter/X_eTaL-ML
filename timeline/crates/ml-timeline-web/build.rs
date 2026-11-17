//! Gathers the demos' descriptions, `demos/<slug>/demo.toml`, into one
//! file the page builds in (`demos.txt` in `OUT_DIR`, each separated
//! by a NUL), so a demo event's title, words, picture and link come
//! from its demo and are never written twice.

use std::fs;
use std::path::Path;

fn main() {
    let demos = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../demos");
    println!("cargo::rerun-if-changed={}", demos.display());
    let mut found: Vec<_> = fs::read_dir(&demos)
        .expect("the demos")
        .filter_map(Result::ok)
        .map(|d| d.path().join("demo.toml"))
        .filter(|f| f.is_file())
        .collect();
    found.sort();
    let mut all = String::new();
    for f in &found {
        println!("cargo::rerun-if-changed={}", f.display());
        all.push_str(&fs::read_to_string(f).expect("a demo.toml"));
        all.push('\0');
    }
    let out = Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR")).join("demos.txt");
    fs::write(out, all).expect("demos.txt written");
}
