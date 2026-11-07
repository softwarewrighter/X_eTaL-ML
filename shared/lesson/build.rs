//! What the teaching shell needs to know of the repository whose pages
//! it builds (X_eTaL-manifolds' ask GM1, X_eTaL-fine-tuning's GF1): the
//! dependent names it at build time, in its `.cargo/config.toml`
//! `[env]` (`relative = true` makes a path absolute from that file):
//!
//! - `LESSON_ROOT`: the dependent's root (default: this repository's).
//!   Its `demos/*/demo.toml` give the lessons' titles, its `libs/` its
//!   own libraries, its git HEAD the build's commit.
//! - `LESSON_SITE`: `name|url|catalog title` (default `X_eTaL-ML|
//!   https://github.com/softwarewrighter/X_eTaL-ML|All demos`).
//! - `LESSON_GLOSSARY`: glossary files under the root, `:`-separated,
//!   read after this crate's `glossary.tsv` (a row of the same id
//!   replaces one there).
//! - `LESSON_PINS`: `Name=pin file` pairs under the root, `:`-separated,
//!   shown in the footer in that order (default: X_eTaL-libraries and
//!   X_eTaL, this repository's pins).
//! - `MICROSCOPE_LIBS` (microscope's, ask FM1): further library roots,
//!   counted as X_eTaL-libraries' here. `LESSON_LIBRARIES_PIN`: the pin
//!   file of those (default `XETAL_LIBRARIES_COMMIT`).
//!
//! It writes `$OUT_DIR/site.rs` (the site, the pins, the build),
//! `lessons.rs` (each demo's slug and title), `origins.rs` (where each
//! library lives: its repository, the commit, its path there, its page
//! in the dependent's cross-reference) and `glossary.rs`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn run(dir: &Path, cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .current_dir(dir)
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

/// An environment value, rerun when it changes.
fn env(name: &str) -> Option<String> {
    println!("cargo:rerun-if-env-changed={name}");
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// The commit a pin file holds, in full ("" when it has none).
fn pinned(file: &Path) -> String {
    println!("cargo:rerun-if-changed={}", file.display());
    std::fs::read_to_string(file).unwrap_or_default().trim().to_string()
}

/// A library directory's sources: Name.xtl (functions), Name.xtlm (macros).
fn sources(dir: &Path) -> Vec<String> {
    let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
    ["xtl", "xtlm"].into_iter().map(|ext| format!("{name}.{ext}")).filter(|f| dir.join("src").join(f).is_file()).collect()
}

/// The libraries under a root (each a directory with src/Name.xtl or
/// .xtlm), or the root itself when it is one library's directory.
fn libraries(root: &Path) -> Vec<PathBuf> {
    if !sources(root).is_empty() {
        return vec![root.to_path_buf()];
    }
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root).map(|d| d.filter_map(Result::ok).map(|e| e.path()).filter(|p| p.is_dir()).collect()).unwrap_or_default();
    dirs.sort();
    dirs.into_iter().filter(|d| !sources(d).is_empty()).collect()
}

/// `p` relative to `base` as text, both made absolute first; `p` as
/// given when it is not under `base`.
fn under(p: &Path, base: &Path) -> String {
    let (p, base) = (absolute(p), absolute(base));
    p.strip_prefix(&base).map(|r| r.to_string_lossy().to_string()).unwrap_or_else(|_| p.to_string_lossy().to_string())
}

/// A path made absolute, `..` taken out (without following links, so a
/// library linked into work/ keeps the path it is listed under).
fn absolute(p: &Path) -> PathBuf {
    let p = if p.is_absolute() { p.to_path_buf() } else { std::env::current_dir().unwrap().join(p) };
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            c => out.push(c),
        }
    }
    out
}

fn main() {
    let here = absolute(Path::new(env!("CARGO_MANIFEST_DIR")));
    let ml_root = absolute(&here.join("../.."));
    let root = env("LESSON_ROOT").map(|r| absolute(&here.join(r))).unwrap_or_else(|| ml_root.clone());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let same = root == ml_root;

    // The site and the build.
    let site = env("LESSON_SITE").unwrap_or_else(|| "X_eTaL-ML|https://github.com/softwarewrighter/X_eTaL-ML|All demos".into());
    let mut f = site.split('|');
    let name = f.next().unwrap_or("X_eTaL-ML").trim().to_string();
    let url = f.next().map(|u| u.trim().to_string()).unwrap_or_else(|| format!("https://github.com/softwarewrighter/{name}"));
    let catalog = f.next().unwrap_or("All demos").trim().to_string();
    let sha = run(&root, "git", &["rev-parse", "--short", "HEAD"]);
    let full = run(&root, "git", &["rev-parse", "HEAD"]);
    let ml_full = run(&ml_root, "git", &["rev-parse", "HEAD"]);
    let pins = env("LESSON_PINS").unwrap_or_else(|| "X_eTaL-libraries=XETAL_LIBRARIES_COMMIT:X_eTaL=XETAL_COMMIT".into());
    let mut code = String::new();
    code += &format!("pub const SITE_NAME: &str = {name:?};\npub const SITE_URL: &str = {url:?};\npub const CATALOG: &str = {catalog:?};\n");
    code += &format!("pub const SHA: &str = {sha:?};\npub const COMMIT: &str = {full:?};\npub const ML_FULL: &str = {ml_full:?};\n");
    code += &format!("pub const HOST: &str = {:?};\npub const TIMESTAMP: &str = {:?};\n", run(&root, "hostname", &["-s"]), run(&root, "date", &["-u", "+%Y%m%dT%H%M"]));
    code += "pub const PINS: &[(&str, &str)] = &[\n";
    for pair in pins.split(':').filter(|p| p.contains('=')) {
        let (n, file) = pair.split_once('=').unwrap();
        code += &format!("    ({:?}, {:?}),\n", n.trim(), pinned(&root.join(file.trim())));
    }
    code += "];\n";
    std::fs::write(out.join("site.rs"), code).unwrap();

    // The lessons: each demo's slug and title.
    let demos = root.join("demos");
    println!("cargo:rerun-if-changed={}", demos.display());
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&demos).map(|d| d.filter_map(Result::ok).map(|e| e.path()).filter(|p| p.is_dir()).collect()).unwrap_or_default();
    dirs.sort();
    let mut code = String::from("pub const LESSONS: &[(&str, &str)] = &[\n");
    for d in dirs {
        let toml = d.join("demo.toml");
        println!("cargo:rerun-if-changed={}", toml.display());
        let text = std::fs::read_to_string(&toml).unwrap_or_default();
        let title = text.lines().find_map(|l| l.strip_prefix("title = \"")).and_then(|t| t.split('"').next());
        if let (Some(slug), Some(title)) = (d.file_name().and_then(|n| n.to_str()), title) {
            code += &format!("    ({slug:?}, {title:?}),\n");
        }
    }
    code += "];\n";
    std::fs::write(out.join("lessons.rs"), code).unwrap();

    // The origins: the dependent's own libraries, X_eTaL-ML's (this
    // crate's repository, unless it is the dependent), then
    // X_eTaL-libraries' (MICROSCOPE_LIBS, and this repository's work/libs).
    let libraries_commit = pinned(&root.join(env("LESSON_LIBRARIES_PIN").unwrap_or_else(|| "XETAL_LIBRARIES_COMMIT".into())));
    let own_commit = if same { ml_full.clone() } else { full.clone() };
    let mut roots: Vec<(PathBuf, String, String)> = vec![(root.join("libs"), name.clone(), own_commit)];
    if !same {
        roots.push((ml_root.join("libs"), "X_eTaL-ML".into(), ml_full.clone()));
    }
    for r in env("MICROSCOPE_LIBS").unwrap_or_default().split(':').filter(|r| !r.trim().is_empty()) {
        roots.push((here.join(r.trim()), "X_eTaL-libraries".into(), libraries_commit.clone()));
    }
    roots.push((ml_root.join("work/libs"), "X_eTaL-libraries".into(), libraries_commit.clone()));
    let mut code = String::from("pub const ORIGINS: &[(&str, &str, &str, &str, &str)] = &[\n");
    let mut seen: Vec<String> = Vec::new();
    for (dir, repo, commit) in roots.iter().rev() {
        println!("cargo:rerun-if-changed={}", dir.display());
        for lib in libraries(dir) {
            let lname = lib.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
            for file in sources(&lib) {
                // A later root wins a name both have, as microscope embeds them.
                if seen.contains(&file) {
                    continue;
                }
                seen.push(file.clone());
                let path = format!("libs/{lname}/src/{file}");
                let page = format!("{}.html", under(&lib.join("src").join(&file), &root).replace('/', "-"));
                code += &format!("    ({file:?}, {repo:?}, {commit:?}, {path:?}, {page:?}),\n");
            }
        }
    }
    code += "];\n";
    std::fs::write(out.join("origins.rs"), code).unwrap();

    // The glossary: this crate's base, then the dependent's files.
    let mut text = std::fs::read_to_string(here.join("glossary.tsv")).unwrap();
    println!("cargo:rerun-if-changed={}", here.join("glossary.tsv").display());
    for file in env("LESSON_GLOSSARY").unwrap_or_default().split(':').filter(|f| !f.trim().is_empty()) {
        let p = root.join(file.trim());
        println!("cargo:rerun-if-changed={}", p.display());
        text += "\n";
        text += &std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("LESSON_GLOSSARY: {}: {e}", p.display()));
    }
    std::fs::write(out.join("glossary.rs"), format!("pub const GLOSSARY: &str = {text:?};\n")).unwrap();
    println!("cargo:rerun-if-changed={}", root.join(".git/logs/HEAD").display());
}
