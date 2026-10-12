//! Embed every library of ../libs: its name, summary, recommended alias,
//! source, reference page (rendered to HTML), demos and tests' types,
//! as a generated catalog (OUT_DIR/catalog.rs); and the libraries of
//! X_eTaL-libraries this repository uses (../work/libs), source only,
//! for the store (ours import them). Adapted from X_eTaL-libraries'
//! site (MIT, the same author).

#[path = "src/render.rs"]
mod render;

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

const REPO: &str = "https://github.com/softwarewrighter/X_eTaL-ML/blob/main";

fn main() {
    let libs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../libs");
    println!("cargo:rerun-if-changed={}", libs.display());
    let mut names: Vec<PathBuf> = fs::read_dir(&libs)
        .expect("libs/")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("src").is_dir())
        .collect();
    names.sort();
    let mut out = String::from("pub static LIBRARIES: &[Library] = &[\n");
    for dir in names {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        for sub in ["src", "docs", "demos", "tests"] {
            println!("cargo:rerun-if-changed={}", dir.join(sub).display());
        }
        let src = fs::read_to_string(dir.join("src").join(format!("{name}.xtl"))).unwrap_or_default();
        let macros = fs::read_to_string(dir.join("src").join(format!("{name}.xtlm"))).unwrap_or_default();
        // A macro-only library (Tags) has its header in its macro library.
        let head = if src.is_empty() { &macros } else { &src };
        // The header's first line and its continuation, up to the import line.
        let summary = head
            .lines()
            .take_while(|l| !l.contains("u_se<"))
            .map(|l| l.trim_start_matches('#').trim())
            .collect::<Vec<_>>()
            .join(" ");
        let summary = summary.trim_start_matches(&format!("{name}: ")).to_string();
        let alias = head
            .find("\" u_se<")
            .and_then(|end| head[..end].rfind('"').map(|start| head[start + 1..end].to_string()))
            .unwrap_or_default();
        let docs = fs::read_to_string(dir.join("docs/README.md")).unwrap_or_default();
        let docs = html(&name, &docs);
        let types = fs::read_to_string(dir.join("tests/types.out")).unwrap_or_default();
        let mut demos: Vec<PathBuf> = fs::read_dir(dir.join("demos"))
            .map(|d| d.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "xtl")).collect())
            .unwrap_or_default();
        demos.sort();
        // Data files beside the demos (a demo builds them in with i_nclude<).
        let mut data: Vec<PathBuf> = fs::read_dir(dir.join("demos"))
            .map(|d| d.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "csv")).collect())
            .unwrap_or_default();
        data.sort();
        let data: Vec<(String, String)> = data
            .iter()
            .map(|p| (p.file_name().unwrap().to_string_lossy().to_string(), fs::read_to_string(p).unwrap()))
            .collect();
        let _ = write!(
            out,
            "  Library {{ name: {name:?}, summary: {summary:?}, alias: {alias:?}, source: {src:?}, macros: {macros:?}, data: &{data:?}, docs: {docs:?}, types: {types:?}, demos: &["
        );
        for d in demos {
            let stem = d.file_stem().unwrap().to_string_lossy().to_string();
            let text = fs::read_to_string(&d).unwrap();
            let expected = fs::read_to_string(dir.join(format!("tests/demo-{stem}.out"))).unwrap_or_default();
            let _ = write!(out, "Demo {{ name: {stem:?}, source: {text:?}, expected: {expected:?} }}, ");
        }
        out.push_str("] },\n");
    }
    out.push_str("];\n");
    // X_eTaL-libraries' libraries this repository uses, for the store.
    let used = Path::new(env!("CARGO_MANIFEST_DIR")).join("../work/libs");
    println!("cargo:rerun-if-changed={}", used.display());
    let mut more: Vec<PathBuf> = fs::read_dir(&used).map(|d| d.flatten().map(|e| e.path()).collect()).unwrap_or_default();
    more.sort();
    out.push_str("pub static USED: &[(&str, &str)] = &[\n");
    for dir in more {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        for ext in ["xtl", "xtlm"] {
            if let Ok(text) = fs::read_to_string(dir.join("src").join(format!("{name}.{ext}"))) {
                let _ = writeln!(out, "  ({:?}, {text:?}),", format!("{name}.{ext}"));
            }
        }
    }
    out.push_str("];\n");
    let dest = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("catalog.rs");
    fs::write(dest, out).unwrap();
}

/// A library's page as HTML, its X_eTaL in rendered form: session
/// examples (an expression indented six spaces, its result below) have
/// each expression rendered and results as printed; other blocks of
/// X_eTaL and inline X_eTaL code are rendered; shell blocks, paths,
/// commands and types stay as typed. Links to its demos go to the demo
/// in this site; other relative links go to the file in the repository.
fn html(name: &str, md: &str) -> String {
    use pulldown_cmark::{CodeBlockKind, CowStr, Event, Options, Parser, Tag, TagEnd};
    let mut events = Vec::new();
    let mut block: Option<(String, String)> = None; // (language, text)
    for event in Parser::new_ext(md, Options::ENABLE_TABLES) {
        match (&mut block, event) {
            (None, Event::Start(Tag::CodeBlock(kind))) => {
                let lang = match kind {
                    CodeBlockKind::Fenced(l) => l.to_string(),
                    CodeBlockKind::Indented => String::new(),
                };
                block = Some((lang, String::new()));
            }
            (Some((_, text)), Event::Text(t)) => text.push_str(&t),
            (Some(_), Event::End(TagEnd::CodeBlock)) => {
                let (lang, text) = block.take().unwrap();
                events.push(Event::Html(CowStr::from(code_block(&lang, &text))));
            }
            (None, Event::Code(code)) if render::is_xetal(&code) => {
                events.push(Event::Html(CowStr::from(format!(
                    "<code class=\"xtl\">{}</code>",
                    render::decorated(&code)
                ))));
            }
            (None, Event::Start(Tag::Link { link_type, dest_url, title, id })) => {
                let url = dest_url.to_string();
                let url = if let Some(demo) = url.strip_prefix("../demos/").and_then(|d| d.strip_suffix(".xtl")) {
                    format!("#{name}/{demo}")
                } else if url.contains("://") || url.starts_with('#') {
                    url
                } else {
                    format!("{REPO}/libs/{name}/docs/{url}")
                };
                events.push(Event::Start(Tag::Link { link_type, dest_url: CowStr::from(url), title, id }));
            }
            (None, e) => events.push(e),
            (Some(_), _) => {}
        }
    }
    let mut html = String::new();
    pulldown_cmark::html::push_html(&mut html, events.into_iter());
    html
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// A fenced block: shell stays as typed; a session has its expressions
/// rendered; any other block is X_eTaL, rendered.
fn code_block(lang: &str, text: &str) -> String {
    let body = if lang == "bash" || lang == "sh" {
        escape(text)
    } else if text.lines().any(|l| l.starts_with("      ")) {
        text.lines()
            .map(|l| match l.strip_prefix("      ") {
                Some(expr) => format!("      {}", render::decorated(expr)),
                None => format!("<span class=\"result\">{}</span>", escape(l)),
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        render::decorated(text)
    };
    let class = if lang == "bash" || lang == "sh" { "typed" } else { "xtl" };
    format!("<pre class=\"{class}\">{body}</pre>\n")
}
