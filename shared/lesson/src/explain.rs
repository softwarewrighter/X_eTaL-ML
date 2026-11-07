//! A library function explained from the library's own source at the
//! pin: its doc comment, its definition, the file and line it is on,
//! and where to read more. A page never retypes library code: what it
//! shows is what runs (the user, 2026-11-05: a call shown without its
//! import, its source or its purpose leaves a new reader lost).

use yew::prelude::*;

use crate::code::Linker;

include!(concat!(env!("OUT_DIR"), "/origins.rs"));

/// A library a program imports: `"lo:" u_se< "LoRA"`.
#[derive(Clone, Debug, PartialEq)]
pub struct Import {
    pub alias: String,
    pub library: String,
    /// The import line as the program has it.
    pub line: String,
}

/// The imports of a program, in its order.
pub fn imports(program: &str) -> Vec<Import> {
    program
        .lines()
        .filter_map(|l| {
            let t = l.trim();
            let rest = t.strip_prefix('"')?;
            let (alias, rest) = rest.split_once('"')?;
            let lib = rest.trim().strip_prefix("u_se<")?.trim().strip_prefix('"')?.strip_suffix('"')?;
            Some(Import { alias: alias.to_string(), library: lib.to_string(), line: t.to_string() })
        })
        .collect()
}

/// The library functions a program calls, `(library, name)`, each
/// once, in the order first met; comments are not code.
pub fn calls(program: &str) -> Vec<(String, String)> {
    let imports = imports(program);
    let mut out: Vec<(String, String)> = Vec::new();
    for line in program.lines() {
        let code = line.split('#').next().unwrap_or("");
        for im in &imports {
            let mut rest = code;
            while let Some(i) = rest.find(im.alias.as_str()) {
                let before_ok = rest[..i].chars().next_back().is_none_or(|c| !(c.is_alphanumeric() || c == '_' || c == '"'));
                let after = &rest[i + im.alias.len()..];
                let name: String = after.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
                if before_ok && !name.is_empty() && !out.contains(&(im.library.clone(), name.clone())) {
                    out.push((im.library.clone(), name.clone()));
                }
                rest = after;
            }
        }
    }
    out
}

/// A function as its library defines it.
#[derive(Clone, Debug, PartialEq)]
pub struct Definition {
    pub library: String,
    pub name: String,
    /// The library's own description (its doc comment, without the examples).
    pub doc: String,
    /// The definition, as written in the library.
    pub source: String,
    /// The line of the library's file the definition starts on (from 1).
    pub line: usize,
    /// That line in the repository, at the pinned commit.
    pub source_url: String,
    /// The function in this site's cross-reference.
    pub xref: String,
}

/// A library file's source (`Name.xtl`, its functions, or `Name.xtlm`,
/// its macros), as microscope embedded it (X_eTaL-ML's own, and
/// X_eTaL-libraries' at this repository's pin).
fn file_source(file: &str) -> Option<&'static str> {
    microscope::libs::LIBRARIES.iter().find(|(f, _)| *f == file).map(|(_, s)| *s)
}

/// The source of a library: its functions, or its macros when it has
/// only those.
fn library_source(library: &str) -> Option<&'static str> {
    file_source(&format!("{library}.xtl")).or_else(|| file_source(&format!("{library}.xtlm")))
}

/// `name` of `library` as the library defines it, or None when the
/// library or the name is not found: a function (`l:name := ` in
/// `Name.xtl`), or a macro (`m:name< := ` in `Name.xtlm`).
pub fn definition(library: &str, name: &str) -> Option<Definition> {
    defined(library, name, &format!("{library}.xtl"), &format!("l:{name} := "), &format!("l:{name}"))
        .or_else(|| defined(library, name, &format!("{library}.xtlm"), &format!("m:{name}< := "), &format!("m:{name}<")))
}

/// The definition starting with `start` in the library's `file`, its
/// entry in the cross-reference being `id`.
fn defined(library: &str, name: &str, file: &str, start: &str, id: &str) -> Option<Definition> {
    let text = file_source(file)?;
    let lines: Vec<&str> = text.lines().collect();
    let at = lines.iter().position(|l| l.starts_with(start))?;
    let end = (at + 1..lines.len()).find(|&i| lines[i].trim().is_empty() || lines[i].starts_with("##")).unwrap_or(lines.len());
    let first = (0..at).rev().take_while(|&i| lines[i].starts_with("##") && !lines[i].starts_with("###")).last().unwrap_or(at);
    let doc: Vec<&str> = lines[first..at]
        .iter()
        .map(|l| l.trim_start_matches('#').trim())
        .take_while(|l| !l.starts_with(">>"))
        .collect();
    let (repo, commit, path) = ORIGINS.iter().find(|o| o.0 == file).map(|o| (o.1, o.2, o.3))?;
    let line = at + 1;
    let commit = if commit.is_empty() { "main" } else { commit };
    Some(Definition {
        library: library.to_string(),
        name: name.to_string(),
        doc: doc.join(" "),
        source: lines[at..end].join("\n"),
        line,
        source_url: format!("https://github.com/softwarewrighter/{repo}/blob/{commit}/{path}#L{line}"),
        xref: format!("{}#{}", file_page(file)?, crate::code::anchor(id)),
    })
}

/// A library's page in this site's cross-reference, from a page of a
/// demo (`../doc/...`), or None for a library not on the search path:
/// the page of its functions, or of its macros when it has only those.
pub fn xref_page(library: &str) -> Option<String> {
    file_page(&format!("{library}.xtl")).or_else(|| file_page(&format!("{library}.xtlm")))
}

/// The page of one library file (`Name.xtl` or `Name.xtlm`) in the
/// dependent's cross-reference: its path under the dependent's root,
/// its slashes as dashes (build.rs).
pub fn file_page(file: &str) -> Option<String> {
    ORIGINS.iter().find(|o| o.0 == file).map(|o| format!("../doc/{}", o.4))
}

/// The repository a library lives in, for a sentence: "X_eTaL-ML".
pub fn home(library: &str) -> &'static str {
    let (functions, macros) = (format!("{library}.xtl"), format!("{library}.xtlm"));
    match ORIGINS.iter().find(|o| o.0 == functions || o.0 == macros).map(|o| o.1) {
        Some(repo) if repo != crate::chrome::SITE_NAME => repo,
        _ => "this repository",
    }
}

/// A function's explanation, hidden until opened: the page's own words
/// on what it is for (`purpose`) and how to read it (`reading`), and
/// between them, from the library itself, the import that brings it
/// in, its definition drawn as X_eTaL draws it, its own description,
/// and the links to its source and the cross-reference.
pub fn function(alias: &str, d: &Definition, purpose: Html, reading: Html) -> Html {
    let import = format!("\"{alias}\" u_se< \"{}\"", d.library);
    let within = Linker::within(&d.library, library_source(&d.library).unwrap_or(""));
    let outside = Linker::of(&import);
    html! {
        <details class="aside" data-aside={format!("{alias}{}", d.name)}>
            <summary>{ format!("See the code behind this: the function {} of the {} library, where it comes from, what it does, how it is written", d.name.replace('_', ""), d.library) }</summary>
            <div class="aside-body">
                <h3>{"The code, and what it is for"}</h3>
                {purpose}
                <h3>{"Where it comes from"}</h3>
                <p>{ format!("It is a function of the {} library, which is part of {}. A program opens the library with the one line below, giving it a short name of its choice, here {}. After that, a name drawn with a small raised {} before it is one of that library's functions.", d.library, home(&d.library), alias.trim_end_matches(':'), alias.trim_end_matches(':')) }</p>
                <p>{ outside.code(&import) }</p>
                <h3>{"How it is written"}</h3>
                <p>{ format!("This is the function exactly as the library has it (line {} of its file), not a copy made for this page:", d.line) }</p>
                { within.block(&d.source) }
                {reading}
                <h3>{"The library's own description (written for programmers)"}</h3>
                <p class="quote">{ d.doc.clone() }</p>
                <p class="links">
                    <a href={d.source_url.clone()} target="_blank">{"Its source on GitHub"}</a>{" \u{00b7} "}
                    <a href={d.xref.clone()}>{"In the cross-reference"}</a>
                </p>
            </div>
        </details>
    }
}

/// Every library function a program calls, one line each: its name,
/// the library's own description, and its links. Functions with no
/// definition found are left out (a page's test checks none is).
pub fn table(program: &str) -> Html {
    let ims = imports(program);
    let linker = Linker::of(program);
    let rows = calls(program).into_iter().filter_map(|(lib, name)| {
        let alias = ims.iter().find(|i| i.library == lib).map(|i| i.alias.clone())?;
        let d = definition(&lib, &name)?;
        Some(html! {
            <tr>
                <td>{ linker.code(&format!("{alias}{name}{}", if d.source.starts_with("m:") { "<" } else { "" })) }</td>
                <td>{ d.doc.clone() }</td>
                <td class="links"><a href={d.source_url.clone()} target="_blank">{"source"}</a>{" \u{00b7} "}<a href={d.xref.clone()}>{"cross-reference"}</a></td>
            </tr>
        })
    });
    html! { <table class="functions"><tr><th>{"Function"}</th><th>{"What the library says it does"}</th><th>{"Read it"}</th></tr>{ for rows }</table> }
}

const BUILTINS: &str = include_str!("../builtins.tsv");

/// What a built-in word of X_eTaL does, in plain words (builtins.tsv),
/// or None for one not yet explained there.
pub fn word(typed: &str) -> Option<&'static str> {
    BUILTINS.lines().filter(|l| !l.starts_with('#')).find_map(|l| l.split_once('\t').filter(|(w, _)| *w == typed).map(|(_, m)| m))
}

/// The built-in words a program uses that builtins.tsv does not
/// explain: empty, in a page's test.
pub fn unexplained(program: &str) -> Vec<String> {
    crate::code::builtins(program).into_iter().filter(|w| word(w).is_none()).collect()
}

/// Every built-in word a program uses, a line each: the word, drawn
/// and linked to its entry in the cross-reference, and what it does.
pub fn words(program: &str) -> Html {
    let linker = Linker::of(program);
    let rows = crate::code::builtins(program).into_iter().filter_map(|w| {
        let meaning = word(&w)?;
        Some(html! { <tr><td>{ linker.code(&w) }</td><td>{ meaning }</td></tr> })
    });
    html! { <table class="functions"><tr><th>{"Word"}</th><th>{"What it does"}</th></tr>{ for rows }</table> }
}
