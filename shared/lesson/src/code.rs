//! X_eTaL code drawn as X_eTaL draws it, with every function a link
//! into the cross-reference (pages/doc, made by `xetal doc`): a
//! built-in to its entry, a library function to its definition in its
//! library, whether the library is X_eTaL-ML's, X_eTaL-libraries' or
//! this repository's (the user, 2026-11-05: "direct links from code
//! snippets to the xref for any library function"). The drawing is
//! microscope's (`microscope::source`), which has no links.

use xetal_play::{decorate, Class};
use yew::prelude::*;

use crate::explain::{file_page, imports};

/// The operators that are built-in functions with an entry of their own.
const OPERATORS: [&str; 13] = ["+", "-", "*", "/", "^", "=", "!=", "<", ">", "<=", ">=", "&", "|"];

/// An element id of the cross-reference for `name`, as `xetal doc`
/// makes it: letters, digits and `_` as they are, `:` as `.`, anything
/// else as `-` and its code in hex (`r_/` is `r_-2f`).
pub fn anchor(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            c if c.is_ascii_alphanumeric() || c == '_' => c.to_string(),
            ':' => ".".to_string(),
            c => format!("-{:x}", c as u32),
        })
        .collect()
}

fn css(class: Class) -> &'static str {
    match class {
        Class::Builtin => "t-builtin",
        Class::UserFunc | Class::LibFunc | Class::Macro => "t-user",
        Class::LambdaArg => "t-arg",
        Class::Number | Class::Exponent => "t-num",
        Class::Symbol | Class::Quote => "t-sym",
        Class::Comment => "t-comment",
        _ => "t-plain",
    }
}

/// What links a snippet's names: the short names its program gave the
/// libraries it imports (`lo:` for LoRA), each with its library's page.
#[derive(Clone, Debug, PartialEq)]
pub struct Linker {
    /// (the alias with its colon, the library it names).
    aliases: Vec<(String, String)>,
}

impl Linker {
    /// The linker of a program: its imports.
    pub fn of(program: &str) -> Linker {
        Linker { aliases: imports(program).into_iter().map(|i| (i.alias, i.library)).collect() }
    }

    /// The linker of a library's own source: `l:` is the library
    /// itself, and its imports are as it names them.
    pub fn within(library: &str, source: &str) -> Linker {
        let mut l = Linker::of(source);
        l.aliases.push(("l:".to_string(), library.to_string()));
        l
    }

    /// Where a segment of class `class`, typed `raw`, is defined in the
    /// cross-reference, if it is a function with an entry there.
    pub fn href(&self, class: Class, raw: &str) -> Option<String> {
        match class {
            // The quote before a quoted function is drawn with it; it has no entry.
            Class::Builtin if raw == "'" => None,
            Class::Builtin => Some(format!("../doc/builtins.html#{}", anchor(raw))),
            Class::Symbol if OPERATORS.contains(&raw) => Some(format!("../doc/builtins.html#{}", anchor(raw))),
            // A library's function is in its Name.xtl, under l:name.
            Class::LibFunc | Class::UserFunc => {
                let (alias, library) = self.aliases.iter().find(|(a, _)| raw.starts_with(a.as_str()))?;
                Some(format!("{}#l.{}", file_page(&format!("{library}.xtl"))?, anchor(&raw[alias.len()..])))
            }
            // A library's macro (its name ends in <) is in its Name.xtlm, under m:name<.
            Class::Macro => {
                let (alias, library) = self.aliases.iter().find(|(a, _)| raw.starts_with(a.as_str()))?;
                Some(format!("{}#m.{}", file_page(&format!("{library}.xtlm"))?, anchor(&raw[alias.len()..])))
            }
            _ => None,
        }
    }

    fn segments(&self, src: &str) -> Html {
        decorate(src)
            .into_iter()
            .map(|s| {
                let raw = &src[s.raw.start..s.raw.end];
                match self.href(s.class, raw) {
                    Some(href) => html! { <a class={classes!(css(s.class), "xl")} {href} title="Its entry in the cross-reference">{s.text}</a> },
                    None => html! { <span class={css(s.class)}>{s.text}</span> },
                }
            })
            .collect()
    }

    /// A snippet, drawn decorated and linked, inline.
    pub fn code(&self, src: &str) -> Html {
        html! { <code class="xtl">{ self.segments(src) }</code> }
    }

    /// A program (several lines, with comments), drawn decorated and linked.
    pub fn block(&self, src: &str) -> Html {
        html! { <pre class="source">{ self.segments(src) }</pre> }
    }
}

/// The built-in words a program uses (`r_eshape`, `i_nner`), as typed,
/// each once, in the order first met: what a page must explain to a
/// reader who has never seen X_eTaL.
pub fn builtins(src: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for s in decorate(src) {
        let raw = &src[s.raw.start..s.raw.end];
        if s.class == Class::Builtin && raw != "'" && !out.iter().any(|b| b == raw) {
            out.push(raw.to_string());
        }
    }
    out
}
