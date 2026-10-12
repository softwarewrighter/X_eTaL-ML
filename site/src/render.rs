//! X_eTaL source in its rendered form, as HTML: the decorated text
//! (underlined letters, raised namespaces, arrows, the lamp for
//! comments) with X_eTaL's own token classes (`c-builtin`, ...), as
//! its live demo draws it. Shared by build.rs (the reference pages,
//! rendered at build time) and the app (programs, source, types).

use xetal_view::{html, view, Class};

/// `src` rendered, as HTML for inside a `<pre>` or `<code>`.
pub fn decorated(src: &str) -> String {
    html(&view(src))
}

/// Whether an inline snippet from a page is X_eTaL code (drawn
/// rendered) rather than a path, a command, a type or a name of the
/// environment (drawn as typed): it lexes cleanly and holds a function,
/// a quote, a symbol, a string or a binding.
pub fn is_xetal(code: &str) -> bool {
    if code.chars().all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit()) {
        return false; // XETAL_PATH, REG_RS_DATA_DIR
    }
    if code.starts_with("just ") || code.starts_with("xetal ") || code.starts_with('-') || code.contains('/') && !code.contains(' ') {
        return false; // commands, flags and paths
    }
    if code.contains("error[") {
        return false; // an error message
    }
    if code.contains("->") && !code.contains('{') || code.contains("=>") {
        return false; // a type (a lambda's arrow is inside braces)
    }
    let segments = view(code);
    if segments.iter().any(|s| s.class == Class::Error) {
        return false;
    }
    code.contains(":=")
        || segments.iter().any(|s| {
            matches!(
                s.class,
                Class::Builtin | Class::UserFunc | Class::LibFunc | Class::Macro | Class::LambdaArg | Class::Quote | Class::Symbol | Class::String
            )
        })
}
