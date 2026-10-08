//! Running an X_eTaL program and reading the arrays it prints back, and
//! writing values into a program as literals.

use std::str::FromStr;

/// `x` as an X_eTaL Float literal: the shortest plain decimal that
/// reads back as the same f64 (X_eTaL number literals have no exponent;
/// Rust's `{}` never writes one). Sizes under `floor` are written 0.0
/// (keeping literals short where tiny values do not matter).
pub fn lit_or_zero(x: f64, floor: f64) -> String {
    if x == 0.0 || x.abs() < floor || !x.is_finite() {
        return "0.0".into();
    }
    let s = format!("{x}");
    if s.contains('.') { s } else { format!("{s}.0") }
}

/// `x` as an X_eTaL Float literal that reads back exactly.
pub fn lit(x: f64) -> String {
    lit_or_zero(x, 0.0)
}

/// A binding of a rows x cols matrix: `name := rows cols r_eshape ...`.
pub fn matrix(name: &str, rows: usize, cols: usize, items: impl IntoIterator<Item = String>) -> String {
    let body: Vec<String> = items.into_iter().collect();
    format!("{name} := {rows} {cols} r_eshape {}\n", body.join(" "))
}

/// Run `src` and return its output lines, which must number `lines`;
/// an X_eTaL error (or a different count) is the Err.
pub fn output(src: &str, lines: usize) -> Result<Vec<String>, String> {
    output_pictures(src, lines).map(|(out, _)| out)
}

/// Run `src`: its output lines, which must number `lines`, and the
/// pictures it showed with `[]S_HOW` (Plot's charts), each an SVG
/// document.
pub fn output_pictures(src: &str, lines: usize) -> Result<(Vec<String>, Vec<String>), String> {
    crate::libs::install();
    let run = xetal_play::run(src, 1);
    if !run.err.is_empty() {
        return Err(run.err);
    }
    let out: Vec<String> = run.out.lines().map(str::to_string).collect();
    match out.len() == lines {
        true => Ok((out, run.pictures)),
        false => Err(format!("expected {lines} lines of output, got {}", out.len())),
    }
}

/// `src` after macro expansion, as `xetal expand` prints it (the
/// libraries, macro libraries among them, from the pages' store), or
/// the error a macro call was refused with.
pub fn expanded(src: &str) -> Result<String, String> {
    crate::libs::install();
    xetal_program::expanded_with("page", src, &xetal_macro::StoreLibraries).map_err(|d| d.to_string())
}

/// The `want` numbers of one printed line (an `r_avel` of an array).
pub fn numbers<T: FromStr>(line: &str, want: usize) -> Result<Vec<T>, String>
where
    T::Err: std::fmt::Display,
{
    let v: Result<Vec<T>, _> = line.split_whitespace().map(str::parse).collect();
    let v = v.map_err(|e| format!("unexpected output {line:.60}: {e}"))?;
    match v.len() == want {
        true => Ok(v),
        false => Err(format!("expected {want} numbers, got {}", v.len())),
    }
}

/// The part of `src` from the line holding `start` up to `end`.
pub fn section<'a>(src: &'a str, start: &str, end: &str) -> &'a str {
    let a = src.find(start).unwrap_or(0);
    let b = src[a..].find(end).map_or(src.len(), |i| a + i);
    &src[a..b]
}

/// Milliseconds from the browser's clock (0 when not in a browser).
pub fn now() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::window().and_then(|w| w.performance()).map_or(0.0, |p| p.now())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        0.0
    }
}
