//! The program the page shows (everything it runs).

use microscope::source::{between, block, Range};
use yew::Html;

use crate::micro::{head, tail, NAMES, STEPS};

/// What the page runs, all of it: the head, then the run's lines, the
/// browser's uniform numbers elided.
pub fn program(t: f64, start: &str, pick: usize) -> String {
    let u = vec![0.5; NAMES * STEPS];
    let lines: Vec<String> = tail(t, start, &u, pick)
        .lines()
        .map(|l| match l.starts_with("U := ") {
            true => format!("U := {NAMES} {STEPS} r_eshape ...   # {} uniform numbers from the browser's own randomness", NAMES * STEPS),
            false => l.to_string(),
        })
        .collect();
    format!("{}# -- each set of names: the uniform numbers, the names, one looked into --\n{}\n", head(), lines.join("\n"))
}

/// The forward pass.
pub fn range(src: &str) -> Range {
    between(src, "u:g_pt := { ids ->", "\n}")
}

pub fn source(t: f64, start: &str, pick: usize) -> Html {
    let src = program(t, start, pick);
    block(&src, range(&src))
}
