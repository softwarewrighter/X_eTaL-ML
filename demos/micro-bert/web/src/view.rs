//! The program the page shows (everything it runs).

use microscope::source::{between, block, Range};
use yew::Html;

use crate::micro::{head, tail};

/// What the page runs, all of it: the head, then the run's lines.
pub fn program(name: &str) -> String {
    format!("{}# -- each name: the blank filled, where each head looks, the model as math --\n{}", head(), tail(name))
}

/// The encoder layer, the pipeline.
pub fn range(src: &str) -> Range {
    between(src, "u:l_ayer := { (H, l) ->", "\n}")
}

pub fn source(name: &str) -> Html {
    let src = program(name);
    block(&src, range(&src))
}
