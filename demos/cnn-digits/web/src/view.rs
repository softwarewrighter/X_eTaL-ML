//! The program the page shows (everything it runs: the head, then the
//! run's own lines), the stages and the part computing each.

use microscope::source::{between, block, Range};
use yew::Html;

use crate::micro::{head, tail, Input};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Digit,
    Conv,
    Pool,
    Classify,
}

pub const STAGES: [Stage; 4] = [Stage::Digit, Stage::Conv, Stage::Pool, Stage::Classify];

/// What the page runs, all of it: the head (the network read from
/// data/, the core), then the run's lines. Only a drawn digit's 784
/// values are elided, and a comment says so.
pub fn program(input: &Input) -> String {
    let run: Vec<String> = tail(input)
        .lines()
        .map(|l| match (input, l.starts_with("x := 28 28 r_eshape ")) {
            (Input::Drawn(_), true) => "x := 28 28 r_eshape ...   # the digit you drew, 784 numbers".to_string(),
            _ => l.to_string(),
        })
        .collect();
    format!("{}# -- the run, when the digit changes --------------\n{}\n", head(), run.join("\n"))
}

pub fn range(src: &str, stage: Stage) -> Range {
    match stage {
        Stage::Digit => between(src, "x := ", "x := "),
        Stage::Conv => between(src, "u:c_onv := { x ->", "\n}"),
        Stage::Pool => between(src, "u:p_ool := ", "u:p_ool := "),
        Stage::Classify => between(src, "u:c_lassify := ", "u:c_lassify := "),
    }
}

pub fn source(input: &Input, focus: Stage) -> Html {
    let src = program(input);
    block(&src, range(&src, focus))
}
