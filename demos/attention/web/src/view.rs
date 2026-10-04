//! The program the page shows (everything it runs: the head, then the
//! run's own lines), the stages and the part computing each.

use microscope::source::{between, block, Range};
use yew::Html;

use crate::micro::program as run_program;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Embed,
    QueryKey,
    Scores,
    Weights,
    Output,
}

pub const STAGES: [Stage; 5] = [Stage::Embed, Stage::QueryKey, Stage::Scores, Stage::Weights, Stage::Output];

/// What the page runs, all of it (nothing elided: the sentence is short).
pub fn program(ids: &[usize], causal: bool) -> String {
    let p = run_program(ids, causal);
    let head = crate::micro::head();
    format!("{head}# -- the run, for the sentence -------------------------\n{}", &p[head.len()..])
}

pub fn range(src: &str, stage: Stage) -> Range {
    match stage {
        Stage::Embed => between(src, "X := ids s_elect F", "X := ids s_elect F"),
        Stage::QueryKey => between(src, "Q := X ", "K := X "),
        Stage::Scores => between(src, "u:s_cores := ", "u:s_cores := "),
        Stage::Weights => between(src, "u:a_ttend := ", "u:a_ttend := "),
        Stage::Output => between(src, "Y := A ", "Y := A "),
    }
}

pub fn source(ids: &[usize], causal: bool, focus: Stage) -> Html {
    let src = program(ids, causal);
    block(&src, range(&src, focus))
}
