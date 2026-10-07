//! The program the page shows (everything it runs) and the stages.

use microscope::source::{between, block, Range};
use yew::Html;

use crate::micro::{head, tail, Setup};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Forward,
    Backward,
    Step,
    Check,
}

pub const STAGES: [Stage; 4] = [Stage::Forward, Stage::Backward, Stage::Step, Stage::Check];

/// What the page runs, all of it: the head (with the weights it holds
/// elided once it has taken steps), then the check and what it reads.
pub fn program(s: &Setup) -> String {
    let h: Vec<String> = head(s)
        .lines()
        .map(|l| match (s.weights.is_some(), l.split_once(" r_eshape ")) {
            (true, Some((lhs, _))) if l.starts_with("W1 := ") || l.starts_with("W2 := ") => format!("{lhs} r_eshape ...   # the weights after the steps taken"),
            _ => l.to_string(),
        })
        .collect();
    format!("{}\n# -- the run: the check, and every array the page shows --\n{}", h.join("\n"), tail())
}

pub fn range(src: &str, stage: Stage) -> Range {
    match stage {
        Stage::Forward => between(src, "H := ", "L := "),
        Stage::Backward => between(src, "D2 := ", "G1 := "),
        Stage::Step => between(src, "V1 := ", "V2 := "),
        Stage::Check => between(src, "F1 := ", "F2 := "),
    }
}

pub fn source(s: &Setup, focus: Stage) -> Html {
    let src = program(s);
    block(&src, range(&src, focus))
}
