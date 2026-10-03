//! The program the page shows (the router weights, the core, the
//! sentence's lines), the stages and the part computing each.

use microscope::source::{between, block, find, Range};
use yew::Html;

use crate::micro::{core, nudge_core, weights, Nudge, SIDE, STEPS};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Embed,
    Scores,
    Softmax,
    Top2,
    Load,
    Nudge,
}

pub const STAGES: [Stage; 6] = [Stage::Embed, Stage::Scores, Stage::Softmax, Stage::Top2, Stage::Load, Stage::Nudge];

/// The router weights, the core, the lines that route the sentence,
/// and the nudge.
pub fn program(ids: &[usize], n: &Nudge) -> String {
    let ids: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
    format!(
        "# F, E: the 37 words' features and embeddings (not shown)\n{}{}# -- the sentence\nids := {}\nx := ids s_elect E\np := u:s_oftmax u:s_cores x\ngates := u:t_op2 p\nu:l_oad gates\n\
         # -- the nudge\nw0 := {}\nwa := {}\nwb := {}\nk := {STEPS}\nside := {SIDE}\n{}",
        weights(),
        core(),
        ids.join(" "),
        n.w0,
        n.wa,
        n.wb,
        nudge_core()
    )
}

pub fn range(src: &str, stage: Stage) -> Range {
    match stage {
        Stage::Embed => find(src, "x := ids s_elect E"),
        Stage::Scores => between(src, "u:s_cores := ", "u:s_cores := "),
        Stage::Softmax => between(src, "u:s_oftmax := { s ->", "\n}"),
        Stage::Top2 => between(src, "u:t_op2 := { p ->", "\n}"),
        Stage::Load => between(src, "u:l_oad := ", "u:l_oad := "),
        Stage::Nudge => between(src, "x0 := w0 s_elect E", "gs := "),
    }
}

pub fn source(ids: &[usize], n: &Nudge, focus: Stage) -> Html {
    let src = program(ids, n);
    block(&src, range(&src, focus))
}

/// Grid geometry of the routing picture.
pub const TOK_X: f64 = 120.0;
pub const GRID_X: f64 = 330.0;
pub const CELL_W: f64 = 112.0;
pub const CELL_H: f64 = 64.0;
pub const GAP: f64 = 8.0;
pub const TOP: f64 = 24.0;

/// The top-left corner of expert e's box.
pub fn expert_box(e: usize) -> (f64, f64) {
    (GRID_X + (e % 4) as f64 * (CELL_W + GAP), TOP + (e / 4) as f64 * (CELL_H + GAP))
}

/// The y of token t's label, spread over the grid's height.
pub fn token_y(t: usize, n: usize) -> f64 {
    let h = 4.0 * (CELL_H + GAP) - GAP;
    TOP + h * (t as f64 + 0.5) / n as f64
}

fn hsl(h: f64, s: f64, l: f64) -> [u8; 3] {
    let a = s * l.min(1.0 - l);
    let f = |n: f64| {
        let k = (n + h / 30.0) % 12.0;
        let v = l - a * (k - 3.0).min(9.0 - k).clamp(-1.0, 1.0);
        (255.0 * v) as u8
    };
    [f(0.0), f(8.0), f(4.0)]
}

/// A colour for each (first, second) pair of experts.
pub fn pair_colour(first: f64, second: f64) -> [u8; 3] {
    let k = (first as usize * 7 + second as usize * 3) % 16;
    hsl(k as f64 * 360.0 / 16.0, 0.65, if first as usize % 2 == 0 { 0.72 } else { 0.62 })
}

/// The slice: each point coloured by its pair; the token, the two
/// target words and the current point marked dark.
pub fn slice(first: &[f64], second: &[f64]) -> Vec<u8> {
    let mut px = Vec::with_capacity(4 * first.len());
    for i in 0..first.len() {
        let [r, g, b] = pair_colour(first[i], second[i]);
        px.extend([r, g, b, 255]);
    }
    for (a, b) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)] {
        let (r, c) = slice_cell(a, b);
        for (dr, dc) in [(0i64, 0i64), (0, 1), (1, 0), (1, 1)] {
            let (rr, cc) = ((r as i64 + dr).min(SIDE as i64 - 1) as usize, (c as i64 + dc).min(SIDE as i64 - 1) as usize);
            px[4 * (rr * SIDE + cc)..4 * (rr * SIDE + cc) + 3].copy_from_slice(&[20, 20, 30]);
        }
    }
    px
}

/// The slice cell (row, column) of the point x0 + a d1 + b d2.
pub fn slice_cell(a: f64, b: f64) -> (usize, usize) {
    let k = |v: f64| (((v + 0.25) / 1.5 * (SIDE - 1) as f64).round().clamp(0.0, (SIDE - 1) as f64)) as usize;
    (SIDE - 1 - k(b), k(a))
}

/// The strip: experts down, epsilon across, gates as brightness, the
/// cursor's column outlined.
pub fn strip(gates: &[f64], cursor: usize) -> Vec<u8> {
    let mut px = vec![0u8; 4 * 16 * STEPS];
    for e in 0..16 {
        for i in 0..STEPS {
            let g = gates[i * 16 + e];
            let [r, gg, b] = microscope::colour::ramp(&microscope::colour::GLOW, g);
            let [r, gg, b] = if i == cursor && g == 0.0 { [255, 146, 43] } else { [r, gg, b] };
            px[4 * (e * STEPS + i)..4 * (e * STEPS + i) + 4].copy_from_slice(&[r, gg, b, 255]);
        }
    }
    px
}
