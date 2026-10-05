//! The program the page shows, the stages and the part computing each,
//! and the maps' colors.

use microscope::color::Rgb;
use microscope::source::{between, block, Range};
use yew::Html;

use crate::micro::{head, program_formats, program_point, program_ternary, Setup, SIDE};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Inputs,
    Network,
    Formats,
    Additions,
    Maps,
    Measures,
}

pub const STAGES: [Stage; 6] = [Stage::Inputs, Stage::Network, Stage::Formats, Stage::Additions, Stage::Maps, Stage::Measures];

/// What the page runs, all of it: the head every run shares (the
/// page's settings, the weights read from data/, the functions), then
/// each of its three runs as run. Only the FP32 outputs the second run
/// is given (SIDE x SIDE x 3 numbers from the first) are elided, and a
/// comment says so. (The command-line program, ternary-net.xtl, is the
/// head and its own report at the end; see the demo's README.)
pub fn program(s: &Setup) -> String {
    let h = head(s);
    let tail = |p: String| p.strip_prefix(h.as_str()).map(str::to_string).unwrap_or(p);
    let ternary: Vec<String> = tail(program_ternary(s, &[]))
        .lines()
        .map(|l| match l.starts_with("y32 := (") {
            true => format!("y32 := ({} c_at 3) r_eshape ...   # FP32's outputs from run 1, passed in", SIDE * SIDE),
            false => l.to_string(),
        })
        .collect();
    format!(
        "{h}# -- run 1, when the weight set changes: FP32, FP16, INT8 --\n{}\
         # -- run 2, when the threshold moves: the ternary weights --\n{}\n\
         # -- run 3, for the point clicked: one input through the layers --\n{}",
        tail(program_formats(s)),
        ternary.join("\n"),
        tail(program_point(s)),
    )
}

pub fn range(src: &str, stage: Stage) -> Range {
    match stage {
        Stage::Inputs => between(src, "e1 := ", "test := "),
        Stage::Network => between(src, "u:l_ayer := ", "  3 t_ake_2"),
        Stage::Formats => between(src, "u:f_loats := ", "m2 := "),
        Stage::Additions => between(src, "u:a_dds := ", "u:a_dds := "),
        Stage::Maps => between(src, "y32 := grid", "y8 := grid"),
        Stage::Measures => between(src, "u:m_easures := { k y ->", "  acc c_at agree c_at err"),
    }
}

pub fn source(s: &Setup, focus: Stage) -> Html {
    let src = program(s);
    block(&src, range(&src, focus))
}

/// Each arm's color, strong (points) and light (the map).
pub const ARMS: [(Rgb, Rgb); 3] =
    [([112, 72, 232], [222, 211, 255]), ([232, 89, 12], [255, 222, 189]), ([12, 140, 100], [200, 245, 228])];

/// Pixels per map cell.
pub const CELL: usize = 4;
pub const PIC: usize = SIDE * CELL;

/// A map (arms 1 to 3, SIDE x SIDE) with the test points drawn on it.
pub fn map(arms: &[f64], px: &[f64], py: &[f64], pl: &[f64]) -> Vec<u8> {
    let mut img = vec![255u8; PIC * PIC * 4];
    for r in 0..PIC {
        for c in 0..PIC {
            let a = arms[(r / CELL) * SIDE + c / CELL] as usize;
            let rgb = ARMS[a.clamp(1, 3) - 1].1;
            img[4 * (r * PIC + c)..4 * (r * PIC + c) + 3].copy_from_slice(&rgb);
        }
    }
    let to = |v: f64| ((v + 1.1) / 2.2 * PIC as f64) as i64;
    for k in 0..px.len() {
        let (r, c) = (to(-py[k]), to(px[k]));
        let rgb = ARMS[(pl[k] as usize).clamp(1, 3) - 1].0;
        for (dr, dc) in [(0, 0), (0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1), (-1, 1)] {
            let (rr, cc) = (r + dr, c + dc);
            if (0..PIC as i64).contains(&rr) && (0..PIC as i64).contains(&cc) {
                let i = 4 * (rr as usize * PIC + cc as usize);
                img[i..i + 3].copy_from_slice(&rgb);
            }
        }
    }
    img
}

/// The picture cell (row, column) of the point (x, y).
pub fn cell(x: f64, y: f64) -> (usize, usize) {
    let to = |v: f64| (((v + 1.1) / 2.2 * PIC as f64) as usize).min(PIC - 1);
    (to(-y), to(x))
}
