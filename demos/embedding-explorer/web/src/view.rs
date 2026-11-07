//! The program the page shows (everything it runs) and the cloud's
//! drawing: each word turned, put in perspective, nearest drawn last.

use microscope::source::{between, block, Range};
use microscope::turning::{project, typical, Angle};
use yew::Html;

use crate::micro::program;

/// The program, the line that makes the embeddings highlighted.
pub fn source() -> Html {
    let src = program();
    let range: Range = between(&src, "E := ", "E := ");
    block(&src, range)
}

/// A word on the screen: where, how near (bigger is nearer), its index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dot {
    pub x: f64,
    pub y: f64,
    pub depth: f64,
    pub word: usize,
}

/// The picture is SIZE by SIZE.
pub const SIZE: f64 = 520.0;

/// Each word's place on the screen, turned by yaw and pitch, the
/// farthest first (so the nearest are drawn over them): microscope's
/// turning view's projection, scaled by a typical word's distance from
/// the middle (the 85th percentile), so a few outliers do not shrink
/// the rest.
pub fn dots(xyz: &[f64], yaw: f64, pitch: f64) -> Vec<Dot> {
    let scale = typical(xyz);
    let a = Angle { yaw, pitch };
    let mut d: Vec<Dot> = xyz
        .chunks(3)
        .enumerate()
        .map(|(i, p)| {
            let (x, y, depth) = project([p[0], p[1], p[2]], a, scale, SIZE);
            Dot { x, y, depth, word: i }
        })
        .collect();
    d.sort_by(|a, b| a.depth.total_cmp(&b.depth));
    d
}

/// A cluster's color.
pub const COLORS: [&str; 6] = ["#e8590c", "#2b8a3e", "#1c7ed6", "#c2255c", "#7048e8", "#868e96"];
