//! The program the page shows (everything it runs) and the cloud's
//! drawing: each word turned, put in perspective, nearest drawn last.

use microscope::source::{between, block, Range};
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
/// farthest first (so the nearest are drawn over them).
pub fn dots(xyz: &[f64], yaw: f64, pitch: f64) -> Vec<Dot> {
    // Scaled by a typical word's distance from the middle (the 85th
    // percentile), not the farthest, so a few outliers do not shrink the rest.
    let mut radii: Vec<f64> = xyz.chunks(3).map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()).collect();
    radii.sort_by(f64::total_cmp);
    let scale = radii.get(radii.len() * 85 / 100).copied().unwrap_or(1.0).max(1e-12) * 1.15;
    let (cy, sy, cp, sp) = (yaw.cos(), yaw.sin(), pitch.cos(), pitch.sin());
    let mut d: Vec<Dot> = xyz
        .chunks(3)
        .enumerate()
        .map(|(i, p)| {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() / scale;
            let c = if r > 1.25 { 1.25 / r } else { 1.0 };
            let (x, y, z) = (c * p[0] / scale, c * p[1] / scale, c * p[2] / scale);
            let (x1, z1) = (x * cy + z * sy, -x * sy + z * cy);
            let (y2, z2) = (y * cp - z1 * sp, y * sp + z1 * cp);
            let k = 1.0 / (1.6 - 0.35 * z2);
            Dot { x: SIZE / 2.0 + 0.9 * SIZE / 2.0 * x1 * k * 1.6, y: SIZE / 2.0 - 0.9 * SIZE / 2.0 * y2 * k * 1.6, depth: z2, word: i }
        })
        .collect();
    d.sort_by(|a, b| a.depth.total_cmp(&b.depth));
    d
}

/// A cluster's color.
pub const COLORS: [&str; 6] = ["#e8590c", "#2b8a3e", "#1c7ed6", "#c2255c", "#7048e8", "#868e96"];
