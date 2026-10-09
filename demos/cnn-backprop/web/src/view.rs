//! The program the page shows (everything it runs), the filters' and
//! the digits' pixels.

use microscope::color::{pixels, ramp, DIVERGE};
use microscope::source::{between, block, Range};
use yew::Html;

use crate::micro::{head, tail};

/// What the page runs, all of it: the head, then a run's lines, the
/// history it carries from run to run elided.
pub fn program(trained: bool, steps: usize, runs: usize) -> String {
    let earlier = vec![(0.0, 0.0); runs];
    let t: Vec<String> = tail(trained, steps, &earlier)
        .lines()
        .map(|l| match l {
            l if l.starts_with("losses := (") => "losses := (...) c_at 1 t_ake now   # the test loss after each run so far".to_string(),
            l if l.starts_with("right := (") => "right := (...) c_at -1 t_ake now   # the test share right after each run so far".to_string(),
            l => l.to_string(),
        })
        .collect();
    format!(
        "{}# -- each run: two steps from the state the page keeps in state/*.txt --\n{}\n",
        head(),
        t.join("\n")
    )
}

/// The backward pass.
pub fn range(src: &str) -> Range {
    between(src, "u:g_rad := { (K, B, W) i ->", "\n}")
}

pub fn source(trained: bool, steps: usize, runs: usize) -> Html {
    let src = program(trained, steps, runs);
    block(&src, range(&src))
}

/// The 8 filters (K, 8 rows of 9) side by side as 3 x 3 squares, a
/// column apart: blue negative, red positive, scaled by the largest.
pub const FILTER_COLS: usize = 8 * 4 - 1;
pub fn filters(k: &[f64]) -> Vec<u8> {
    let m = k.iter().fold(1e-12_f64, |m, v| m.max(v.abs()));
    let mut v = vec![f64::NAN; 3 * FILTER_COLS];
    for f in 0..8 {
        for r in 0..3 {
            for c in 0..3 {
                v[r * FILTER_COLS + f * 4 + c] = k[f * 9 + r * 3 + c];
            }
        }
    }
    pixels(&v, |x| if x.is_nan() { [255, 255, 255] } else { ramp(&DIVERGE, 0.5 + x / (2.0 * m)) })
}

/// Digits (28 x 28 each, 0 to 255) side by side, a column apart: dark on light.
pub fn strip(digits: &[&[f64]]) -> Vec<u8> {
    let cols = digits.len() * 29 - 1;
    let mut v = vec![0.0; 28 * cols];
    for (d, px) in digits.iter().enumerate() {
        for r in 0..28 {
            for c in 0..28 {
                v[r * cols + d * 29 + c] = px[r * 28 + c] / 255.0;
            }
        }
    }
    pixels(&v, |x| {
        let g = (250.0 - 230.0 * x) as u8;
        [g, g, g]
    })
}
