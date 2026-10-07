//! The program the page shows, the map's pixels and the curves.

use microscope::color::Rgb;
use microscope::source::{between, block, Range};
use yew::Html;

use crate::micro::{head, tail, SIDE, STATE};

/// What the page runs, all of it: the head, then the run's lines, the
/// state it carries from run to run elided once it has trained.
pub fn program(lr: f64, trained: bool, steps: usize) -> String {
    let t: Vec<String> = tail(trained.then_some(&[0.0; STATE][..]), steps)
        .lines()
        .map(|l| match l.starts_with("s := (e_nclose ") {
            true => "s := ...   # the state after the steps so far: seven boxed arrays (W1 W2 M1 M2 V1 V2 k)".to_string(),
            false => l.to_string(),
        })
        .collect();
    format!("{}# -- each run: a few dozen steps from the state the page holds --\n{}\n", head(lr), t.join("\n"))
}

/// Adam's step.
pub fn range(src: &str) -> Range {
    between(src, "u:a_dam := { s ->", "\n}")
}

pub fn source(lr: f64, trained: bool, steps: usize) -> Html {
    let src = program(lr, trained, steps);
    block(&src, range(&src))
}

const ARMS: [Rgb; 3] = [[255, 224, 178], [200, 230, 201], [187, 222, 251]];
const DOTS: [Rgb; 3] = [[230, 81, 0], [27, 94, 32], [13, 71, 161]];

/// The map as pixels, the training points on it in their arm's color.
pub fn map(arms: &[usize], points: &[f64], labels: &[usize]) -> Vec<u8> {
    let mut px: Vec<u8> = arms.iter().flat_map(|&a| { let [r, g, b] = ARMS[(a.max(1) - 1) % 3]; [r, g, b, 255] }).collect();
    for (k, &arm) in labels.iter().enumerate() {
        let (x, y) = (points[2 * k], points[2 * k + 1]);
        let col = ((x + 1.1) / 2.2 * SIDE as f64) as i64;
        let row = ((1.1 - y) / 2.2 * SIDE as f64) as i64;
        if (0..SIDE as i64).contains(&col) && (0..SIDE as i64).contains(&row) {
            let i = 4 * (row as usize * SIDE + col as usize);
            px[i..i + 3].copy_from_slice(&DOTS[(arm.max(1) - 1) % 3]);
        }
    }
    px
}

/// An SVG polyline's points for values over steps (0 .. max_steps
/// across, 0 .. top up).
pub fn polyline(values: &[(usize, f64)], max_steps: usize, top: f64, w: f64, h: f64) -> String {
    values
        .iter()
        .map(|&(s, v)| format!("{:.1},{:.1}", w * s as f64 / max_steps.max(1) as f64, h - h * (v / top).clamp(0.0, 1.0)))
        .collect::<Vec<_>>()
        .join(" ")
}
