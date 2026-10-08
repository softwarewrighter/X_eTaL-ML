//! The program the page shows, the map's pixels and the curves.

use microscope::color::Rgb;
use microscope::source::{between, block, Range};
use yew::Html;

use crate::micro::{head, tail, SIDE, STATE};

/// What the page runs, all of it: the head, then the run's lines, the
/// state it carries from run to run elided once it has trained.
pub fn program(lr: f64, trained: bool, steps: usize) -> String {
    let earlier = [(0.0, 0.0)];
    let t: Vec<String> = tail(trained.then_some(&[0.0; STATE][..]), steps, if trained { &earlier[..] } else { &[] })
        .lines()
        .map(|l| match l {
            l if l.starts_with("s := (3 16 r_eshape ") => "s := (...)   # the state after the steps so far: (W1, W2, M1, M2, V1, V2, k)".to_string(),
            l if l.starts_with("losses := (") => "losses := (...) c_at 1 t_ake now   # the loss after each run so far".to_string(),
            l if l.starts_with("right := (") => "right := (...) c_at -1 t_ake now   # the share right after each run so far".to_string(),
            l => l.to_string(),
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
