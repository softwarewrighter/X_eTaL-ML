//! The program the page shows (everything it runs) and the map's colors.

use microscope::color::Rgb;
use microscope::source::{between, block, Range};
use yew::Html;

use crate::micro::{head, parts, tail, train_head, train_tail, typed_programs, Network, SIDE};

/// What the page runs, all of it: for a network of the program, the
/// head (the data, the three networks) then the run's lines; for a
/// typed spec, the two small programs (one expanded, one run).
pub fn program(picked: Option<&Network>, spec: &str) -> String {
    match picked {
        Some(n) => format!("{}# -- the run, for the network picked ----------------\n{}", head(), tail(n)),
        None => {
            let (network, params) = typed_programs(spec);
            format!("# -- expanded (not run: it has no weights) ------------\n{network}# -- run: its parameter count ------------------------\n{params}")
        }
    }
}

/// The line that writes the network.
pub fn range(src: &str, picked: Option<&Network>) -> Range {
    let start = match picked {
        Some(n) => n.line(),
        None => "u:n_et := ".to_string(),
    };
    between(src, &start, &start)
}

pub fn source(picked: Option<&Network>, spec: &str) -> Html {
    let src = program(picked, spec);
    block(&src, range(&src, picked))
}

/// What the page runs to train `spec`, all of it: train-it.xtl's head
/// for the spec, then a run's lines, the state and the history it
/// carries from run to run elided.
pub fn train_program(spec: &str, sizes: &[usize], steps: usize, runs: usize) -> String {
    let state = vec![0.0; parts(sizes).iter().map(|(r, c)| r * c).sum()];
    let earlier = vec![(0.0, 0.0); runs];
    let t: Vec<String> = train_tail(spec, sizes, (runs > 0).then_some(&state[..]), steps, &earlier)
        .lines()
        .map(|l| match l {
            l if l.starts_with("s := (") && l.contains(" r_eshape ") => "s := (...)   # the state after the steps so far: each layer's weights, Adam's averages, the step count".to_string(),
            l if l.starts_with("losses := (") => "losses := (...) c_at 1 t_ake now   # the loss after each run so far".to_string(),
            l if l.starts_with("right := (") => "right := (...) c_at -1 t_ake now   # the share right after each run so far".to_string(),
            l => l.to_string(),
        })
        .collect();
    format!("{}# -- each run: a few steps from the state the page holds --\n{}\n", train_head(spec, sizes), t.join("\n"))
}

pub fn train_source(spec: &str, sizes: &[usize], steps: usize, runs: usize) -> Html {
    let src = train_program(spec, sizes, steps, runs);
    let start = "\"u:s_tep X Y lr\" net:t_rain<";
    block(&src, between(&src, start, start))
}

const ARMS: [Rgb; 3] = [[255, 224, 178], [200, 230, 201], [187, 222, 251]];
const DOTS: [Rgb; 3] = [[230, 81, 0], [27, 94, 32], [13, 71, 161]];

/// The map as pixels: each point's arm as a light color, the test
/// points on top in their own arm's dark color (a miss shows as a dot
/// on another arm's ground).
pub fn map(arms: &[usize], points: &[f64]) -> Vec<u8> {
    let mut px: Vec<u8> = arms.iter().flat_map(|&a| { let [r, g, b] = ARMS[(a.max(1) - 1) % 3]; [r, g, b, 255] }).collect();
    let n = points.len() / 3;
    for i in 0..n {
        let (x, y, arm) = (points[i], points[n + i], points[2 * n + i] as usize);
        let col = ((x + 1.1) / 2.2 * SIDE as f64) as i64;
        let row = ((1.1 - y) / 2.2 * SIDE as f64) as i64;
        if (0..SIDE as i64).contains(&col) && (0..SIDE as i64).contains(&row) {
            let k = 4 * (row as usize * SIDE + col as usize);
            px[k..k + 3].copy_from_slice(&DOTS[(arm.max(1) - 1) % 3]);
        }
    }
    px
}
