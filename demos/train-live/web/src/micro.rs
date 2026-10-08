//! The model: the demo's own X_eTaL program (train-live.xtl) run for a
//! number of training steps from a state (a tuple of seven arrays: W1,
//! W2, Adam's averages per layer, the step count). Nothing here knows about
//! the browser.

use microscope::run::{lit, numbers, output_pictures, section};

/// The command-line program.
pub const SOURCE: &str = include_str!("../../train-live.xtl");

/// The state's seven arrays, in order: W1, W2, M1, M2, V1, V2, k; their
/// shapes (rows, columns; the step count a single number).
pub const PARTS: [(usize, usize); 7] = [(3, 16), (17, 3), (3, 16), (17, 3), (3, 16), (17, 3), (1, 1)];
/// Every number of the state, the seven arrays one after another.
pub const STATE: usize = 298;
/// The decision map is SIDE by SIDE points over -1.1 .. 1.1.
pub const SIDE: usize = 40;
/// Training points.
pub const POINTS: usize = 300;

/// The program's head: the spiral, the network, the gradient, Adam and
/// the starting state, with the page's learning rate.
pub fn head(lr: f64) -> String {
    section(SOURCE, "\"nn:\" u_se< \"NN\"", "# -- end of the core")
        .lines()
        .map(|l| match l.starts_with("lr := ") {
            true => format!("lr := {}", lit(lr)),
            false => l.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

/// The program's line binding `name`.
fn line(name: &str) -> &'static str {
    SOURCE.lines().find(|l| l.starts_with(&format!("{name} := "))).unwrap_or("")
}

/// Earlier values and this run's, as X_eTaL: `(v1 v2 ...) c_at NOW`.
fn history(earlier: &[f64], now: &str) -> String {
    match earlier.is_empty() {
        true => now.to_string(),
        false => format!("({}) c_at {now}", earlier.iter().map(|&v| lit(v)).collect::<Vec<_>>().join(" ")),
    }
}

/// The loss and the share right after every run so far, drawn by Plot
/// in one chart.
fn curves(earlier: &[(f64, f64)]) -> String {
    format!(
        "\"p:\" u_se< \"Plot\"\n\
         losses := {}\n\
         right := {}\n\
         chart := (\"Training\" \"run (25 steps each)\" \"\" \"loss\" \"share right\") p:c_hart! (e_nclose losses) c_at e_nclose right\n",
        history(&earlier.iter().map(|e| e.0).collect::<Vec<_>>(), "1 t_ake now"),
        history(&earlier.iter().map(|e| e.1).collect::<Vec<_>>(), "-1 t_ake now"),
    )
}

/// The run's own lines: the state to start from (the program's s0, or
/// the page's), `steps` Adam steps, what the page shows, then the loss
/// and share right so far (`earlier`: after each earlier run) drawn by
/// the Plot library.
pub fn tail(state: Option<&[f64]>, steps: usize, earlier: &[(f64, f64)]) -> String {
    let start = match state {
        None => "s := s0".to_string(),
        Some(s) => format!("s := {}", tuple(s)),
    };
    format!(
        "{start}\ns := {steps} 'u:a_dam p_ower s\n{}\n{}\ng := {SIDE}\n\
         c := -1.1 + 2.2 * (0.5 + f_loat o_ffsets g) / f_loat g\n\
         grid := o_\\ (2 c_at g * g) r_eshape (r_avel (o_ffsets g) 'r_ight t_able c) c_at r_avel (r_ev c) 'l_eft t_able o_ffsets g\n\
         (w1, w2, m1, m2, v1, v2, k) := s\n\
         (r_avel w1) c_at (r_avel w2) c_at (r_avel m1) c_at (r_avel m2) c_at (r_avel v1) c_at (r_avel v2) c_at k\n\
         now := (u:l_oss s) c_at u:r_ight s\nnow\nnn:a_rgmax nn:s_oftmax (nn:t_anh grid nn:d_ense w1) nn:d_ense w2\nr_avel X\n1 + arm\n{}",
        line("u:l_oss"),
        line("u:r_ight"),
        curves(earlier)
    )
}

/// The state's numbers as X_eTaL: a tuple of the seven arrays.
pub fn tuple(s: &[f64]) -> String {
    let mut at = 0;
    let t = PARTS
        .iter()
        .map(|&(r, c)| {
            let part: Vec<String> = s[at..at + r * c].iter().map(|&v| lit(v)).collect();
            at += r * c;
            match (r, c) {
                (1, 1) => part[0].clone(),
                _ => format!("{r} {c} r_eshape {}", part.join(" ")),
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("({t})")
}

pub fn program(lr: f64, state: Option<&[f64]>, steps: usize, earlier: &[(f64, f64)]) -> String {
    format!("{}{}", head(lr), tail(state, steps, earlier))
}

/// After a run of steps, as X_eTaL computed it.
#[derive(Clone, Debug, PartialEq)]
pub struct After {
    pub state: Vec<f64>,
    pub loss: f64,
    pub right: f64,
    /// The arm (1 to 3) decided at each map point, row by row.
    pub map: Vec<usize>,
    /// The training points, x y pairs, and their arms.
    pub points: Vec<f64>,
    pub arms: Vec<usize>,
    /// Plot's charts of the loss and of the share right so far, SVG
    /// (none on the first run).
    pub pictures: Vec<String>,
}

pub fn run(lr: f64, state: Option<&[f64]>, steps: usize, earlier: &[(f64, f64)]) -> Result<After, String> {
    let (out, pictures) = output_pictures(&program(lr, state, steps, earlier), 5)?;
    let lr2 = numbers::<f64>(&out[1], 2)?;
    Ok(After {
        state: numbers(&out[0], STATE)?,
        loss: lr2[0],
        right: lr2[1],
        map: numbers(&out[2], SIDE * SIDE)?,
        points: numbers(&out[3], 2 * POINTS)?,
        arms: numbers(&out[4], POINTS)?,
        pictures,
    })
}
