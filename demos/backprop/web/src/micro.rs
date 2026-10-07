//! The model: the demo's own X_eTaL program (backprop.xtl) and its
//! data, run for one training step from the weights the page holds.
//! Nothing here knows about the browser.

use microscope::run::{lit, numbers, output, section};

/// The command-line program.
pub const SOURCE: &str = include_str!("../../backprop.xtl");

/// The data the program reads, as the page's X_eTaL finds it.
pub const DATA: [(&str, &str); 3] = [
    ("data/batch.txt", include_str!("../../data/batch.txt")),
    ("data/w1.txt", include_str!("../../data/w1.txt")),
    ("data/w2.txt", include_str!("../../data/w2.txt")),
];

/// The sizes: 6 points, 2 inputs, 4 hidden units, 3 classes.
pub const N: usize = 6;
pub const I: usize = 2;
pub const J: usize = 4;
pub const K: usize = 3;

/// What the page sets: the weights (None: the starting ones, read
/// from data/) and the learning rate.
#[derive(Clone, Debug, PartialEq)]
pub struct Setup {
    pub weights: Option<(Vec<f64>, Vec<f64>)>,
    pub lr: f64,
}

impl Default for Setup {
    fn default() -> Self {
        Setup { weights: None, lr: 0.5 }
    }
}

/// The program's head: the batch and weights, forward, backward, the
/// step (up to the end of the core), with the page's weights and
/// learning rate in place of the files' when it has them.
pub fn head(s: &Setup) -> String {
    section(SOURCE, "\"nn:\" u_se< \"NN\"", "# -- end of the core")
        .lines()
        .map(|l| match (l, &s.weights) {
            (l, _) if l.starts_with("lr := ") => format!("lr := {}", lit(s.lr)),
            (l, Some((w1, _))) if l.starts_with("W1 := ") => format!("W1 := {I_} {J} r_eshape {}", w1.iter().map(|&v| lit(v)).collect::<Vec<_>>().join(" "), I_ = I + 1),
            (l, Some((_, w2))) if l.starts_with("W2 := ") => format!("W2 := {J_} {K} r_eshape {}", w2.iter().map(|&v| lit(v)).collect::<Vec<_>>().join(" "), J_ = J + 1),
            (l, _) => l.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

/// The program's line binding `name` (the check's lines are its own).
fn line(name: &str) -> &'static str {
    SOURCE.lines().find(|l| l.starts_with(&format!("{name} := "))).unwrap_or("")
}

/// The run's own lines: the finite-difference check, as the program
/// writes it, and every array the page draws.
pub fn tail() -> String {
    format!(
        "{}\n{}\n{}\n{}\n{}\n\
         r_avel X\nr_avel Y\nr_avel H\nr_avel P\nL c_at V1 u:l_oss V2\nr_avel D2\nr_avel G2\nr_avel D1\nr_avel G1\nr_avel V1\nr_avel V2\nr_avel F1\nr_avel F2\ngap\n",
        line("e"),
        line("u:b_ump"),
        line("F1"),
        line("F2"),
        line("gap")
    )
}

pub fn program(s: &Setup) -> String {
    format!("{}{}", head(s), tail())
}

/// One training step, as X_eTaL computed it.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub h: Vec<f64>,
    pub p: Vec<f64>,
    /// The loss before and after the step.
    pub loss: (f64, f64),
    pub d2: Vec<f64>,
    pub g2: Vec<f64>,
    pub d1: Vec<f64>,
    pub g1: Vec<f64>,
    /// The weights after the step.
    pub v1: Vec<f64>,
    pub v2: Vec<f64>,
    /// The finite-difference gradients.
    pub f1: Vec<f64>,
    pub f2: Vec<f64>,
    /// The largest difference between the analytic and finite-difference gradients.
    pub gap: f64,
}

pub fn run(s: &Setup) -> Result<Step, String> {
    for (path, text) in DATA {
        microscope::libs::add(path, text);
    }
    let out = output(&program(s), 14)?;
    let l = numbers::<f64>(&out[4], 2)?;
    Ok(Step {
        x: numbers(&out[0], N * I)?,
        y: numbers(&out[1], N * K)?,
        h: numbers(&out[2], N * J)?,
        p: numbers(&out[3], N * K)?,
        loss: (l[0], l[1]),
        d2: numbers(&out[5], N * K)?,
        g2: numbers(&out[6], (J + 1) * K)?,
        d1: numbers(&out[7], N * J)?,
        g1: numbers(&out[8], (I + 1) * J)?,
        v1: numbers(&out[9], (I + 1) * J)?,
        v2: numbers(&out[10], (J + 1) * K)?,
        f1: numbers(&out[11], (I + 1) * J)?,
        f2: numbers(&out[12], (J + 1) * K)?,
        gap: numbers::<f64>(&out[13], 1)?[0],
    })
}
