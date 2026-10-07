//! The model: the demo's own X_eTaL program (train-live.xtl) run for a
//! number of training steps from a state (seven boxed arrays: W1, W2,
//! Adam's averages per layer, the step count). Nothing here knows about
//! the browser.

use microscope::run::{lit, numbers, output, section};

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

/// The run's own lines: the state to start from (the program's s0, or
/// the page's), `steps` Adam steps, then what the page shows.
pub fn tail(state: Option<&[f64]>, steps: usize) -> String {
    let start = match state {
        None => "s := s0".to_string(),
        Some(s) => format!("s := {}", boxes(s)),
    };
    format!(
        "{start}\ns := {steps} 'u:a_dam p_ower s\n{}\n{}\nu:j_oin := {{ b -> d_isclose '{{ x y -> e_nclose (d_isclose x) c_at d_isclose y }} r_/ b }}\ng := {SIDE}\n\
         c := -1.1 + 2.2 * (0.5 + f_loat o_ffsets g) / f_loat g\n\
         grid := o_\\ (2 c_at g * g) r_eshape (r_avel (o_ffsets g) 'r_ight t_able c) c_at r_avel (r_ev c) 'l_eft t_able o_ffsets g\n\
         r_avel u:j_oin '{{ b -> r_avel d_isclose b }} m_ap s\n(u:l_oss s) c_at u:r_ight s\nnn:a_rgmax nn:s_oftmax (nn:t_anh grid nn:d_ense d_isclose 1 s_elect s) nn:d_ense d_isclose 2 s_elect s\nr_avel X\n1 + arm\n",
        line("u:l_oss"),
        line("u:r_ight")
    )
}

/// The state's numbers as X_eTaL: the seven arrays boxed and joined.
pub fn boxes(s: &[f64]) -> String {
    let mut at = 0;
    PARTS
        .iter()
        .map(|&(r, c)| {
            let part: Vec<String> = s[at..at + r * c].iter().map(|&v| lit(v)).collect();
            at += r * c;
            match (r, c) {
                (1, 1) => format!("(e_nclose {})", part[0]),
                _ => format!("(e_nclose {r} {c} r_eshape {})", part.join(" ")),
            }
        })
        .collect::<Vec<_>>()
        .join(" c_at ")
}

pub fn program(lr: f64, state: Option<&[f64]>, steps: usize) -> String {
    format!("{}{}", head(lr), tail(state, steps))
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
}

pub fn run(lr: f64, state: Option<&[f64]>, steps: usize) -> Result<After, String> {
    let out = output(&program(lr, state, steps), 5)?;
    let lr2 = numbers::<f64>(&out[1], 2)?;
    Ok(After {
        state: numbers(&out[0], STATE)?,
        loss: lr2[0],
        right: lr2[1],
        map: numbers(&out[2], SIDE * SIDE)?,
        points: numbers(&out[3], 2 * POINTS)?,
        arms: numbers(&out[4], POINTS)?,
    })
}
