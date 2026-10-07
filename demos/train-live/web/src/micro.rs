//! The model: the demo's own X_eTaL program (train-live.xtl) run for a
//! number of training steps from a state (the weights and Adam's
//! averages, one vector). Nothing here knows about the browser.

use microscope::run::{lit, numbers, output, section};

/// The command-line program.
pub const SOURCE: &str = include_str!("../../train-live.xtl");

/// Weights (99) and Adam's two averages and step count: 3 * 99 + 1.
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
        Some(s) => format!("s := {} r_eshape {}", STATE, s.iter().map(|&v| lit(v)).collect::<Vec<_>>().join(" ")),
    };
    format!(
        "{start}\ns := {steps} 'u:a_dam p_ower s\n{}\n{}\nw := 99 t_ake s\ng := {SIDE}\n\
         c := -1.1 + 2.2 * (0.5 + f_loat o_ffsets g) / f_loat g\n\
         grid := o_\\ (2 c_at g * g) r_eshape (r_avel (o_ffsets g) 'r_ight t_able c) c_at r_avel (r_ev c) 'l_eft t_able o_ffsets g\n\
         r_avel s\n(u:l_oss w) c_at u:r_ight w\nnn:a_rgmax nn:s_oftmax (nn:t_anh grid nn:d_ense u:w_1 w) nn:d_ense u:w_2 w\nr_avel X\n1 + arm\n",
        line("u:l_oss"),
        line("u:r_ight")
    )
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
