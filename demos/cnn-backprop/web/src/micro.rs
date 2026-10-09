//! The model: the demo's own X_eTaL program (cnn-backprop.xtl) run for
//! a number of training steps from a state (a tuple of ten arrays: the
//! filters K, their biases B, the dense weights W, Adam's averages of
//! each, the step count). The page keeps the state's parts in files the
//! program reads, state/K.txt and so on, as it reads its digits from
//! data/. Nothing here knows about the browser.

use microscope::run::{lit, numbers, output_with, section};

/// The command-line program.
pub const SOURCE: &str = include_str!("../../cnn-backprop.xtl");

/// The digits the program reads, as the page's X_eTaL finds them.
pub const DATA: [(&str, &str); 2] = [("data/train.txt", include_str!("../../data/train.txt")), ("data/test.txt", include_str!("../../data/test.txt"))];

/// The state's parts in order, with their shapes (rows, columns; a
/// vector is one row, the step count one number).
pub const PARTS: [(&str, usize, usize); 10] = [
    ("K", 8, 9),
    ("B", 1, 8),
    ("W", 1353, 10),
    ("MK", 8, 9),
    ("MB", 1, 8),
    ("MW", 1353, 10),
    ("VK", 8, 9),
    ("VB", 1, 8),
    ("VW", 1353, 10),
    ("t", 1, 1),
];

/// The test digits each run reads (the first ones of data/test.txt).
pub const SHOWN: usize = 20;

/// The program's head: the digits, the network, its gradient, Adam, the
/// starting state and the test.
pub fn head() -> &'static str {
    section(SOURCE, "\"nn:\" u_se< \"NN\"", "# -- end of the core")
}

/// The state read back from the files the page keeps.
fn start(trained: bool) -> String {
    if !trained {
        return "s := s0".into();
    }
    let part = |&(name, r, c): &(&str, usize, usize)| match (r, c) {
        (1, 1) => format!("f_irst n_umbers []N_GET \"state/{name}.txt\""),
        (1, _) => format!("n_umbers []N_GET \"state/{name}.txt\""),
        _ => format!("{r} {c} r_eshape n_umbers []N_GET \"state/{name}.txt\""),
    };
    format!("s := ({})", PARTS.iter().map(part).collect::<Vec<_>>().join(", "))
}

/// Earlier values and this run's, as X_eTaL: `(v1 v2 ...) c_at NOW`.
fn history(earlier: &[f64], now: &str) -> String {
    match earlier.is_empty() {
        true => now.to_string(),
        false => format!("({}) c_at {now}", earlier.iter().map(|&v| lit(v)).collect::<Vec<_>>().join(" ")),
    }
}

/// A run's own lines: the state to start from, `steps` training steps,
/// the state's parts written out (the page keeps them), the test loss
/// and share right, what the network reads in the first test digits,
/// then the test loss and share right after every run so far
/// (`earlier`) drawn by Plot.
pub fn tail(trained: bool, steps: usize, earlier: &[(f64, f64)]) -> String {
    let names: Vec<&str> = PARTS.iter().map(|p| p.0).collect();
    let out: String = PARTS.iter().map(|&(n, r, c)| if r * c == 1 || r == 1 { format!("{n}\n") } else { format!("r_avel {n}\n") }).collect();
    format!(
        "{}\ns := {steps} 'u:s_tep p_ower s\n({}) := s\n{out}\
         now := u:t_est s\nnow\n(nn:a_rgmax (K, B, W) u:p_robs {SHOWN} t_ake TX) - 1\n\
         \"p:\" u_se< \"Plot\"\nlosses := {}\nright := {}\n\
         chart := (\"Training\" \"run ({steps} steps each)\" \"\" \"test loss\" \"test share right\") p:c_hart! (e_nclose losses) c_at e_nclose right\n",
        start(trained),
        names.join(", "),
        history(&earlier.iter().map(|e| e.0).collect::<Vec<_>>(), "1 t_ake now"),
        history(&earlier.iter().map(|e| e.1).collect::<Vec<_>>(), "-1 t_ake now"),
    )
}

pub fn program(trained: bool, steps: usize, earlier: &[(f64, f64)]) -> String {
    format!("{}{}", head(), tail(trained, steps, earlier))
}

/// After a run of steps, as X_eTaL computed it.
#[derive(Clone, Debug, PartialEq)]
pub struct After {
    /// The state's ten parts, each raveled.
    pub state: Vec<Vec<f64>>,
    pub loss: f64,
    pub right: f64,
    /// The digit (0 to 9) read in each of the first SHOWN test digits.
    pub reads: Vec<usize>,
    /// Plot's chart of the test loss and share right so far, SVG.
    pub pictures: Vec<String>,
}

/// The state's parts as the files the program reads, state/K.txt and so on.
fn files(state: &[Vec<f64>]) -> Vec<(String, String)> {
    PARTS.iter().zip(state).map(|(&(name, _, _), v)| (format!("state/{name}.txt"), v.iter().map(|&x| lit(x)).collect::<Vec<_>>().join(" "))).collect()
}

pub fn run(state: Option<&[Vec<f64>]>, steps: usize, earlier: &[(f64, f64)]) -> Result<After, String> {
    for (path, text) in DATA {
        microscope::libs::add(path, text);
    }
    let (out, pictures) = output_with(&state.map(files).unwrap_or_default(), &program(state.is_some(), steps, earlier), PARTS.len() + 2)?;
    let parts = PARTS.iter().enumerate().map(|(i, &(_, r, c))| numbers(&out[i], r * c)).collect::<Result<Vec<_>, _>>()?;
    let now = numbers::<f64>(&out[PARTS.len()], 2)?;
    Ok(After { state: parts, loss: now[0], right: now[1], reads: numbers(&out[PARTS.len() + 1], SHOWN)?, pictures })
}

/// The first SHOWN test digits from data/test.txt: each its label and
/// its 784 pixels (0 to 255).
pub fn test_digits() -> Vec<(usize, Vec<f64>)> {
    DATA[1]
        .1
        .lines()
        .take(SHOWN)
        .map(|l| {
            let v: Vec<f64> = l.split_whitespace().map(|t| t.parse().unwrap_or(0.0)).collect();
            (v[0] as usize, v[1..].to_vec())
        })
        .collect()
}
