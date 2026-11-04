//! The model: the demo's program (lora-explorer.xtl) run at the rank
//! and sizes the page sets, then its pipeline drawn as math, every
//! stage's value, and the arrays W, A, B, the update and the merged
//! matrix read back. Nothing here knows about the browser.

use microscope::run::{numbers, output, section};
pub use microscope::stages::{values, Value};

/// The command line's program, which the page runs with its settings.
pub const PROGRAM: &str = include_str!("../../lora-explorer.xtl");

/// What the page sets: the rank (1..8), and the large model's sizes
/// for the costs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub r: usize,
    pub d: usize,
    pub f: usize,
    pub experts: usize,
    pub layers: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { r: 3, d: 4096, f: 14336, experts: 64, layers: 24 }
    }
}

/// The program with the settings put in its data.
pub fn with(s: Settings) -> String {
    [("r := 3\n", format!("r := {}\n", s.r)), ("d := 4096\n", format!("d := {}\n", s.d)), ("f := 14336\n", format!("f := {}\n", s.f)), ("experts := 64\n", format!("experts := {}\n", s.experts)), ("layers := 24\n", format!("layers := {}\n", s.layers))]
        .iter()
        .fold(PROGRAM.to_string(), |p, (from, to)| p.replacen(from, to, 1))
}

/// The pipeline's text, as the program writes it after `flow:p_ipe<`.
pub fn pipeline() -> &'static str {
    let p = section(PROGRAM, "y := @ flow:p_ipe<", "\n## ");
    p.trim_start_matches("y := @ flow:p_ipe<").trim_end()
}

/// The program the page runs: the program, then what it reads back: the
/// pipeline drawn as math, every stage's value (flow:s_tages<), and W,
/// A, B, the update and the merged matrix, each ravelled.
pub fn program(s: Settings) -> String {
    format!(
        "{}\n# -- read back by the page ------------------------------\ndrawn\n@ flow:s_tages<{}\nr_avel W\nr_avel A\nr_avel B\nr_avel delta\nr_avel merged\n",
        with(s).trim_end(),
        pipeline()
    )
}

/// How many lines the program itself prints, then the page's.
const PRINTED: usize = 7;
const READ: usize = 7;

/// A matrix read back: its rows, its columns, its items by row.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub items: Vec<f64>,
}

/// What a run gave.
#[derive(Clone, Debug, PartialEq)]
pub struct Explored {
    /// The adapted layer's output, 8 numbers.
    pub y: Vec<f64>,
    /// The update's rank, and whether the merged output is the factored one.
    pub rank: usize,
    pub merged_same: bool,
    /// The numbers trained, and W's; the multiply-adds factored and to make B A.
    pub trained: (usize, usize),
    pub costs: (usize, usize),
    /// On the large model: one expert's adapters' numbers, and their MiB
    /// in BF16 for one expert, a layer of experts, every layer.
    pub expert: u64,
    pub mib: (f64, f64, f64),
    /// The pipeline as LaTeX, and every stage's value.
    pub tex: String,
    pub stages: Vec<Value>,
    pub w: Matrix,
    pub a: Matrix,
    pub b: Matrix,
    pub delta: Matrix,
    pub merged: Matrix,
}

/// Run the program at settings `s`.
pub fn run(s: Settings) -> Result<Explored, String> {
    let out = output(&program(s), PRINTED + READ)?;
    let two = |line: &str| numbers::<usize>(line, 2).map(|v| (v[0], v[1]));
    let m = |line: &str, rows: usize, cols: usize| numbers::<f64>(line, rows * cols).map(|items| Matrix { rows, cols, items });
    let mib = numbers::<f64>(&out[6], 3)?;
    Ok(Explored {
        y: numbers(&out[0], 8)?,
        rank: out[1].trim().parse().map_err(|_| format!("not a rank: {}", out[1]))?,
        merged_same: out[2].trim() == "1",
        trained: two(&out[3])?,
        costs: two(&out[4])?,
        expert: out[5].trim().parse().map_err(|_| format!("not a count: {}", out[5]))?,
        mib: (mib[0], mib[1], mib[2]),
        tex: out[7].clone(),
        stages: values(&out[8])?,
        w: m(&out[9], 8, 8)?,
        a: m(&out[10], s.r, 8)?,
        b: m(&out[11], 8, s.r)?,
        delta: m(&out[12], 8, 8)?,
        merged: m(&out[13], 8, 8)?,
    })
}
