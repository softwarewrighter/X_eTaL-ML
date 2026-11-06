//! The model: the demo's program (lora-explorer.xtl) run at the rank
//! and sizes the page sets, up to its rank sweep, then its two
//! pipelines (the forward pass and a backward step) drawn as math,
//! every stage's value, and the arrays read back: W, A, B, the update,
//! the merged matrix, the known update and what LoRA and LoRA-FA
//! learned. The rank sweep is the whole program, run once. Nothing here
//! knows about the browser.

use microscope::run::{numbers, output, printed, section};
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
    /// The quantization's block (8, 16, 32 or 64 of W's 64 numbers).
    pub qk: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { r: 3, d: 4096, f: 14336, experts: 64, layers: 24, qk: 16 }
    }
}

/// The program with the settings put in its data.
pub fn with(s: Settings) -> String {
    [("r := 3\n", format!("r := {}\n", s.r)), ("d := 4096\n", format!("d := {}\n", s.d)), ("f := 14336\n", format!("f := {}\n", s.f)), ("experts := 64\n", format!("experts := {}\n", s.experts)), ("layers := 24\n", format!("layers := {}\n", s.layers)), ("qk := 16\n", format!("qk := {}\n", s.qk))]
        .iter()
        .fold(PROGRAM.to_string(), |p, (from, to)| p.replacen(from, to, 1))
}

/// A pipeline's text, as the program writes it after `NAME := @ flow:p_ipe<`.
fn text(name: &str) -> &'static str {
    let start = format!("{name} := @ flow:p_ipe<");
    section(PROGRAM, &start, "\n## ").trim_start_matches(start.as_str()).trim_end()
}

/// The forward pass's pipeline, then the backward step's.
pub fn pipeline() -> &'static str {
    text("y")
}
pub fn backward() -> &'static str {
    text("back")
}

/// Where the page's run stops: the rank sweep after it runs once, apart.
pub const END: &str = "# -- end of the explorer at rank r";

/// The program the page runs at each setting: the program up to the rank
/// sweep, then what it reads back: the two pipelines drawn as math and
/// every stage's value (flow:s_tages<), and W, A, B, the update, the
/// merged matrix, the known update T and the two learned updates, each
/// ravelled.
pub fn program(s: Settings) -> String {
    let p = with(s);
    format!(
        "{}\n# -- read back by the page ------------------------------\ndrawn\n@ flow:s_tages<{}\nbackDrawn\n@ flow:s_tages<{}\nr_avel W\nr_avel A\nr_avel B\nr_avel delta\nr_avel merged\nr_avel T\nr_avel learned\nr_avel learnedFA\nr_avel Wu\nr_avel Wn\n",
        p[..p.find(END).expect("the end marker")].trim_end(),
        pipeline(),
        backward()
    )
}

/// The rank sweep's run: the whole program, at the defaults.
pub fn sweep_program() -> &'static str {
    PROGRAM
}

/// How many lines the program prints up to the sweep, then the page's.
const PRINTED: usize = 18;
const READ: usize = 14;

/// A matrix read back (the shared heat map's).
pub use microscope::heat::Matrix;

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
    /// The backward step: whether its dx is the library's, its pipeline
    /// as LaTeX, its stages (g, b, dA, dB, dx).
    pub back_same: bool,
    pub back_tex: String,
    pub back_stages: Vec<Value>,
    /// Training: the loss after every 10 steps (the start first) of LoRA
    /// and of LoRA-FA; the best loss any update of rank k can reach, for
    /// k = 1 .. 8; the known update and the two learned.
    pub loss: Vec<f64>,
    pub loss_fa: Vec<f64>,
    pub best: Vec<f64>,
    pub t: Matrix,
    pub learned: Matrix,
    pub learned_fa: Matrix,
    /// W in 4 bits: the RMSE of uniform 4 bits and of NF4 against W,
    /// then against a normal matrix; the layer's output's largest change
    /// on each base; bits a weight (single, double-quantized); a 4096 x
    /// 14336 projection's MiB (FP32, FP16, NF4, NF4 double); W through
    /// each.
    pub rmse: [f64; 4],
    pub moved: (f64, f64),
    pub bits: (f64, f64),
    pub big_mib: [f64; 4],
    pub w_uniform: Matrix,
    pub w_nf4: Matrix,
    /// QLoRA: the loss every 10 steps of the adapter on the NF4 base, the
    /// best its rank allows for T less the 4-bit error, and the merge
    /// into 4 bits: its largest miss of W~ + s B A, and of the output.
    pub loss_q: Vec<f64>,
    pub best_q: f64,
    pub merge_miss: (f64, f64),
    pub w: Matrix,
    pub a: Matrix,
    pub b: Matrix,
    pub delta: Matrix,
    pub merged: Matrix,
}

/// The rank sweep: the loss LoRA and LoRA-FA reach at each rank 1 .. 8.
#[derive(Clone, Debug, PartialEq)]
pub struct Sweep {
    pub lora: Vec<f64>,
    pub fa: Vec<f64>,
}

/// Run the whole program for its last two lines, the sweep.
pub fn sweep() -> Result<Sweep, String> {
    let out = printed(sweep_program())?;
    match out.len() {
        20 => Ok(Sweep { lora: numbers(&out[18], 8)?, fa: numbers(&out[19], 8)? }),
        n => Err(format!("expected 20 lines of output, got {n}")),
    }
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
        back_same: out[7].trim() == "1",
        loss: numbers(&out[8], 31)?,
        loss_fa: numbers(&out[9], 31)?,
        best: numbers(&out[10], 8)?,
        tex: out[18].clone(),
        stages: values(&out[19])?,
        back_tex: out[20].clone(),
        back_stages: values(&out[21])?,
        w: m(&out[22], 8, 8)?,
        a: m(&out[23], s.r, 8)?,
        b: m(&out[24], 8, s.r)?,
        delta: m(&out[25], 8, 8)?,
        merged: m(&out[26], 8, 8)?,
        t: m(&out[27], 8, 8)?,
        learned: m(&out[28], 8, 8)?,
        learned_fa: m(&out[29], 8, 8)?,
        rmse: { let v = numbers::<f64>(&out[11], 4)?; [v[0], v[1], v[2], v[3]] },
        moved: { let v = numbers::<f64>(&out[12], 2)?; (v[0], v[1]) },
        bits: { let v = numbers::<f64>(&out[16], 2)?; (v[0], v[1]) },
        big_mib: { let v = numbers::<f64>(&out[17], 4)?; [v[0], v[1], v[2], v[3]] },
        w_uniform: m(&out[30], 8, 8)?,
        w_nf4: m(&out[31], 8, 8)?,
        loss_q: numbers(&out[13], 31)?,
        best_q: out[14].trim().parse().map_err(|_| format!("not a loss: {}", out[14]))?,
        merge_miss: { let v = numbers::<f64>(&out[15], 2)?; (v[0], v[1]) },
    })
}
