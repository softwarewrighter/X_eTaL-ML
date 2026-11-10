//! The model: the demo's program (moe-finetune.xtl) up to its end of
//! the core, then one run's own lines: the base fine-tuned by one method
//! on one choice of tokens from one seed, and what it measured read
//! back. Nothing here knows about the browser.

use microscope::run::{numbers, output_with, section};

/// The command line's program; the page runs its core.
pub const PROGRAM: &str = include_str!("../../moe-finetune.xtl");

/// The data files the program reads, as the page hands them to X_eTaL.
pub const FILES: [(&str, &str); 5] = [
    ("data/base.txt", include_str!("../../data/base.txt")),
    ("data/train.txt", include_str!("../../data/train.txt")),
    ("data/test.txt", include_str!("../../data/test.txt")),
    ("data/shift-train.txt", include_str!("../../data/shift-train.txt")),
    ("data/shift-test.txt", include_str!("../../data/shift-test.txt")),
];

/// Steps of Adam for every method, as the command line takes.
pub const STEPS: usize = 120;

/// The methods, in the research's order, then a new expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Method {
    None,
    Shared,
    Experts,
    Both,
    Selected,
    Router,
    Added,
}

impl Method {
    pub const ALL: [Method; 7] = [Method::None, Method::Shared, Method::Experts, Method::Both, Method::Selected, Method::Router, Method::Added];

    /// Its name in the program.
    pub fn xetal(self) -> &'static str {
        match self {
            Method::None => "none",
            Method::Shared => "shared",
            Method::Experts => "experts",
            Method::Both => "both",
            Method::Selected => "selected",
            Method::Router => "router",
            Method::Added => "added",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Method::None => "None (the base)",
            Method::Shared => "Shared layer only",
            Method::Experts => "Every expert, router frozen",
            Method::Both => "Shared layer and experts",
            Method::Selected => "Selected experts",
            Method::Router => "Experts and the router",
            Method::Added => "A new expert",
        }
    }

    /// What it trains.
    pub fn trains(self) -> &'static str {
        match self {
            Method::None => "nothing",
            Method::Shared => "a LoRA adapter on the shared layer (the research's attention-only)",
            Method::Experts => "a LoRA adapter on each of the 4 experts; the router as it was",
            Method::Both => "the shared layer's adapter and every expert's",
            Method::Selected => "adapters on the two experts the base sends most new tokens to",
            Method::Router => "every expert's adapter and the router's 4 rows",
            Method::Added => "a fifth expert from 0 and its own router row, nothing of the base",
        }
    }
}

/// The tokens a method is fine-tuned on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Data {
    /// The 200 new tokens alone.
    New,
    /// Mixed with 200 of the old task's (replay).
    Mixed,
}

impl Data {
    pub fn xetal(self) -> &'static str {
        match self {
            Data::New => "(x, Y)",
            Data::Mixed => "mixed",
        }
    }
}

/// The program's core: everything up to its end-of-core marker.
pub fn head() -> &'static str {
    section(PROGRAM, "\"moe:\" u_se< \"MoE\"", "# -- end of the core")
}

/// One run's own lines: the method on the tokens from the seed, then
/// what the page reads back.
pub fn tail(method: Method, data: Data, seed: usize, steps: usize) -> String {
    let steps = if method == Method::None { 0 } else { steps };
    let m = method.xetal();
    format!(
        "(m, losses) := ({m}, {}, {steps}) u:t_une {seed}\n\
         (a, o, n, b, sn, so, sz) := m u:r_eport {m}\n\
         a c_at o c_at n c_at b\nsn\nso\nsz\n0.0 c_at losses\nbase u:s_hare xn\nbase u:s_hare xo\n",
        data.xetal()
    )
}

pub fn program(method: Method, data: Data, seed: usize, steps: usize) -> String {
    format!("{}{}", head(), tail(method, data, seed, steps))
}

/// What a run measured.
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    /// The new task's accuracy (held out) and the old task's.
    pub new: f64,
    pub old: f64,
    /// The numbers it trained, and their bytes in FP32.
    pub numbers: usize,
    pub bytes: usize,
    /// The share of the new and the old tokens each of the 5 experts
    /// gets after (each summing to 2), and before (the base).
    pub share_new: Vec<f64>,
    pub share_old: Vec<f64>,
    pub base_new: Vec<f64>,
    pub base_old: Vec<f64>,
    /// The update's size on each base expert, the shared layer, the new
    /// expert.
    pub sizes: Vec<f64>,
    /// The loss after each step.
    pub losses: Vec<f64>,
}

pub fn files() -> Vec<(String, String)> {
    FILES.iter().map(|(p, t)| (p.to_string(), t.to_string())).collect()
}

pub fn run(method: Method, data: Data, seed: usize, steps: usize) -> Result<Run, String> {
    let out = output_with(&files(), &program(method, data, seed, steps), 7)?.0;
    let first: Vec<f64> = numbers(&out[0], 4)?;
    let steps = if method == Method::None { 0 } else { steps };
    let losses: Vec<f64> = numbers(&out[4], steps + 1)?;
    Ok(Run {
        new: first[0],
        old: first[1],
        numbers: first[2] as usize,
        bytes: first[3] as usize,
        share_new: numbers(&out[1], 5)?,
        share_old: numbers(&out[2], 5)?,
        sizes: numbers(&out[3], 6)?,
        losses: losses[1..].to_vec(),
        base_new: numbers(&out[5], 5)?,
        base_old: numbers(&out[6], 5)?,
    })
}
