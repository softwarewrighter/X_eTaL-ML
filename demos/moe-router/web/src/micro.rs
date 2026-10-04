//! The model: the demo's own X_eTaL program (moe-router.xtl): its
//! vocabulary and router weights, and its scores, softmax, top-2 and
//! load, run on a sentence the page picks. Nothing here knows about the
//! browser.

use microscope::run::{numbers, output, section};

/// The command-line program; the page runs its sections.
pub const SOURCE: &str = include_str!("../../moe-router.xtl");

/// The data the program reads, as the page's X_eTaL finds it: the
/// same path as at the command line.
pub const DATA: [(&str, &str); 1] = [("data/features.txt", include_str!("../../data/features.txt"))];

/// Run `src` with the data files in the store (see `output`).
fn out(src: &str, lines: usize) -> Result<Vec<String>, String> {
    for (path, text) in DATA {
        microscope::libs::add(path, text);
    }
    output(src, lines)
}

/// What both programs the page runs start with: the vocabulary's
/// features (read from data/), the embeddings, the router weights,
/// scores, softmax, top-2 and load.
pub fn head() -> String {
    format!("{}{}", tables(), core())
}

/// The vocabulary's features, the embeddings and the router weights.
pub fn tables() -> &'static str {
    section(SOURCE, "# -- the vocabulary", "# -- the scores")
}

/// The router weights (shown on the page; the vocabulary is long).
pub fn weights() -> &'static str {
    section(SOURCE, "# -- the router weights", "# -- the scores")
}

/// Scores, softmax, top-2, load.
pub fn core() -> &'static str {
    section(SOURCE, "# -- the scores", "# -- end of the core")
}

pub const EXPERTS: usize = 16;
pub const FEATURES: [&str; 8] = ["animal", "number", "colour", "action", "place", "food", "function", "time"];

/// The vocabulary, in order, from the program's `# words:` line; the
/// last, `?`, stands for any word not listed.
pub fn words() -> Vec<&'static str> {
    SOURCE.lines().find_map(|l| l.strip_prefix("# words: ")).map_or(vec![], |w| w.split_whitespace().collect())
}

/// Expert e (from 0) is on grid row e / 4 (it likes feature e / 4) and
/// column e % 4 (feature 4 + e % 4).
pub fn expert_name(e: usize) -> String {
    format!("{} \u{00d7} {}", FEATURES[e / 4], FEATURES[4 + e % 4])
}

pub const SENTENCES: &[&str] = &[
    "the red fox eats fish in the park",
    "two brown horses run to the river today",
    "a cat sees seven blue birds at night",
    "the dog sleeps at home and eats bread",
    "one green apple and three fish",
];

/// A sentence's tokens: its words in the vocabulary (lowercase, without
/// punctuation; a plural "s" dropped when only the singular is listed),
/// any other word as `?`. Returns the shown words and their numbers
/// (from 1).
pub fn tokens(sentence: &str) -> Vec<(String, usize)> {
    let vocab = words();
    let find = |w: &str| vocab.iter().position(|v| *v == w);
    sentence
        .split_whitespace()
        .map(|raw| {
            let w: String = raw.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase();
            let i = find(&w).or_else(|| w.strip_suffix('s').and_then(find)).unwrap_or(vocab.len() - 1);
            (w, i + 1)
        })
        .filter(|(w, _)| !w.is_empty())
        .collect()
}

/// Every array of the routing, as X_eTaL computed them.
#[derive(Clone, Debug, PartialEq)]
pub struct Anatomy {
    pub words: Vec<String>,
    /// n x 8: each token's embedding.
    pub x: Vec<f64>,
    /// n x 16: scores, probabilities, gates.
    pub scores: Vec<f64>,
    pub p: Vec<f64>,
    pub gates: Vec<f64>,
    /// 16: how many tokens each expert got.
    pub load: Vec<f64>,
}

impl Anatomy {
    pub fn n(&self) -> usize {
        self.words.len()
    }

    /// Token t's two experts, the first (larger gate) first.
    pub fn top2(&self, t: usize) -> [usize; 2] {
        let row = &self.gates[t * EXPERTS..(t + 1) * EXPERTS];
        let mut picked: Vec<usize> = (0..EXPERTS).filter(|&e| row[e] > 0.0).collect();
        picked.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
        [picked[0], picked.get(1).copied().unwrap_or(picked[0])]
    }
}

/// The program: the tables, the core, the sentence, and the arrays.
pub fn program(ids: &[usize]) -> String {
    let lits: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
    format!(
        "{}{}ids := {}\nx := ids s_elect E\np := u:s_oftmax u:s_cores x\ngates := u:t_op2 p\n\
         r_avel x\nr_avel u:s_cores x\nr_avel p\nr_avel gates\nr_avel u:l_oad gates\n",
        tables(),
        core(),
        lits.join(" ")
    )
}

/// Route `sentence` through X_eTaL.
pub fn run(sentence: &str) -> Result<Anatomy, String> {
    let toks = tokens(sentence);
    if toks.len() < 2 {
        return Err("type a sentence of at least two words".into());
    }
    let ids: Vec<usize> = toks.iter().map(|t| t.1).collect();
    let n = ids.len();
    let out = out(&program(&ids), 5)?;
    Ok(Anatomy {
        words: toks.into_iter().map(|t| t.0).collect(),
        x: numbers(&out[0], n * 8)?,
        scores: numbers(&out[1], n * EXPERTS)?,
        p: numbers(&out[2], n * EXPERTS)?,
        gates: numbers(&out[3], n * EXPERTS)?,
        load: numbers(&out[4], EXPERTS)?,
    })
}

/// The nudge: its settings (word w0 pushed towards wa; the slice also
/// towards wb; words numbered from 1) and the epsilon section.
pub fn nudge_core() -> &'static str {
    section(SOURCE, "# -- epsilon", "# -- end of the nudge")
}

/// Samples along epsilon (0 to 1.2) and the slice's side.
pub const STEPS: usize = 121;
pub const SIDE: usize = 40;

/// What the page nudges: words numbered from 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nudge {
    pub w0: usize,
    pub wa: usize,
    pub wb: usize,
}

impl Default for Nudge {
    fn default() -> Self {
        Nudge { w0: 19, wa: 9, wb: 32 }
    }
}

/// The nudge as X_eTaL computed it.
#[derive(Clone, Debug, PartialEq)]
pub struct Nudged {
    pub eps: Vec<f64>,
    /// Along epsilon: first and second expert (1 to 16), and the gates.
    pub first: Vec<f64>,
    pub second: Vec<f64>,
    pub gates: Vec<f64>,
    /// Over the slice: each point's first and second expert.
    pub slice_first: Vec<f64>,
    pub slice_second: Vec<f64>,
}

impl Nudged {
    /// The sample indices i where the pair changes between i and i + 1.
    pub fn changes(&self) -> Vec<usize> {
        (0..self.eps.len() - 1).filter(|&i| (self.first[i], self.second[i]) != (self.first[i + 1], self.second[i + 1])).collect()
    }
}

pub fn nudge_program(n: &Nudge) -> String {
    format!(
        "{}{}w0 := {}\nwa := {}\nwb := {}\nk := {STEPS}\nside := {SIDE}\n{}\
         r_avel eps\nr_avel u:f_irst gs\nr_avel u:s_econd gs\nr_avel gs\nr_avel u:f_irst gg\nr_avel u:s_econd gg\n",
        tables(),
        core(),
        n.w0,
        n.wa,
        n.wb,
        nudge_core()
    )
}

/// Run the nudge through X_eTaL.
pub fn run_nudge(n: &Nudge) -> Result<Nudged, String> {
    let out = out(&nudge_program(n), 6)?;
    Ok(Nudged {
        eps: numbers(&out[0], STEPS)?,
        first: numbers(&out[1], STEPS)?,
        second: numbers(&out[2], STEPS)?,
        gates: numbers(&out[3], STEPS * EXPERTS)?,
        slice_first: numbers(&out[4], SIDE * SIDE)?,
        slice_second: numbers(&out[5], SIDE * SIDE)?,
    })
}

/// The embedding table (37 x 8) and router weights (8 x 16), as X_eTaL
/// holds them (for the tests' direct computations).
pub fn tables_values() -> Result<(Vec<f64>, Vec<f64>), String> {
    let out = out(&format!("{}r_avel E\nr_avel W\n", tables()), 2)?;
    Ok((numbers(&out[0], 37 * 8)?, numbers(&out[1], 8 * EXPERTS)?))
}
