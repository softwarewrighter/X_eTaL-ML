//! The model: the demo's own X_eTaL program (icl-trained.xtl) run on a
//! prompt through one of the three trained transformers, with an edit
//! of its residual stream or none, every array read back. Nothing here
//! knows about the browser.

use microscope::run::{numbers, output_with, section};

/// The command-line program.
pub const SOURCE: &str = include_str!("../../icl-trained.xtl");
/// What the command line printed (its reg-rs baseline): the paper's
/// measures on each model, which the page shows as a table.
pub const CLI: &str = include_str!("../../reg/cli-icl-trained.out");

/// The three training mixes: the name the program loads, as the page
/// says it, and what it trained on.
pub const MODELS: [(&str, &str, &str); 3] = [
    ("mix", "Mix", "half unique-label prompts (the query's label shown once), half recurring"),
    ("recurring", "Recurring only", "recurring-label prompts only"),
    ("gate", "Gate", "half recurring, half unique-label prompts whose query label is fresh: retrieving never pays on them"),
];

macro_rules! weights {
    ($m:literal) => {
        [
            concat!("data/", $m, "/emb.txt"),
            include_str!(concat!("../../data/", $m, "/emb.txt")),
            concat!("data/", $m, "/pos.txt"),
            include_str!(concat!("../../data/", $m, "/pos.txt")),
            concat!("data/", $m, "/g1.txt"),
            include_str!(concat!("../../data/", $m, "/g1.txt")),
            concat!("data/", $m, "/g2.txt"),
            include_str!(concat!("../../data/", $m, "/g2.txt")),
            concat!("data/", $m, "/wq.txt"),
            include_str!(concat!("../../data/", $m, "/wq.txt")),
            concat!("data/", $m, "/wk.txt"),
            include_str!(concat!("../../data/", $m, "/wk.txt")),
            concat!("data/", $m, "/wv.txt"),
            include_str!(concat!("../../data/", $m, "/wv.txt")),
            concat!("data/", $m, "/wo.txt"),
            include_str!(concat!("../../data/", $m, "/wo.txt")),
            concat!("data/", $m, "/wg.txt"),
            include_str!(concat!("../../data/", $m, "/wg.txt")),
            concat!("data/", $m, "/wu.txt"),
            include_str!(concat!("../../data/", $m, "/wu.txt")),
            concat!("data/", $m, "/wd.txt"),
            include_str!(concat!("../../data/", $m, "/wd.txt")),
            concat!("data/", $m, "/gf.txt"),
            include_str!(concat!("../../data/", $m, "/gf.txt")),
            concat!("data/", $m, "/un.txt"),
            include_str!(concat!("../../data/", $m, "/un.txt")),
        ]
    };
}

const MIX: [&str; 26] = weights!("mix");
const RECURRING: [&str; 26] = weights!("recurring");
const GATE: [&str; 26] = weights!("gate");

/// The training logs (a line every so many steps: the step, the loss,
/// the mean margin on held-out unique, recurring and noise prompts).
pub const LOGS: [&str; 3] = [include_str!("../../data/mix/log.txt"), include_str!("../../data/recurring/log.txt"), include_str!("../../data/gate/log.txt")];

const PROMPTS: [(&str, &str); 4] = [
    ("data/unique.txt", include_str!("../../data/unique.txt")),
    ("data/recurring.txt", include_str!("../../data/recurring.txt")),
    ("data/flipped.txt", include_str!("../../data/flipped.txt")),
    ("data/words.txt", include_str!("../../data/words.txt")),
];

/// Every file the program reads, and the gate direction if one was found.
pub fn files(direction: Option<&[f64]>) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = vec![];
    for set in [&MIX, &RECURRING, &GATE] {
        for pair in set.chunks(2) {
            v.push((pair[0].to_string(), pair[1].to_string()));
        }
    }
    v.extend(PROMPTS.iter().map(|(a, b)| (a.to_string(), b.to_string())));
    if let Some(d) = direction {
        v.push(("data/direction.txt".into(), d.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")));
    }
    v
}

/// The classes, as the page names them.
pub const CLASS: [&str; 8] = ["A", "B", "C", "D", "E", "F", "G", "H"];

/// Each word's two classes (from 0).
pub fn words() -> Vec<(usize, usize)> {
    PROMPTS[3].1.lines().map(|l| {
        let v: Vec<usize> = l.split_whitespace().map(|x| x.parse::<usize>().unwrap() - 1).collect();
        (v[0], v[1])
    }).collect()
}

/// Which prompts: the unique-label ones or the recurring-label ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Set {
    Unique,
    Recurring,
}

impl Set {
    /// The program's name for it.
    pub fn name(self) -> &'static str {
        match self {
            Set::Unique => "uniq",
            Set::Recurring => "rec",
        }
    }
}

/// The prompts the page offers of each kind (the program's N).
pub const N: usize = 8;

/// Prompt k (from 1) of a set: its tokens, the answer last.
pub fn prompt(set: Set, k: usize) -> Vec<usize> {
    let text = if set == Set::Unique { PROMPTS[0].1 } else { PROMPTS[1].1 };
    let row: Vec<usize> = text.lines().nth(k - 1).unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
    row[1..=row[0]].to_vec()
}

/// A token as the page writes it: the start, an arrow, a line's end, a
/// word (w and its number), a label (L and its number).
pub fn token(t: usize) -> String {
    match t {
        1 => "start".into(),
        2 => "\u{2192}".into(),
        3 => "\u{21b5}".into(),
        4..=51 => format!("w{}", t - 4),
        _ => format!("L{}", t - 52),
    }
}

/// The class a sentence's three words share (each word's tokens).
pub fn class_of(ws: &[usize], cls: &[(usize, usize)]) -> Option<usize> {
    (0..8).find(|&c| ws.iter().all(|&w| w >= 4 && w < 52 && (cls[w - 4].0 == c || cls[w - 4].1 == c)))
}

/// An edit of the residual stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Edit {
    None,
    /// The answer's slot and another label's swapped after this layer
    /// (0 the embedding).
    Swap(usize),
    /// The gate direction (found in the browser) added at the arrow after
    /// this layer, times this many gaps (negative: toward unique).
    Steer(usize, f64),
    /// The same push in the program's fixed random direction.
    Random(usize, f64),
}

/// What the page runs first: the whole program up to the command line's
/// own runs (the models, the forward pass, the prompts, the measures,
/// the analysis).
pub fn head() -> &'static str {
    section(SOURCE, "\"act:\" u_se<", "### The three models, taken apart")
}

/// The lines for one prompt through one model with an edit.
pub fn tail(model: usize, set: Set, k: usize, e: Edit) -> String {
    let edit = match e {
        Edit::None => "e := none".to_string(),
        Edit::Swap(l) => format!("(a, b) := ids u:p_air ans\ne := ({l}, a, b, 9, D r_eshape 0.0)"),
        Edit::Steer(l, s) => format!("v := {} * n_umbers []N_GET \"data/direction.txt\"\ne := (9, 1, 1, {l}, v)", microscope::run::lit(s)),
        Edit::Random(l, s) => format!("e := (9, 1, 1, {l}, {} * rnd)", microscope::run::lit(s)),
    };
    format!(
        "Wt := u:l_oad \"{}\"\n(ids, ans) := u:p_rompt ({}, {k})\n{edit}\n(run, mg, alt, _) := u:m_easure (Wt, {}, {k}, e)\n(_, _, As, lg, _) := run\nmg\nalt c_at ans\nlg\nWt u:d_la (run, mg, alt, ans)\nr_avel (d_isclose 1 s_elect As) c_at (d_isclose 2 s_elect As) c_at (d_isclose 3 s_elect As) c_at d_isclose 4 s_elect As\n",
        MODELS[model].0,
        set.name(),
        set.name()
    )
}

/// The lines that find a model's gate direction.
pub fn gate_tail(model: usize) -> String {
    format!(
        "Wt := u:l_oad \"{}\"\n(mu, mr, Bu, Br, fl) := u:i_ntact Wt\nmu\nmr\nu:c_mean fl\n(d1, gap1, acc1) := (Bu, Br) u:g_ate 1\n(d2, gap2, acc2) := (Bu, Br) u:g_ate 2\n(d3, gap3, acc3) := (Bu, Br) u:g_ate 3\ngap1 c_at acc1 c_at gap2 c_at acc2 c_at gap3 c_at acc3\nd1\nd2\nd3\n",
        MODELS[model].0
    )
}

/// One prompt through a model, as X_eTaL computed it.
#[derive(Clone, Debug, PartialEq)]
pub struct Anatomy {
    pub model: usize,
    pub set: Set,
    pub k: usize,
    pub edit: Edit,
    /// The tokens without the answer, and the answer.
    pub ids: Vec<usize>,
    pub answer: usize,
    pub margin: f64,
    /// The best other label shown.
    pub alt: usize,
    /// The logits at the query's arrow (84).
    pub logits: Vec<f64>,
    /// Each head's direct contribution to the margin (16, layer by layer).
    pub dla: Vec<f64>,
    /// Each head's attention from the arrow (16 x n).
    pub looks: Vec<f64>,
}

impl Anatomy {
    pub fn n(&self) -> usize {
        self.ids.len()
    }
    /// The head contributing most, (layer, head) from 1.
    pub fn top(&self) -> (usize, usize) {
        let i = (0..16).fold(0, |b, i| if self.dla[i] > self.dla[b] { i } else { b });
        (i / 4 + 1, i % 4 + 1)
    }
    /// Head (l, h)'s attention from the arrow.
    pub fn look(&self, l: usize, h: usize) -> &[f64] {
        let i = (l - 1) * 4 + h - 1;
        &self.looks[i * self.n()..(i + 1) * self.n()]
    }
}

/// Run a prompt (the gate direction given, for a steer).
pub fn run(model: usize, set: Set, k: usize, e: Edit, direction: Option<&[f64]>) -> Result<Anatomy, String> {
    let src = format!("{}{}", head(), tail(model, set, k, e));
    let (out, _) = output_with(&files(direction), &src, 5)?;
    let full = prompt(set, k);
    let ids = full[..full.len() - 1].to_vec();
    let n = ids.len();
    let pair = numbers::<f64>(&out[1], 2)?;
    Ok(Anatomy {
        model,
        set,
        k,
        edit: e,
        answer: pair[1] as usize,
        alt: pair[0] as usize,
        margin: numbers::<f64>(&out[0], 1)?[0],
        logits: numbers(&out[2], 84)?,
        dla: numbers(&out[3], 16)?,
        looks: numbers(&out[4], 16 * n)?,
        ids,
    })
}

/// A model's gate, as X_eTaL found it on N prompts of each kind.
#[derive(Clone, Debug, PartialEq)]
pub struct Gate {
    pub unique: Vec<f64>,
    pub recurring: Vec<f64>,
    /// Each head's class-flip attribution (16).
    pub flip: Vec<f64>,
    /// After layers 1 to 3: the held-out gap and sign accuracy.
    pub gaps: Vec<(f64, f64)>,
    /// The direction after layers 1 to 3 (64 each).
    pub directions: Vec<Vec<f64>>,
}

impl Gate {
    /// The layer of the largest gap (1 to 3).
    pub fn layer(&self) -> usize {
        (0..3).fold(0, |b, i| if self.gaps[i].0 > self.gaps[b].0 { i } else { b }) + 1
    }
}

pub fn gate(model: usize) -> Result<Gate, String> {
    let src = format!("{}{}", head(), gate_tail(model));
    let (out, _) = output_with(&files(None), &src, 7)?;
    let g = numbers::<f64>(&out[3], 6)?;
    Ok(Gate {
        unique: numbers(&out[0], N)?,
        recurring: numbers(&out[1], N)?,
        flip: numbers(&out[2], 16)?,
        gaps: (0..3).map(|i| (g[2 * i], g[2 * i + 1])).collect(),
        directions: (4..7).map(|i| numbers(&out[i], 64)).collect::<Result<_, _>>()?,
    })
}

/// The command line's measures of each model (mix, recurring only,
/// gate), as it printed them.
#[derive(Clone, Debug, PartialEq)]
pub struct Measured {
    pub margins: (f64, f64),
    /// The two heads the class flip moves most: (layer, head, attribution).
    pub readers: Vec<(usize, usize, f64)>,
    /// After layers 1 to 3: (gap, sign accuracy).
    pub gaps: Vec<(f64, f64)>,
    pub layer: usize,
    /// Recurring toward unique, its random control, unique toward
    /// recurring, its random control.
    pub steer: [f64; 4],
    /// The label swapped in against the answer: with no swap, then at the
    /// embedding and after layers 1 to 3.
    pub swap: Vec<f64>,
}

pub fn measured() -> Vec<Measured> {
    let lines: Vec<&str> = CLI.lines().collect();
    let num = |l: &str| l.split_whitespace().map(|x| x.parse::<f64>().unwrap()).collect::<Vec<f64>>();
    lines
        .chunks(6)
        .map(|c| {
            let (m, h, a, g, s, w) = (num(c[0]), num(c[1]), num(c[2]), num(c[3]), num(c[4]), num(c[5]));
            Measured {
                margins: (m[0], m[1]),
                readers: (0..2).map(|i| (h[i] as usize, h[2 + i] as usize, a[i])).collect(),
                gaps: (0..3).map(|i| (g[2 * i], g[2 * i + 1])).collect(),
                layer: s[0] as usize,
                steer: [s[1], s[2], s[3], s[4]],
                swap: w,
            }
        })
        .collect()
}

/// A training log: (step, unique margin, recurring margin, noise margin).
pub fn log(model: usize) -> Vec<(f64, f64, f64, f64)> {
    LOGS[model]
        .lines()
        .filter_map(|l| {
            let v: Vec<&str> = l.split_whitespace().collect();
            (v.len() >= 8).then(|| (v[0].parse().unwrap(), v[3].parse().unwrap(), v[5].parse().unwrap(), v[7].parse().unwrap()))
        })
        .collect()
}
