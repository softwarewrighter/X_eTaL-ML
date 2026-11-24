//! The model: the demo's own X_eTaL program (icl-circuit.xtl) run on a
//! prompt, with the circuit's parts kept or knocked out. Nothing here
//! knows about the browser.

use microscope::run::{numbers, output, section};

/// The command-line program.
pub const SOURCE: &str = include_str!("../../icl-circuit.xtl");

/// The eight classes a demonstration's sentence can have (the toy's
/// class numbers 0 to 7, named so a prompt reads).
pub const CLASSES: [&str; 8] = ["animal", "color", "food", "tool", "place", "number", "plant", "weather"];

/// The sixteen labels (the toy's label numbers 0 to 15): made-up words,
/// as the paper's foo, bar, quil and zorp, so no label means anything.
pub const LABELS: [&str; 16] = ["foo", "bar", "quil", "zorp", "baz", "qux", "mip", "tov", "rel", "dax", "wug", "blick", "fep", "lorn", "kib", "sarn"];

/// The most demonstrations a prompt may have (each is three tokens; the
/// position block has 128).
pub const MOST: usize = 12;

/// The widths of the named blocks the page reads back, in the residual
/// stream's order with the position block left out: class, label, type,
/// duplicate flag, gate, slot key, query code.
pub const BLOCKS: [(&str, usize); 7] = [("class", 8), ("label", 16), ("type", 4), ("dup", 1), ("gate", 1), ("slot key", 8), ("query code", 8)];

/// A prompt: each demonstration's class and label, and the query's class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prompt {
    pub classes: Vec<usize>,
    pub labels: Vec<usize>,
    pub query: usize,
}

impl Prompt {
    pub fn new(classes: &[usize], labels: &[usize], query: usize) -> Self {
        Prompt { classes: classes.to_vec(), labels: labels.to_vec(), query }
    }

    /// As the page writes it: `animal foo, color bar ? animal`.
    pub fn text(&self) -> String {
        let demos: Vec<String> = self.classes.iter().zip(&self.labels).map(|(&c, &l)| format!("{} {}", CLASSES[c], LABELS[l])).collect();
        format!("{} ? {}", demos.join(", "), CLASSES[self.query])
    }

    /// Read a prompt as the page writes it.
    pub fn parse(text: &str) -> Result<Prompt, String> {
        let (demos, query) = text.split_once('?').ok_or("end the prompt with ? and the query's class, as in \u{201c}animal foo, color bar ? animal\u{201d}")?;
        let class = |w: &str| CLASSES.iter().position(|c| *c == w).ok_or(format!("\u{201c}{w}\u{201d} is not a class: {}", CLASSES.join(", ")));
        let label = |w: &str| LABELS.iter().position(|l| *l == w).ok_or(format!("\u{201c}{w}\u{201d} is not a label: {}", LABELS.join(", ")));
        let (mut classes, mut labels) = (vec![], vec![]);
        for d in demos.split(',').map(str::trim).filter(|d| !d.is_empty()) {
            let words: Vec<String> = d.split_whitespace().map(str::to_lowercase).collect();
            if words.len() != 2 {
                return Err(format!("\u{201c}{d}\u{201d}: a demonstration is a class and a label"));
            }
            classes.push(class(&words[0])?);
            labels.push(label(&words[1])?);
        }
        let query = class(&query.trim().to_lowercase())?;
        Prompt { classes, labels, query }.checked()
    }

    /// The prompt, if the circuit can answer it.
    pub fn checked(self) -> Result<Prompt, String> {
        if self.classes.len() < 2 || self.classes.len() > MOST {
            return Err(format!("give 2 to {MOST} demonstrations"));
        }
        if !self.classes.contains(&self.query) {
            return Err(format!("no demonstration has the query's class, {}: there is nothing to retrieve", CLASSES[self.query]));
        }
        Ok(self)
    }

    /// The answer: the label of the first demonstration of the query's class.
    pub fn target(&self) -> usize {
        self.labels[self.classes.iter().position(|&c| c == self.query).unwrap()]
    }

    /// Each token as (type, class, label), as the program's p_rompt makes
    /// them: 0 the start, 1 a sentence, 2 an arrow, 3 a label.
    pub fn tokens(&self) -> Vec<(usize, usize, usize)> {
        let mut t = vec![(0, 0, 0)];
        for (&c, &l) in self.classes.iter().zip(&self.labels) {
            t.extend([(1, c, 0), (2, 0, 0), (3, 0, l)]);
        }
        t.extend([(1, self.query, 0), (2, 0, 0)]);
        t
    }

    /// The labels used, in first-use order.
    pub fn used(&self) -> Vec<usize> {
        let mut u: Vec<usize> = vec![];
        for &l in &self.labels {
            if !u.contains(&l) {
                u.push(l);
            }
        }
        u
    }

    /// A label shown again later in the prompt.
    pub fn repeats(&self) -> bool {
        self.used().len() < self.labels.len()
    }

    /// As X_eTaL's tuple (classes, labels, query).
    fn literal(&self) -> String {
        let list = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
        format!("({}, {}, {})", list(&self.classes), list(&self.labels), self.query)
    }
}

/// The paper's two prompts (its Section 12), and a pair of twins that
/// differ in one label (its slotgate battery's idea).
pub fn presets() -> Vec<(&'static str, Prompt)> {
    vec![
        ("Unique labels", Prompt::new(&[0, 1, 2, 3, 4, 5], &[0, 1, 2, 3, 4, 5], 0)),
        ("Recurring labels", Prompt::new(&[0, 1, 2, 0, 1, 2], &[0, 1, 2, 0, 1, 2], 0)),
        ("Twins: unique", Prompt::new(&[0, 1, 0, 1], &[0, 1, 2, 3], 0)),
        ("Twins: recurring", Prompt::new(&[0, 1, 0, 1], &[0, 1, 0, 1], 0)),
    ]
}

/// Which parts run, and how the readers read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Config {
    pub writers: bool,
    pub dup: bool,
    pub pool: bool,
    /// The gate forced open at the query (the paper's "+g").
    pub gate_added: bool,
    /// The gate kept out of the readers' queries.
    pub gate_out: bool,
    pub readers: usize,
    /// Each reader copies at a fifth of the strength and the readout is
    /// normalized against unit noise (the paper's gain bound).
    pub capped: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config { writers: true, dup: true, pool: true, gate_added: false, gate_out: false, readers: 1, capped: false }
    }
}

impl Config {
    /// As X_eTaL's tuple (kw, kd, kp, gadd, gblock, H, ov, noise).
    pub fn literal(&self) -> String {
        let f = |b: bool| if b { "1.0" } else { "0.0" };
        let (ov, noise) = if self.capped { ("0.2", "1.0") } else { ("1.0", "0.0") };
        format!("({}, {}, {}, {}, {}, {}.0, {ov}, {noise})", f(self.writers), f(self.dup), f(self.pool), f(self.gate_added), f(self.gate_out), self.readers)
    }
}

/// What the page's program starts with: the imports, the blocks, the
/// prompt, one head, the circuit and its readout (the CLI's core).
pub fn head() -> &'static str {
    section(SOURCE, "\"nn:\" u_se< \"NN\"", "# -- end of the core")
}

/// The run's own lines, after the head.
pub fn tail(p: &Prompt, c: &Config) -> String {
    format!(
        "pr := {}\ncfg := {}\n(X, A, l) := cfg u:r_un u:p_rompt pr\n\
         r_avel (f_irst s_hape A) s_elect A\nl\npr u:m_argin l\npr u:r_ead (X, A)\n\
         r_avel X u:m_m (oC u:b_lock 24) c_at_2 oT u:b_lock 22\n",
        p.literal(),
        c.literal()
    )
}

/// The whole program the page runs.
pub fn program(p: &Prompt, c: &Config) -> String {
    format!("{}{}", head(), tail(p, c))
}

/// What the page shows: all of the program, the run's lines marked.
pub fn shown(p: &Prompt, c: &Config) -> String {
    format!("{}# -- the run, for your prompt -------------------------\n{}", head(), tail(p, c))
}

/// One prompt through the circuit, as X_eTaL computed it.
#[derive(Clone, Debug, PartialEq)]
pub struct Anatomy {
    pub prompt: Prompt,
    pub config: Config,
    /// Each token: (type, class, label).
    pub tokens: Vec<(usize, usize, usize)>,
    /// The readers' attention at the query, one weight a token.
    pub attention: Vec<f64>,
    /// Each label's logit at the query.
    pub logits: Vec<f64>,
    pub margin: f64,
    /// The attention on the sink, the matching slots and the other slots.
    pub sink: f64,
    pub matching: f64,
    pub other: f64,
    /// The gate at the query.
    pub gate: f64,
    /// The read law's matching mass (Proposition 1).
    pub law: f64,
    /// Each token's named blocks (BLOCKS' widths, 46 a token).
    pub blocks: Vec<f64>,
}

impl Anatomy {
    pub fn n(&self) -> usize {
        self.tokens.len()
    }

    /// Token i's block b (an index into BLOCKS).
    pub fn block(&self, i: usize, b: usize) -> &[f64] {
        let at: usize = BLOCKS[..b].iter().map(|x| x.1).sum();
        &self.blocks[i * 46 + at..i * 46 + at + BLOCKS[b].1]
    }

    /// The slot a label token reads from, matching the query's class.
    pub fn matches(&self, i: usize) -> bool {
        self.tokens[i].0 == 3 && self.tokens[i - 2].1 == self.prompt.query
    }
}

/// The largest item's place and value, if there is one above a floor.
pub fn strongest(v: &[f64]) -> Option<(usize, f64)> {
    let (i, x) = v.iter().enumerate().fold((0, f64::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b });
    (x > 0.05).then_some((i, x))
}

/// Run a prompt through X_eTaL.
pub fn run(p: &Prompt, c: &Config) -> Result<Anatomy, String> {
    let p = p.clone().checked()?;
    let tokens = p.tokens();
    let n = tokens.len();
    let out = output(&program(&p, c), 5)?;
    let read = numbers::<f64>(&out[3], 5)?;
    Ok(Anatomy {
        attention: numbers(&out[0], n)?,
        logits: numbers(&out[1], 16)?,
        margin: numbers::<f64>(&out[2], 1)?[0],
        sink: read[0],
        matching: read[1],
        other: read[2],
        gate: read[3],
        law: read[4],
        blocks: numbers(&out[4], n * 46)?,
        tokens,
        prompt: p,
        config: *c,
    })
}

/// The paper's runs, the CLI's own lines: the two prompts read, then
/// each knockout's margin and sink.
pub fn paper_program() -> String {
    format!("{}{}", head(), section(SOURCE, "## Rounded to 4 places", "### The read law"))
}

/// The knockouts of the paper's Section 12: a name, the prompt, the
/// paper's margin and sink (None where it gives none).
pub const KNOCKOUTS: [(&str, &str, f64, Option<f64>); 7] = [
    ("intact", "unique", 0.24, Some(0.67)),
    ("the gate added (duplicate detector out)", "unique", 0.73, None),
    ("intact", "recurring", 0.83, Some(0.02)),
    ("duplicate detector knocked out", "recurring", 0.39, Some(0.54)),
    ("... and the gate added back", "recurring", 0.84, None),
    ("... but kept out of the readers' queries", "recurring", 0.39, None),
    ("writers knocked out", "recurring", 0.00, None),
];

/// Each knockout's (margin, sink), as X_eTaL computed them.
pub fn paper() -> Result<Vec<(f64, f64)>, String> {
    let out = output(&paper_program(), 9)?;
    out[2..].iter().map(|l| numbers::<f64>(l, 2).map(|v| (v[0], v[1]))).collect()
}
