//! The model: the demo's own X_eTaL program (attention.xtl) and its
//! data, run on a sentence. Nothing here knows about the browser.

use microscope::run::{numbers, output, section};

/// The command-line program.
pub const SOURCE: &str = include_str!("../../attention.xtl");

/// The data the program reads, as the page's X_eTaL finds it.
pub const DATA: [(&str, &str); 4] = [
    ("data/words.txt", include_str!("../../data/words.txt")),
    ("data/features.txt", include_str!("../../data/features.txt")),
    ("data/wq.txt", include_str!("../../data/wq.txt")),
    ("data/wk.txt", include_str!("../../data/wk.txt")),
];

/// The 8 features of every word, in data/features.txt's order.
pub const FEATURES: [&str; 8] = ["animate", "place", "pronoun", "verb", "describes the animate", "describes a place", "function word", "not"];

/// The two dimensions of the head's queries and keys.
pub const DIMS: [&str; 2] = ["animate", "place"];

/// What the page's program starts with: the import, the vocabulary and
/// the head read from data/, and the core.
pub fn head() -> &'static str {
    section(SOURCE, "\"nn:\" u_se< \"NN\"", "# -- end of the core")
}

/// The vocabulary (data/words.txt); the last, `?`, is any other word.
pub fn words() -> Vec<&'static str> {
    DATA[0].1.lines().map(str::trim).collect()
}

/// A sentence's words in the vocabulary (lowercase, punctuation
/// dropped, a plural found by its singular): each word and its number
/// (from 1); any other word is `?`.
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

/// The run's own lines, after the head: the sentence, the mask, every
/// stage, and every array the page draws.
pub fn tail(ids: &[usize], causal: bool) -> String {
    let ids: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
    let mask = match causal {
        true => "M := f_loat (r_ange n) '>= t_able r_ange n",
        false => "M := (n c_at n) r_eshape 1.0",
    };
    format!(
        "ids := {}\nX := ids s_elect F\nn := t_ally ids\n{mask}\n\
         Q := X '+ '* i_nner Wq\nK := X '+ '* i_nner Wk\nS := u:s_cores X\nA := X u:a_ttend M\nY := A '+ '* i_nner X\n\
         r_avel X\nr_avel Q\nr_avel K\nr_avel S\nr_avel A\nr_avel Y\n",
        ids.join(" ")
    )
}

/// The whole program the page runs.
pub fn program(ids: &[usize], causal: bool) -> String {
    format!("{}{}", head(), tail(ids, causal))
}

/// One sentence through the head, as X_eTaL computed it.
#[derive(Clone, Debug, PartialEq)]
pub struct Anatomy {
    pub words: Vec<String>,
    /// n x 8 embeddings, n x 2 queries and keys, n x n scores and
    /// weights, n x 8 outputs.
    pub x: Vec<f64>,
    pub q: Vec<f64>,
    pub k: Vec<f64>,
    pub s: Vec<f64>,
    pub a: Vec<f64>,
    pub y: Vec<f64>,
}

impl Anatomy {
    pub fn n(&self) -> usize {
        self.words.len()
    }

    /// The word token i looks at most, if one gets half its weight.
    pub fn top(&self, i: usize) -> Option<usize> {
        let n = self.n();
        let row = &self.a[i * n..(i + 1) * n];
        let j = (0..n).fold(0, |b, j| if row[j] > row[b] { j } else { b });
        (row[j] >= 0.5).then_some(j)
    }
}

/// Run a sentence through X_eTaL.
pub fn run(sentence: &str, causal: bool) -> Result<Anatomy, String> {
    let toks = tokens(sentence);
    if toks.len() < 2 {
        return Err("type a sentence of at least two words".into());
    }
    let ids: Vec<usize> = toks.iter().map(|t| t.1).collect();
    let n = ids.len();
    for (path, text) in DATA {
        microscope::libs::add(path, text);
    }
    let out = output(&program(&ids, causal), 6)?;
    Ok(Anatomy {
        words: toks.into_iter().map(|t| t.0).collect(),
        x: numbers(&out[0], n * 8)?,
        q: numbers(&out[1], n * 2)?,
        k: numbers(&out[2], n * 2)?,
        s: numbers(&out[3], n * n)?,
        a: numbers(&out[4], n * n)?,
        y: numbers(&out[5], n * 8)?,
    })
}
