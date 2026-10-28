//! The model: the demo's own X_eTaL program (micro-bert.xtl): the tiny
//! BERT's weights and its forward pass, run for a name with a blank.
//! Nothing here knows about the browser.

use microscope::run::{numbers, output, section};

/// The command-line program.
pub const SOURCE: &str = include_str!("../../micro-bert.xtl");

/// The weights and the letters, as the page's X_eTaL finds them.
pub const DATA: [(&str, &str); 14] = [
    ("data/chars.txt", include_str!("../../data/chars.txt")),
    ("data/wte.txt", include_str!("../../data/wte.txt")),
    ("data/wpe.txt", include_str!("../../data/wpe.txt")),
    ("data/ln0.txt", include_str!("../../data/ln0.txt")),
    ("data/wq.txt", include_str!("../../data/wq.txt")),
    ("data/wk.txt", include_str!("../../data/wk.txt")),
    ("data/wv.txt", include_str!("../../data/wv.txt")),
    ("data/wo.txt", include_str!("../../data/wo.txt")),
    ("data/ln1.txt", include_str!("../../data/ln1.txt")),
    ("data/w1.txt", include_str!("../../data/w1.txt")),
    ("data/w2.txt", include_str!("../../data/w2.txt")),
    ("data/ln2.txt", include_str!("../../data/ln2.txt")),
    ("data/lm.txt", include_str!("../../data/lm.txt")),
    ("data/accuracy.txt", include_str!("../../data/accuracy.txt")),
];

/// The letters, a to z.
pub const LETTERS: usize = 26;
/// The longest name the model has places for.
pub const LONGEST: usize = 16;

pub fn chars() -> Vec<char> {
    DATA[0].1.trim().chars().collect()
}

/// The program's head: the model and its forward pass.
pub fn head() -> &'static str {
    section(SOURCE, "\"nn:\" u_se< \"NN\"", "# -- end of the core")
}

/// A name as it may stand in the program: letters a to z and one blank
/// (`_`), at most 16; a name typed without a blank gets one at its end,
/// a second blank is dropped.
pub fn clean(name: &str) -> String {
    let mut out = String::new();
    for c in name.to_lowercase().chars() {
        if out.chars().count() == LONGEST {
            break;
        }
        if c.is_ascii_lowercase() || (c == '_' && !out.contains('_')) {
            out.push(c);
        }
    }
    if !out.contains('_') {
        if out.chars().count() == LONGEST {
            out.pop();
        }
        out.push('_');
    }
    out
}

/// A run's own lines: the blank's probabilities, the input of each
/// layer, where each head looks, and the model as math (the one line
/// written as LaTeX, the attention as Equations draws it).
pub fn tail(name: &str) -> String {
    let s = clean(name);
    format!(
        "s := \"{s}\"\n\
         r_avel u:f_ill s\n\
         ids := u:t_okens s\n\
         n := t_ally ids\n\
         X := u:e_mbed ids\n\
         (h1, _) := u:l_ayer (X, 1)\n\
         r_avel (1, 1) u:l_ooks X\n\
         r_avel (1, 2) u:l_ooks X\n\
         r_avel (2, 1) u:l_ooks h1\n\
         r_avel (2, 2) u:l_ooks h1\n\
         {}\n\
         \"S[i,j] | n n\" e:t_ex< \"(sum k:=0..7 | Q[i,k] * K[j,k]) / sqrt(8.0)\"\n\
         \"A[i,j] | n n\" e:t_ex< \"exp(S[i,j]) / (sum m:=0..n-1 | exp(S[i,m]))\"\n\
         \"Y[i,d] | n 8\" e:t_ex< \"sum j:=0..n-1 | A[i,j] * V[j,d]\"\n",
        one_line()
    )
}

/// The one line of the model, as the command-line program writes it.
pub fn one_line() -> &'static str {
    SOURCE.lines().find(|l| l.starts_with("r\"\\mathrm{BERT}")).expect("micro-bert.xtl writes BERT in one line")
}

pub fn program(name: &str) -> String {
    format!("{}{}", head(), tail(name))
}

/// A name with its blank filled, as X_eTaL computed it.
#[derive(Clone, Debug, PartialEq)]
pub struct Fill {
    pub name: String,
    /// The places, the blank among them.
    pub n: usize,
    /// The probability of each letter for the blank.
    pub probs: Vec<f64>,
    /// Layer 1's two heads, then layer 2's: n x n each, a row per place.
    pub heads: Vec<Vec<f64>>,
    /// The model in one line, then the attention's three equations, as LaTeX.
    pub tex: Vec<String>,
}

pub fn run(name: &str) -> Result<Fill, String> {
    for (path, text) in DATA {
        microscope::libs::add(path, text);
    }
    let name = clean(name);
    let n = name.chars().count();
    let out = output(&program(&name), 9)?;
    Ok(Fill {
        probs: numbers(&out[0], LETTERS)?,
        heads: (1..5).map(|i| numbers(&out[i], n * n)).collect::<Result<_, _>>()?,
        tex: out[5..9].to_vec(),
        n,
        name,
    })
}

/// How the trainer found it does on 1,000 names it never saw: blanks
/// filled right, in its top three, all the blanks, and right by always
/// guessing the commonest letter.
pub fn accuracy() -> [usize; 4] {
    let v: Vec<usize> = DATA[13].1.split_whitespace().map(|w| w.parse().unwrap_or(0)).collect();
    [v[0], v[1], v[2], v[3]]
}
