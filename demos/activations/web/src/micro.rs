//! The model: the demo's own program (activations.xtl), a family of it
//! run with the sliders' parameters, every curve read back and its two
//! charts drawn by Plot; and an activation of the reader's own, run in
//! the same program and drawn beside a built-in, its slope by central
//! differences.

use microscope::run::{lit, numbers, output_pictures, section};

/// The command line's program, which the page runs part by part.
pub const PROGRAM: &str = include_str!("../../activations.xtl");

/// A family: its title, its section's heading in the program, the
/// names its curves and their slopes are bound to, and its curves' names.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Family {
    pub title: &'static str,
    pub heading: &'static str,
    pub ys: &'static str,
    pub ds: &'static str,
    pub names: &'static [&'static str],
}

pub const FAMILIES: [Family; 4] = [
    Family { title: "Classical", heading: "### Classical", ys: "classical", ds: "dClassical", names: &["sigmoid", "tanh", "hard sigmoid"] },
    Family { title: "Rectifiers", heading: "### Rectifiers", ys: "rect", ds: "dRect", names: &["ReLU", "leaky ReLU", "ELU", "SELU"] },
    Family { title: "Smooth", heading: "### Smooth", ys: "smooth", ds: "dSmooth", names: &["softplus", "SiLU", "Swish", "GELU", "GELU (erf)", "Mish", "hard swish"] },
    Family { title: "Gated", heading: "### Gated", ys: "gated", ds: "dGated", names: &["GLU", "ReGLU", "GEGLU", "SwiGLU"] },
];

/// How many x values a curve has (241 from -6 to 6).
pub const POINTS: usize = 241;

/// The parameters the sliders set: leaky ReLU's slope, ELU's alpha,
/// Swish's beta.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub leak: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params { leak: 0.1, alpha: 1.0, beta: 1.5 }
    }
}

/// The x values, as the program makes them (Plot's l_inspace: lo +
/// (hi - lo) times the fraction of the way).
pub fn xs() -> Vec<f64> {
    (0..POINTS).map(|i| -6.0 + 12.0 * (i as f64 / (POINTS - 1) as f64)).collect()
}

/// What every run starts with: the imports, the data (the parameters
/// as `p` gives them) and the helpers, up to the first family.
pub fn head(p: Params) -> String {
    section(PROGRAM, "", "### Classical")
        .replacen("leak := 0.1\n", &format!("leak := {}\n", lit(p.leak)), 1)
        .replacen("alpha := 1.0\n", &format!("alpha := {}\n", lit(p.alpha)), 1)
        .replacen("beta := 1.5\n", &format!("beta := {}\n", lit(p.beta)), 1)
}

/// The program the page runs for family `f`: the head, the family's
/// section (its curves, its two charts), then its curves and slopes
/// printed for the page to read back.
pub fn program(f: usize, p: Params) -> String {
    let fam = FAMILIES[f];
    let next = FAMILIES.get(f + 1).map_or("### The ranges", |n| n.heading);
    format!("{}{}r_avel {}\nr_avel {}\n", head(p), section(PROGRAM, fam.heading, next), fam.ys, fam.ds)
}

/// What a family's run gave: each curve and its slope over xs, and the
/// two charts' SVG text.
#[derive(Clone, Debug, PartialEq)]
pub struct Drawn {
    pub ys: Vec<Vec<f64>>,
    pub ds: Vec<Vec<f64>>,
    pub charts: Vec<String>,
}

/// Run family `f` with parameters `p`.
pub fn run(f: usize, p: Params) -> Result<Drawn, String> {
    let n = FAMILIES[f].names.len();
    let (out, charts) = output_pictures(&program(f, p), 2)?;
    let rows = |line: &str| numbers::<f64>(line, n * POINTS).map(|v| v.chunks(POINTS).map(<[f64]>::to_vec).collect());
    Ok(Drawn { ys: rows(&out[0])?, ds: rows(&out[1])?, charts })
}

/// The built-ins to draw the reader's own beside: a name, its curve and
/// its slope as X_eTaL over xs.
pub const BUILTINS: [(&str, &str, &str); 8] = [
    ("GELU", "act:g_elu xs", "act:d_gelu xs"),
    ("SiLU", "act:s_ilu xs", "act:d_silu xs"),
    ("ReLU", "act:r_elu xs", "act:d_relu xs"),
    ("sigmoid", "act:s_igmoid xs", "act:d_sigmoid xs"),
    ("tanh", "act:t_anh xs", "act:d_tanh xs"),
    ("softplus", "act:s_oftplus xs", "act:d_softplus xs"),
    ("Mish", "act:m_ish xs", "act:d_mish xs"),
    ("hard swish", "act:h_ardSwish xs", "act:d_hardSwish xs"),
];

/// What the editor starts with.
pub const YOURS: &str = "## Your activation, a function of one number x: here GELU's sigmoid\n## approximation, x sigmoid(1.702 x). Change it, then Run.\nu:m_ine := { x -> x * act:s_igmoid 1.702 * x }\n";

/// The program for the reader's own activation `text` (which defines
/// `u:m_ine`), drawn beside built-in `b`: the head, the text, then its
/// curve (one x at a time), its slope by central differences, a check
/// that both are finite, the two charts if they are, and three lines
/// printed: what happened, the curve, the slope.
pub fn yours(text: &str, b: usize) -> String {
    let (name, y, d) = BUILTINS[b];
    format!(
        "{}### Yours\n\n{}\n## Your activation over xs (one x at a time), its slope by central\n## differences, beside {name}; drawn if every value is finite.\nu:o_ne := {{ x -> 'u:m_ine e_ach x }}\nmine := u:o_ne xs\ndMine := 'u:o_ne u:s_lope xs\n## Finite: v - v is 0 (NaN for NaN and for an infinity).\nfin := (0.0 = mine - mine) & 0.0 = dMine - dMine\nu:b_oth := {{ @ ->\n  _ := (\"Yours\" \"x\" \"f(x)\" \"yours\" \"{name}\") u:c_hart 2 241 r_eshape mine c_at {y}\n  _ := (\"Yours: slopes\" \"x\" \"f'(x)\" \"yours\" \"{name}\") u:c_hart 2 241 r_eshape dMine c_at {d}\n  \"drawn\"\n}}\nu:s_tatus := {{ ok -> ok ? u:b_oth @; \"not finite at x = \" c_at f_ormat 1 t_ake (n_ot fin) r_eplicate xs }}\nu:s_tatus '& r_/ fin\nmine\ndMine\n",
        head(Params::default()),
        text.trim_end()
    )
}

/// What the reader's activation gave: "drawn" or why not, its curve,
/// its slope, and the charts.
#[derive(Clone, Debug, PartialEq)]
pub struct Mine {
    pub status: String,
    pub ys: Vec<f64>,
    pub ds: Vec<f64>,
    pub charts: Vec<String>,
}

/// Run the reader's activation; an X_eTaL error is the Err, its line
/// numbers counted in the reader's text (`placed`).
pub fn run_yours(text: &str, b: usize) -> Result<Mine, String> {
    let src = yours(text, b);
    let before = head(Params::default()).lines().count() + 2;
    let (out, charts) = output_pictures(&src, 3).map_err(|e| placed(&e, before))?;
    let num = |line: &str| numbers::<f64>(line, POINTS).unwrap_or_default();
    Ok(Mine { status: out[0].clone(), ys: num(&out[1]), ds: num(&out[2]), charts })
}

/// An error's line numbers counted in the reader's text: the program's
/// lines after the first `before` are the text's, from 1 (`--> main.xtl:N:C`
/// and the quoted `N | ...`).
pub fn placed(err: &str, before: usize) -> String {
    let shift = |n: usize| if n > before { (n - before).to_string() } else { n.to_string() };
    err.lines()
        .map(|l| {
            if let Some(i) = l.find(".xtl:") {
                let rest = &l[i + 5..];
                let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
                if let Ok(n) = digits.parse::<usize>() {
                    return format!("{}.xtl:{}{}", &l[..i], shift(n), &rest[digits.len()..]);
                }
            }
            let t = l.trim_start();
            let digits: String = t.chars().take_while(char::is_ascii_digit).collect();
            match (digits.parse::<usize>(), t[digits.len()..].starts_with(" |")) {
                (Ok(n), true) => format!("{}{}{}", &l[..l.len() - t.len()], shift(n), &t[digits.len()..]),
                _ => l.to_string(),
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}
