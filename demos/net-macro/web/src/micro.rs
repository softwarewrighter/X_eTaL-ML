//! The model: the demo's own X_eTaL program (net-macro.xtl) and its
//! data, run for one of its networks; and a spec typed on the page,
//! expanded by the Net macro library. Nothing here knows about the
//! browser.

use microscope::run::{expanded, lit, numbers, output, output_pictures, section};

/// The command-line program.
pub const SOURCE: &str = include_str!("../../net-macro.xtl");

/// The data the program reads, as the page's X_eTaL finds it.
pub const DATA: [(&str, &str); 7] = [
    ("data/points.txt", include_str!("../../data/points.txt")),
    ("data/a1.txt", include_str!("../../data/a1.txt")),
    ("data/b1.txt", include_str!("../../data/b1.txt")),
    ("data/b2.txt", include_str!("../../data/b2.txt")),
    ("data/c1.txt", include_str!("../../data/c1.txt")),
    ("data/c2.txt", include_str!("../../data/c2.txt")),
    ("data/c3.txt", include_str!("../../data/c3.txt")),
];

/// The decision map is SIDE by SIDE points.
pub const SIDE: usize = 40;

/// One of the program's networks: its function's name, its weight
/// files' prefix and its spec, as its line `"u:NAME PREFIX"
/// net:m_odel< "SPEC"` writes them.
#[derive(Clone, Debug, PartialEq)]
pub struct Network {
    pub name: String,
    pub prefix: String,
    pub spec: String,
}

impl Network {
    /// The weight arrays the macro loads: PREFIX1 PREFIX2 ...
    pub fn weights(&self) -> String {
        let dense = self.spec.split_whitespace().filter(|w| w.parse::<usize>().is_ok()).count().saturating_sub(1);
        (1..=dense).map(|k| format!("{}{k}", self.prefix)).collect::<Vec<_>>().join(" ")
    }

    /// Its line in the program.
    pub fn line(&self) -> String {
        format!("\"u:{} {}\" net:m_odel< \"{}\"", self.name, self.prefix, self.spec)
    }
}

/// The program's networks, from its lines `"u:NAME PREFIX" net:m_odel< "SPEC"`.
pub fn networks() -> Vec<Network> {
    SOURCE
        .lines()
        .filter_map(|l| {
            let (what, rest) = l.strip_prefix("\"u:")?.split_once("\" net:m_odel< \"")?;
            let (name, prefix) = what.split_once(' ')?;
            Some(Network { name: name.into(), prefix: prefix.into(), spec: rest.strip_suffix('"')?.into() })
        })
        .collect()
}

/// What the page's program starts with: the imports, the test points
/// and the three networks (each line loads its weights and writes its
/// function).
pub fn head() -> &'static str {
    section(SOURCE, "\"nn:\" u_se< \"NN\"", "# -- end of the core")
}

/// The program's line binding `name` (the map's grid is the program's own).
fn line(name: &str) -> &'static str {
    SOURCE.lines().find(|l| l.starts_with(&format!("{name} := "))).unwrap_or("")
}

/// The run's own lines for network `n`: the map's grid, the network on
/// it, and what the page shows.
pub fn tail(n: &Network) -> String {
    format!(
        "g := {SIDE}\n{}\n{}\ny := u:{} grid\nr_avel nn:a_rgmax y\narm nn:a_ccuracy u:{} xy\n\"{}\" net:p_arams< @\nr_avel pts\n",
        line("c"),
        line("grid"),
        n.name,
        n.name,
        n.spec
    )
}

/// The whole program the page runs for a network of the program.
pub fn program(n: &Network) -> String {
    format!("{}{}", head(), tail(n))
}

/// A network of the program, run.
#[derive(Clone, Debug, PartialEq)]
pub struct Trained {
    /// What the macro wrote for its line: the weights loaded and
    /// checked, then the function.
    pub expansion: String,
    /// The arm (1 to 3) decided at each map point, row by row.
    pub map: Vec<usize>,
    pub accuracy: f64,
    pub params: usize,
    /// The test points: every x, then every y, then every arm.
    pub points: Vec<f64>,
}

fn store() {
    for (path, text) in DATA {
        microscope::libs::add(path, text);
    }
}

/// The lines of an expanded program that a model's call became: its
/// weight arrays' loading and the function `name`.
fn written(expansion: &str, name: &str, weights: &str) -> String {
    let mine = |l: &str| l.starts_with(&format!("u:{name} := ")) || weights.split_whitespace().any(|w| l.starts_with(&format!("{w} := ")));
    expansion.lines().filter(|l| mine(l)).collect::<Vec<_>>().join("\n")
}

pub fn run(n: &Network) -> Result<Trained, String> {
    store();
    let src = program(n);
    let out = output(&src, 4)?;
    let points: Vec<f64> = out[3].split_whitespace().map(|t| t.parse().map_err(|e| format!("{e}"))).collect::<Result<_, _>>()?;
    Ok(Trained {
        expansion: written(&expanded(&src)?, &n.name, &n.weights()),
        map: numbers(&out[0], SIDE * SIDE)?,
        accuracy: numbers::<f64>(&out[1], 1)?[0],
        params: numbers::<usize>(&out[2], 1)?[0],
        points,
    })
}

/// A spec as it may stand inside an X_eTaL string: quotes and
/// backslashes are not part of any spec, so they become spaces.
pub fn clean(spec: &str) -> String {
    spec.chars().map(|c| if c == '"' || c == '\\' || c.is_control() { ' ' } else { c }).collect::<String>().trim().to_string()
}

/// Names for a typed spec's weight arrays, one per dense layer (a
/// number after the first): w1 w2 ...
pub fn weight_names(spec: &str) -> String {
    let dense = spec.split_whitespace().filter(|w| w.chars().all(|c| c.is_ascii_digit())).count().saturating_sub(1);
    (1..=dense).map(|k| format!("w{k}")).collect::<Vec<_>>().join(" ")
}

/// The two small programs the page expands and runs for a typed spec.
pub fn typed_programs(spec: &str) -> (String, String) {
    let spec = clean(spec);
    (
        format!("\"nn:\" u_se< \"NN\"\n\"net:\" u_se< \"Net\"\nu:n_et := \"{spec}\" net:n_etwork< \"{}\"\n", weight_names(&spec)),
        format!("\"net:\" u_se< \"Net\"\n\"{spec}\" net:p_arams< @\n"),
    )
}

/// A spec typed on the page: what the macro wrote for it and its
/// parameter count; or the message the macro refused it with.
#[derive(Clone, Debug, PartialEq)]
pub struct Typed {
    pub expansion: String,
    pub params: usize,
}

pub fn typed(spec: &str) -> Result<Typed, String> {
    store();
    let (network, params) = typed_programs(spec);
    let expansion = written(&expanded(&network)?, "n_et", "");
    Ok(Typed { expansion, params: numbers::<usize>(&output(&params, 1)?[0], 1)?[0] })
}

// -- training a spec in the page: train-it.xtl ----------------------

/// The training program.
pub const TRAIN_SOURCE: &str = include_str!("../../train-it.xtl");
/// The spec train-it.xtl trains.
pub const TRAIN_SPEC: &str = "2 16 relu 16 relu 3 softmax";
/// Training points.
pub const POINTS: usize = 300;

/// A spec's sizes, if the page can train it: 2 inputs (a point's x and
/// y), 3 softmax outputs (the arms); or why not.
pub fn trainable(spec: &str) -> Result<Vec<usize>, String> {
    let words: Vec<&str> = spec.split_whitespace().collect();
    let sizes: Vec<usize> = words.iter().filter_map(|w| w.parse().ok()).collect();
    match (sizes.first(), words.len() >= 2 && words[words.len() - 2..] == ["3", "softmax"]) {
        (Some(2), true) if sizes.iter().all(|&n| (1..=64).contains(&n)) => Ok(sizes),
        _ => Err("To train it here a spec takes 2 inputs (a point's x and y) and ends in 3 softmax (the three arms), each layer at most 64 wide.".into()),
    }
}

/// The weight arrays' shapes, inputs + 1 by outputs, layer by layer.
fn shapes(sizes: &[usize]) -> Vec<(usize, usize)> {
    sizes.windows(2).map(|p| (p[0] + 1, p[1])).collect()
}

/// The training state's arrays in order: each layer's weights, Adam's
/// two averages per layer, the step count.
pub fn parts(sizes: &[usize]) -> Vec<(usize, usize)> {
    let s = shapes(sizes);
    [s.clone(), s.clone(), s, vec![(1, 1)]].concat()
}

/// The training program's head for `spec`: the spiral, then train-it's
/// lines with the spec, its random weights and its state's names.
pub fn train_head(spec: &str, sizes: &[usize]) -> String {
    let names = weight_names(spec);
    let mut out = vec![];
    for l in section(TRAIN_SOURCE, "\"nn:\" u_se< \"NN\"", "# -- end of the core").lines() {
        if l.starts_with("w1 := u:r_andom") {
            out.extend(shapes(sizes).iter().enumerate().map(|(k, (r, c))| format!("w{} := u:r_andom {r} {c}", k + 1)));
        } else if !(l.starts_with('w') && l.contains(" := u:r_andom")) {
            out.push(l.replace(TRAIN_SPEC, spec).replace("\"w1 w2 w3\"", &format!("\"{names}\"")));
        }
    }
    out.join("\n") + "\n"
}

/// The state's numbers as X_eTaL: its arrays boxed and joined.
pub fn boxes(sizes: &[usize], s: &[f64]) -> String {
    let mut at = 0;
    parts(sizes)
        .iter()
        .map(|&(r, c)| {
            let part: Vec<String> = s[at..at + r * c].iter().map(|&v| lit(v)).collect();
            at += r * c;
            match (r, c) {
                (1, 1) => format!("(e_nclose {})", part[0]),
                _ => format!("(e_nclose {r} {c} r_eshape {})", part.join(" ")),
            }
        })
        .collect::<Vec<_>>()
        .join(" c_at ")
}

/// Earlier values and this run's, as X_eTaL: `(v1 v2 ...) c_at NOW`.
fn history(earlier: &[f64], now: &str) -> String {
    match earlier.is_empty() {
        true => now.to_string(),
        false => format!("({}) c_at {now}", earlier.iter().map(|&v| lit(v)).collect::<Vec<_>>().join(" ")),
    }
}

/// The loss and the share right after every run so far, drawn by Plot
/// (from the second run: a line takes two points).
fn curves(earlier: &[(f64, f64)]) -> String {
    if earlier.is_empty() {
        return String::new();
    }
    format!(
        "\"p:\" u_se< \"Plot\"\n\
         losses := {}\n\
         right := {}\n\
         # The loss stretched to its own range (Plot's line chart keeps a range of at least 1).\n\
         lossChart := p:l_ine! (losses - 'm_in r_/ losses) / 0.000001 m_ax ('m_ax r_/ losses) - 'm_in r_/ losses\n\
         rightChart := p:l_ine! right\n",
        history(&earlier.iter().map(|e| e.0).collect::<Vec<_>>(), "1 t_ake now"),
        history(&earlier.iter().map(|e| e.1).collect::<Vec<_>>(), "-1 t_ake now"),
    )
}

/// A run's own lines: the state to start from (s0, or the page's),
/// `steps` training steps, the network on the weights reached, what
/// the page shows, then the loss and accuracy so far (`earlier`: the
/// loss and share right after each earlier run) drawn by Plot.
pub fn train_tail(spec: &str, sizes: &[usize], state: Option<&[f64]>, steps: usize, earlier: &[(f64, f64)]) -> String {
    let start = match state {
        None => "s := s0".to_string(),
        Some(s) => format!("s := {}", boxes(sizes, s)),
    };
    let names = weight_names(spec);
    let weights: String = (1..sizes.len()).map(|k| format!("w{k} := d_isclose {k} s_elect s\n")).collect();
    format!(
        "{start}\ns := {steps} 'u:s_tep p_ower s\n{weights}u:t_rained := \"{spec}\" net:n_etwork< \"{names}\"\n\
         u:j_oin := {{ b -> d_isclose '{{ x y -> e_nclose (d_isclose x) c_at d_isclose y }} r_/ b }}\n\
         g := {SIDE}\n{}\n{}\n\
         r_avel u:j_oin '{{ b -> r_avel d_isclose b }} m_ap s\n\
         now := (Y nn:c_rossEntropy u:t_rained X) c_at (1 + arm) nn:a_ccuracy u:t_rained X\n\
         now\n\
         r_avel nn:a_rgmax u:t_rained grid\nr_avel o_\\ X\n1 + arm\n{}",
        line("c"),
        line("grid"),
        curves(earlier),
    )
}

pub fn train_program(spec: &str, sizes: &[usize], state: Option<&[f64]>, steps: usize, earlier: &[(f64, f64)]) -> String {
    format!("{}{}", train_head(spec, sizes), train_tail(spec, sizes, state, steps, earlier))
}

/// After a run of training steps, as X_eTaL computed it.
#[derive(Clone, Debug, PartialEq)]
pub struct After {
    pub state: Vec<f64>,
    pub loss: f64,
    pub right: f64,
    /// The arm (1 to 3) decided at each map point, row by row.
    pub map: Vec<usize>,
    /// The training points: every x, then every y, then every arm.
    pub points: Vec<f64>,
    /// Plot's charts of the loss and of the share right so far, SVG.
    pub pictures: Vec<String>,
}

pub fn train(spec: &str, sizes: &[usize], state: Option<&[f64]>, steps: usize, earlier: &[(f64, f64)]) -> Result<After, String> {
    store();
    let (out, pictures) = output_pictures(&train_program(spec, sizes, state, steps, earlier), 5)?;
    let lr = numbers::<f64>(&out[1], 2)?;
    let mut points = numbers::<f64>(&out[3], 2 * POINTS)?;
    points.extend(numbers::<f64>(&out[4], POINTS)?);
    Ok(After {
        state: numbers(&out[0], parts(sizes).iter().map(|(r, c)| r * c).sum())?,
        loss: lr[0],
        right: lr[1],
        map: numbers(&out[2], SIDE * SIDE)?,
        points,
        pictures,
    })
}

/// The training step the macro wrote for `spec`: the function u:s_tep
/// as `xetal expand` shows it.
pub fn step_written(spec: &str, sizes: &[usize]) -> Result<String, String> {
    store();
    let e = expanded(&train_head(spec, sizes))?;
    let from = e.find("u:s_tep := ").ok_or("no u:s_tep in the expansion")?;
    let to = e[from..].find("\n}").map_or(e.len(), |k| from + k + 2);
    Ok(e[from..to].to_string())
}
