//! The model: the demo's own X_eTaL program (net-macro.xtl) and its
//! data, run for one of its networks; and a spec typed on the page,
//! expanded by the Net macro library. Nothing here knows about the
//! browser.

use microscope::run::{expanded, numbers, output, section};

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
