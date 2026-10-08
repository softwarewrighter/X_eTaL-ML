//! The model: the demo's own X_eTaL program (ternary-net.xtl): its
//! trained weights, its network, the four weight formats and the
//! measures, run with the weight set, threshold and point the page
//! picks. Nothing here knows about the browser.

use microscope::run::{lit, matrix, numbers, output, section};

/// The command-line program; the page runs its sections.
pub const SOURCE: &str = include_str!("../../ternary-net.xtl");

/// The data the program reads (written by train/), as the page's
/// X_eTaL finds it: the same paths as at the command line.
pub const DATA: [(&str, &str); 3] = [
    ("data/fp.txt", include_str!("../../data/fp.txt")),
    ("data/qa.txt", include_str!("../../data/qa.txt")),
    ("data/test.txt", include_str!("../../data/test.txt")),
];

/// Run `src` with the data files in the store (see `output`).
fn out(src: &str, lines: usize) -> Result<Vec<String>, String> {
    for (path, text) in DATA {
        microscope::libs::add(path, text);
    }
    output(src, lines)
}

/// The weights and the test points, read from data/.
pub fn weights() -> &'static str {
    section(SOURCE, "### The weights", "# -- end of the weights")
}

/// The inputs, the network, the formats, additions only, the maps
/// and the measures.
pub fn core() -> &'static str {
    section(SOURCE, "### The inputs", "# -- end of the core")
}

/// The functions and the weights in each format (the core up to the maps).
pub fn functions() -> &'static str {
    section(SOURCE, "### The inputs", "### The maps")
}

/// The maps: every map point through each format.
pub fn maps() -> &'static str {
    section(SOURCE, "### The maps", "### The measures")
}

/// The measures (they read `y32`, FP32's map).
pub fn measures() -> &'static str {
    section(SOURCE, "### The measures", "# -- end of the core")
}

/// The map line computing `name` (`y32 := grid u:n_et m32`, ...).
fn map_line(name: &str) -> &'static str {
    maps().lines().find(|l| l.starts_with(&format!("{name} := "))).unwrap_or("")
}

/// The map is SIDE by SIDE points.
pub const SIDE: usize = 24;

/// The weight sets: a name and the X_eTaL name of the 17 x 3 x 16 model.
pub const SETS: &[(&str, &str)] = &[("Trained in full precision", "fp"), ("Trained for ternary", "qa")];

pub const FORMATS: [&str; 4] = ["FP32", "FP16", "INT8", "1.58-bit"];
/// Bits per weight in each format (ternary: log2 3).
pub const BITS: [f64; 4] = [32.0, 16.0, 8.0, 1.584962500721156];
/// Real weights (2 x 16 + 16 x 16 + 16 x 3) and outputs (16 + 16 + 3).
pub const WEIGHTS: usize = 336;
pub const OUTPUTS: usize = 35;

/// What the page sets.
#[derive(Clone, Debug, PartialEq)]
pub struct Setup {
    pub set: usize,
    pub t: f64,
    /// The point inspected, (x, y).
    pub point: (f64, f64),
}

impl Default for Setup {
    fn default() -> Self {
        Setup { set: 0, t: 0.5, point: (0.3, 0.4) }
    }
}

/// What X_eTaL computed.
#[derive(Clone, Debug, PartialEq)]
pub struct Anatomy {
    /// Per format: accuracy, agreement with FP32, mean output error.
    pub measures: [[f64; 3]; 4],
    /// Per format: the arm (1, 2 or 3) of each map point, row by row.
    pub maps: [Vec<f64>; 4],
    /// The ternary weights, 16 x 3 x 16 (input, layer, output).
    pub q: Vec<f64>,
    /// Each layer's scale.
    pub scales: [f64; 3],
    /// The point through the ternary network: layer 1's output, layer
    /// 2's additions and subtractions, its outputs, its biases.
    pub h1: Vec<f64>,
    pub adds: Vec<f64>,
    pub pre: Vec<f64>,
    pub b2: Vec<f64>,
    /// Per format: the point's three outputs.
    pub outs: [[f64; 3]; 4],
    /// The test points and their arms.
    pub px: Vec<f64>,
    pub py: Vec<f64>,
    pub pl: Vec<f64>,
}

impl Anatomy {
    /// The ternary weight from input i to output o of layer l.
    pub fn q(&self, i: usize, l: usize, o: usize) -> f64 {
        self.q[i * 48 + l * 16 + o]
    }

    /// The kept (nonzero) ternary weights: the additions a point needs.
    pub fn kept(&self) -> usize {
        self.q.iter().filter(|&&v| v != 0.0).count()
    }
}

/// What every program the page runs starts with: the page's settings,
/// the weights read from data/, the weight set, the functions.
pub fn head(s: &Setup) -> String {
    format!("g := {SIDE}\nt := {}\n{}m := {}\n{}", lit(s.t), weights(), SETS[s.set % SETS.len()].1, functions())
}

/// The program for a weight set: FP32, FP16 and INT8 over the map and
/// their measures, FP32's outputs (for the ternary program) and the
/// test points. It depends on the set only.
pub fn program_formats(s: &Setup) -> String {
    format!(
        "{}{}\n{}\n{}\n{}\
         r_avel (m32 u:m_easures y32) c_at (m16 u:m_easures y16) c_at m8 u:m_easures y8\n\
         r_avel u:a_rm y32\nr_avel u:a_rm y16\nr_avel u:a_rm y8\nr_avel y32\nr_avel px\nr_avel py\nr_avel pl\n",
        head(s),
        map_line("y32"),
        map_line("y16"),
        map_line("y8"),
        measures()
    )
}

/// The program for the ternary weights at threshold `t`, given FP32's
/// outputs over the map (`y32`, from the formats program).
pub fn program_ternary(s: &Setup, y32: &[f64]) -> String {
    format!(
        "{}{}\n{}{}r_avel m2 u:m_easures y2\nr_avel u:a_rm y2\nr_avel q\n'm_ax r_/_13 a\n",
        head(s),
        map_line("y2"),
        matrix("y32", SIDE * SIDE, 3, y32.iter().map(|&v| lit(v))),
        measures()
    )
}

/// The program for one point: through the ternary layers as additions,
/// and through every format.
pub fn program_point(s: &Setup) -> String {
    let (x, y) = s.point;
    format!(
        "{}pt := (1 r_eshape {}) u:p_ad 1 r_eshape {}\nh1 := u:r_elu pt u:l_ayer 1 s_elect_2 m2\n\
         r_avel h1\nr_avel (f_irst h1) u:a_dds 2 s_elect_2 q\nr_avel h1 u:l_ayer 2 s_elect_2 m2\nr_avel f_irst -1 t_ake 2 s_elect_2 m2\n\
         r_avel (pt u:n_et m32) c_at (pt u:n_et m16) c_at (pt u:n_et m8) c_at pt u:n_et m2\n",
        head(s),
        lit(x),
        lit(y)
    )
}

fn rows<const N: usize, const M: usize>(v: &[f64]) -> [[f64; N]; M] {
    let mut out = [[0.0; N]; M];
    for (k, row) in out.iter_mut().enumerate() {
        row.copy_from_slice(&v[k * N..(k + 1) * N]);
    }
    out
}

/// FP32, FP16 and INT8 for a weight set.
#[derive(Clone, Debug, PartialEq)]
pub struct Formats {
    pub measures: [[f64; 3]; 3],
    pub maps: [Vec<f64>; 3],
    pub y32: Vec<f64>,
    pub px: Vec<f64>,
    pub py: Vec<f64>,
    pub pl: Vec<f64>,
}

pub fn run_formats(s: &Setup) -> Result<Formats, String> {
    let out = out(&program_formats(s), 8)?;
    let n = SIDE * SIDE;
    let tests = out[7].split_whitespace().count();
    Ok(Formats {
        measures: rows::<3, 3>(&numbers(&out[0], 9)?),
        maps: [numbers(&out[1], n)?, numbers(&out[2], n)?, numbers(&out[3], n)?],
        y32: numbers(&out[4], 3 * n)?,
        px: numbers(&out[5], tests)?,
        py: numbers(&out[6], tests)?,
        pl: numbers(&out[7], tests)?,
    })
}

/// The ternary weights at a threshold.
#[derive(Clone, Debug, PartialEq)]
pub struct Ternary {
    pub measures: [f64; 3],
    pub map: Vec<f64>,
    pub q: Vec<f64>,
    pub scales: [f64; 3],
}

pub fn run_ternary(s: &Setup, y32: &[f64]) -> Result<Ternary, String> {
    let out = out(&program_ternary(s, y32), 4)?;
    let m = numbers::<f64>(&out[0], 3)?;
    let sc = numbers::<f64>(&out[3], 3)?;
    Ok(Ternary { measures: [m[0], m[1], m[2]], map: numbers(&out[1], SIDE * SIDE)?, q: numbers(&out[2], 768)?, scales: [sc[0], sc[1], sc[2]] })
}

/// One point through the network.
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    pub h1: Vec<f64>,
    pub adds: Vec<f64>,
    pub pre: Vec<f64>,
    pub b2: Vec<f64>,
    pub outs: [[f64; 3]; 4],
}

pub fn run_point(s: &Setup) -> Result<Point, String> {
    let out = out(&program_point(s), 5)?;
    Ok(Point {
        h1: numbers(&out[0], 16)?,
        adds: numbers(&out[1], 16)?,
        pre: numbers(&out[2], 16)?,
        b2: numbers(&out[3], 16)?,
        outs: rows::<3, 4>(&numbers(&out[4], 12)?),
    })
}

/// All three programs.
pub fn run(s: &Setup) -> Result<Anatomy, String> {
    let f = run_formats(s)?;
    let t = run_ternary(s, &f.y32)?;
    let p = run_point(s)?;
    Ok(Anatomy {
        measures: [f.measures[0], f.measures[1], f.measures[2], t.measures],
        maps: [f.maps[0].clone(), f.maps[1].clone(), f.maps[2].clone(), t.map],
        q: t.q,
        scales: t.scales,
        h1: p.h1,
        adds: p.adds,
        pre: p.pre,
        b2: p.b2,
        outs: p.outs,
        px: f.px,
        py: f.py,
        pl: f.pl,
    })
}

/// The point (x, y) at the center of map cell (row, column) of a
/// picture `size` cells wide (y upwards, -1.1 .. 1.1).
pub fn at(row: usize, col: usize, size: usize) -> (f64, f64) {
    let f = |k: usize| -1.1 + 2.2 * (k as f64 + 0.5) / size as f64;
    (f(col), -f(row))
}
