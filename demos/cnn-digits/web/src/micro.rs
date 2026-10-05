//! The model: the demo's own X_eTaL program (cnn-digits.xtl) and its
//! data, run on the digit the page picks. Nothing here knows about the
//! browser.

use microscope::run::{lit, numbers, output, section};

/// The command-line program.
pub const SOURCE: &str = include_str!("../../cnn-digits.xtl");

/// The data the program reads, as the page's X_eTaL finds it: the same
/// paths as at the command line.
pub const DATA: [(&str, &str); 4] = [
    ("data/filters.txt", include_str!("../../data/filters.txt")),
    ("data/dense.txt", include_str!("../../data/dense.txt")),
    ("data/samples.txt", include_str!("../../data/samples.txt")),
    ("data/expected.txt", include_str!("../../data/expected.txt")),
];

/// What the page's program starts with: the import of NN, the network
/// read from data/ and the core (convolution, pooling, classify).
pub fn head() -> &'static str {
    section(SOURCE, "\"nn:\" u_se< \"NN\"", "# -- end of the core")
}

/// The digit run: one of the ten samples (0 to 9) or one drawn on the
/// page (784 values, 0 to 1, row by row).
#[derive(Clone, Debug, PartialEq)]
pub enum Input {
    Sample(usize),
    Drawn(Vec<f64>),
}

/// The line binding the digit `x`.
pub fn input_line(input: &Input) -> String {
    match input {
        Input::Sample(d) => format!("x := {} s_elect samples", d + 1),
        Input::Drawn(v) => {
            let lits: Vec<String> = v.iter().map(|&p| lit((p.clamp(0.0, 1.0) * 100.0).round() / 100.0)).collect();
            format!("x := 28 28 r_eshape {}", lits.join(" "))
        }
    }
}

/// The run's own lines, after the head: the digit, each stage, and
/// every array the page draws.
pub fn tail(input: &Input) -> String {
    format!(
        "{}\nc := nn:r_elu u:c_onv x\nm := u:p_ool c\np := u:c_lassify x\n\
         r_avel x\nr_avel k9\nr_avel bc\nr_avel c\nr_avel m\nr_avel p\n",
        input_line(input)
    )
}

/// The whole program the page runs.
pub fn program(input: &Input) -> String {
    format!("{}{}", head(), tail(input))
}

/// Every stage of one digit through the network, as X_eTaL computed it.
#[derive(Clone, Debug, PartialEq)]
pub struct Anatomy {
    /// The digit, 28 x 28.
    pub x: Vec<f64>,
    /// The filters, 8 x 9 (each 3 x 3 row by row), and their biases.
    pub k: Vec<f64>,
    pub bc: Vec<f64>,
    /// After convolution and ReLU, 8 x 26 x 26.
    pub c: Vec<f64>,
    /// After 2 x 2 max-pooling, 8 x 13 x 13.
    pub m: Vec<f64>,
    /// The ten probabilities.
    pub p: Vec<f64>,
}

/// One convolution output, worked by hand: the 3 x 3 patch of the digit
/// under it, the filter, the sum of their products plus the bias, and
/// what ReLU keeps.
#[derive(Clone, Debug, PartialEq)]
pub struct Patch {
    pub pixels: [f64; 9],
    pub weights: [f64; 9],
    pub sum: f64,
    pub bias: f64,
    pub value: f64,
}

impl Anatomy {
    /// The digit read: the most probable class.
    pub fn digit(&self) -> usize {
        (0..10).fold(0, |b, i| if self.p[i] > self.p[b] { i } else { b })
    }

    /// Output (r, c) of filter f (all from 0): its patch is the digit's
    /// rows r..r+2, columns c..c+2.
    pub fn patch(&self, f: usize, r: usize, c: usize) -> Patch {
        let mut pixels = [0.0; 9];
        let mut weights = [0.0; 9];
        for a in 0..3 {
            for b in 0..3 {
                pixels[3 * a + b] = self.x[(r + a) * 28 + c + b];
                weights[3 * a + b] = self.k[9 * f + 3 * a + b];
            }
        }
        let sum = (0..9).map(|i| pixels[i] * weights[i]).sum::<f64>();
        let bias = self.bc[f];
        Patch { pixels, weights, sum, bias, value: (sum + bias).max(0.0) }
    }

    /// Where filter f answers most strongly: (row, column) of its
    /// largest output (the first, on a tie).
    pub fn strongest(&self, f: usize) -> (usize, usize) {
        let map = &self.c[f * 676..(f + 1) * 676];
        let i = (0..676).fold(0, |b, i| if map[i] > map[b] { i } else { b });
        (i / 26, i % 26)
    }

    /// X_eTaL's value of that output.
    pub fn c_at(&self, f: usize, r: usize, c: usize) -> f64 {
        self.c[(f * 26 + r) * 26 + c]
    }
}

/// Run the digit through X_eTaL.
pub fn run(input: &Input) -> Result<Anatomy, String> {
    for (path, text) in DATA {
        microscope::libs::add(path, text);
    }
    let out = output(&program(input), 6)?;
    Ok(Anatomy {
        x: numbers(&out[0], 784)?,
        k: numbers(&out[1], 72)?,
        bc: numbers(&out[2], 8)?,
        c: numbers(&out[3], 8 * 26 * 26)?,
        m: numbers(&out[4], 8 * 13 * 13)?,
        p: numbers(&out[5], 10)?,
    })
}
