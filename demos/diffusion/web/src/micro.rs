//! The model: the demo's own X_eTaL program (diffusion.xtl): the
//! schedule, the denoiser and the DDIM step, run on eight pictures of
//! noise the page supplies, every level of the way kept. Nothing here
//! knows about the browser.

use microscope::run::{lit, numbers, output_with, section};

/// The command-line program.
pub const SOURCE: &str = include_str!("../../diffusion.xtl");

/// The weights, as the page's X_eTaL finds them.
pub const DATA: [(&str, &str); 6] = [
    ("data/filters.txt", include_str!("../../data/filters.txt")),
    ("data/w1.txt", include_str!("../../data/w1.txt")),
    ("data/wh.txt", include_str!("../../data/wh.txt")),
    ("data/b1.txt", include_str!("../../data/b1.txt")),
    ("data/w2.txt", include_str!("../../data/w2.txt")),
    ("data/b2.txt", include_str!("../../data/b2.txt")),
];

/// A picture's side; pictures per run; noise levels (the journey has LEVELS + 1 frames).
pub const SIDE: usize = 10;
pub const N: usize = SIDE * SIDE;
pub const PICTURES: usize = 8;
pub const LEVELS: usize = 10;

/// The program's head: the schedule, the denoiser, the step, the sampler.
pub fn head() -> &'static str {
    section(SOURCE, "\"nn:\" u_se< \"NN\"", "# -- end of the core")
}

/// A Gaussian number from two uniform ones in (0, 1), as the program's
/// u:g_auss makes them (Box-Muller).
pub fn gaussian(u1: f64, u2: f64) -> f64 {
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// PICTURES x N Gaussian numbers from a seed (the page's first run, and
/// the tests: the same every time), by a small linear generator.
pub fn seeded(seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut uniform = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    };
    (0..PICTURES * N).map(|_| gaussian(uniform(), uniform())).collect()
}

/// A run's own lines: the pictures of noise, then ten steps down with
/// every level kept, printed as one line.
pub fn tail(noise: &[f64]) -> String {
    let es: Vec<String> = noise.iter().map(|&x| lit(x)).collect();
    format!(
        "E := {PICTURES} {N} r_eshape {}\n\
         (_, _, L) := {LEVELS} '{{ (X, t, L) -> (X1, t1) := u:s_tep (X, t); (X1, t1, L c_at X1) }} p_ower (E, {LEVELS}, E)\n\
         r_avel L\n",
        es.join(" ")
    )
}

pub fn program(noise: &[f64]) -> String {
    format!("{}{}", head(), tail(noise))
}

/// The journey of PICTURES pictures from noise, as X_eTaL computed it:
/// LEVELS + 1 frames, each PICTURES x N pixels (-1 off to 1 on), the
/// noise first and the clean pictures last.
#[derive(Clone, Debug, PartialEq)]
pub struct Journey {
    pub levels: Vec<Vec<f64>>,
}

impl Journey {
    /// The clean pictures, the last level.
    pub fn finals(&self) -> &[f64] {
        self.levels.last().map_or(&[], |l| &l[..])
    }

    /// Picture i at level l, N pixels.
    pub fn picture(&self, l: usize, i: usize) -> &[f64] {
        &self.levels[l][i * N..(i + 1) * N]
    }
}

pub fn run(noise: &[f64]) -> Result<Journey, String> {
    assert_eq!(noise.len(), PICTURES * N);
    let files: Vec<(String, String)> = DATA.iter().map(|(p, t)| (p.to_string(), t.to_string())).collect();
    let (out, _) = output_with(&files, &program(noise), 1)?;
    let all: Vec<f64> = numbers(&out[0], (LEVELS + 1) * PICTURES * N)?;
    Ok(Journey { levels: all.chunks(PICTURES * N).map(|c| c.to_vec()).collect() })
}

/// How many pixels of a picture are on.
pub fn on(picture: &[f64]) -> usize {
    picture.iter().filter(|&&v| v > 0.0).count()
}
