//! The model: the demo's own X_eTaL program (k-means.xtl) run for one
//! step of k-means from the centers the page holds (or from a start).
//! Nothing here knows about the browser.

use microscope::run::{lit, numbers, output_pictures, section};

/// The command-line program.
pub const SOURCE: &str = include_str!("../../k-means.xtl");
/// The Learn library, for the step the page quotes.
pub const LEARN: &str = include_str!("../../../../libs/Learn/src/Learn.xtl");

/// Points, and the map's side.
pub const POINTS: usize = 300;
pub const SIDE: usize = 40;

/// How the centers start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Start {
    /// The first k points: all in the first blob.
    Poor,
    /// Learn's farthest-first start.
    Farthest,
}

impl Start {
    pub fn expr(self, k: usize) -> String {
        match self {
            Start::Poor => format!("{k} t_ake X"),
            Start::Farthest => format!("{k} ml:f_arthest X"),
        }
    }
}

/// The program's head: the points and the map's grid.
pub fn head() -> &'static str {
    section(SOURCE, "\"ml:\" u_se< \"Learn\"", "# -- end of the core")
}

/// Learn's k_step, as the library writes it (the page quotes it).
pub fn k_step() -> &'static str {
    let from = LEARN.find("l:k_step := {").unwrap_or(0);
    let to = LEARN[from..].find("\n}").map_or(LEARN.len(), |e| from + e + 2);
    &LEARN[from..to]
}

/// Earlier inertias and this run's, as X_eTaL.
fn history(earlier: &[f64]) -> String {
    match earlier.is_empty() {
        true => "C1 ml:i_nertia X".to_string(),
        false => format!("({}) c_at C1 ml:i_nertia X", earlier.iter().map(|&v| lit(v)).collect::<Vec<_>>().join(" ")),
    }
}

/// A run's own lines: the centers to start from (a start, or the
/// page's centers C and one step to C1), what the page draws, then the
/// inertia after every step so far drawn by Plot.
pub fn tail(k: usize, start: Start, centers: Option<&[f64]>, earlier: &[f64]) -> String {
    let first = match centers {
        None => format!("C1 := {}\n", start.expr(k)),
        Some(c) => format!("C := {k} 2 r_eshape {}\nC1 := X ml:k_step C\n", c.iter().map(|&v| lit(v)).collect::<Vec<_>>().join(" ")),
    };
    let same = match centers {
        None => "0".to_string(),
        Some(_) => "(C1 ml:a_ssign X) m_atch C ml:a_ssign X".to_string(),
    };
    format!(
        "{first}r_avel C1\nC1 ml:a_ssign X\nC1 ml:a_ssign grid\nC1 ml:i_nertia X\n{same}\nr_avel X\n\
         \"p:\" u_se< \"Plot\"\ninertia := {}\n\
         chart := (\"The inertia\" \"step\" \"\" \"\") p:c_hart! 1 r_eshape e_nclose inertia\n",
        history(earlier)
    )
}

pub fn program(k: usize, start: Start, centers: Option<&[f64]>, earlier: &[f64]) -> String {
    format!("{}{}", head(), tail(k, start, centers, earlier))
}

/// After a run, as X_eTaL computed it.
#[derive(Clone, Debug, PartialEq)]
pub struct After {
    /// The centers, k rows of x and y.
    pub centers: Vec<f64>,
    /// The center (from 1) each point and each map point is nearest.
    pub points_in: Vec<usize>,
    pub map: Vec<usize>,
    pub inertia: f64,
    /// No point changed center in this step.
    pub settled: bool,
    /// The points, x y pairs.
    pub points: Vec<f64>,
    /// Plot's chart of the inertia so far, SVG.
    pub pictures: Vec<String>,
}

pub fn run(k: usize, start: Start, centers: Option<&[f64]>, earlier: &[f64]) -> Result<After, String> {
    let (out, pictures) = output_pictures(&program(k, start, centers, earlier), 6)?;
    Ok(After {
        centers: numbers(&out[0], 2 * k)?,
        points_in: numbers(&out[1], POINTS)?,
        map: numbers(&out[2], SIDE * SIDE)?,
        inertia: numbers::<f64>(&out[3], 1)?[0],
        settled: numbers::<usize>(&out[4], 1)?[0] == 1,
        points: numbers(&out[5], 2 * POINTS)?,
        pictures,
    })
}
