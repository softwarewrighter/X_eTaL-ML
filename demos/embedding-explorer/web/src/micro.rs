//! The model: the demo's own X_eTaL program (embedding-explorer.xtl)
//! run once: each word's place in the 3 directions that spread the
//! words most, its cluster (k-means on the 64-dimensional embeddings),
//! its 5 nearest words. Nothing here knows about the browser.

use microscope::run::{numbers, output, section};

/// The command-line program.
pub const SOURCE: &str = include_str!("../../embedding-explorer.xtl");

/// The data the program reads, as the page's X_eTaL finds it.
pub const DATA: [(&str, &str); 2] = [("data/words.txt", include_str!("../../data/words.txt")), ("data/ids.txt", include_str!("../../data/ids.txt"))];

/// Clusters k-means finds, and neighbors per word.
pub const CLUSTERS: usize = 6;
pub const NEAR: usize = 5;

/// The words, in order (word 1 first).
pub fn words() -> Vec<&'static str> {
    DATA[0].1.lines().filter(|l| !l.is_empty()).collect()
}

/// The program's head: the corpus, the embeddings, the 3-D view.
pub fn head() -> &'static str {
    section(SOURCE, "\"em:\" u_se< \"Embed\"", "# -- end of the core")
}

/// The run's own lines: k-means' clusters of the embeddings, then what
/// the page draws.
pub fn tail() -> String {
    format!(
        "\"ls:\" u_se< \"Lists\"\n\
         K := ({CLUSTERS}, 10) ml:k_means E\n\
         r_avel xyz\n\
         K ml:a_ssign E\n\
         ls:r_aze '{{ i -> (E, {NEAR}) em:n_earest i }} m_ap r_ange v\n\
         ('+ r_/ ev) / ('+ r_/ r_avel (E - (s_hape E) r_eshape mu) * E - (s_hape E) r_eshape mu) / f_loat v\n"
    )
}

pub fn program() -> String {
    format!("{}{}", head(), tail())
}

/// The cloud, as X_eTaL computed it.
#[derive(Clone, Debug, PartialEq)]
pub struct Cloud {
    /// Each word's x, y, z.
    pub xyz: Vec<f64>,
    /// Each word's cluster, from 1.
    pub cluster: Vec<usize>,
    /// Each word's NEAR nearest words (from 1), nearest first.
    pub near: Vec<usize>,
    /// The share of the variance the 3 directions keep.
    pub kept: f64,
}

pub fn run() -> Result<Cloud, String> {
    for (path, text) in DATA {
        microscope::libs::add(path, text);
    }
    let n = words().len();
    let out = output(&program(), 4)?;
    Ok(Cloud {
        xyz: numbers(&out[0], 3 * n)?,
        cluster: numbers(&out[1], n)?,
        near: numbers(&out[2], NEAR * n)?,
        kept: numbers::<f64>(&out[3], 1)?[0],
    })
}
