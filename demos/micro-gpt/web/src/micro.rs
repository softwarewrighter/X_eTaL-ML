//! The model: the demo's own X_eTaL program (micro-gpt.xtl): the tiny
//! GPT's weights, its forward pass and its generator, run with uniform
//! numbers the page supplies. Nothing here knows about the browser.

use microscope::run::{lit, numbers, output, section};

/// The command-line program.
pub const SOURCE: &str = include_str!("../../micro-gpt.xtl");

/// The weights and the letters, as the page's X_eTaL finds them.
pub const DATA: [(&str, &str); 10] = [
    ("data/chars.txt", include_str!("../../data/chars.txt")),
    ("data/wte.txt", include_str!("../../data/wte.txt")),
    ("data/wpe.txt", include_str!("../../data/wpe.txt")),
    ("data/lm_head.txt", include_str!("../../data/lm_head.txt")),
    ("data/wq.txt", include_str!("../../data/wq.txt")),
    ("data/wk.txt", include_str!("../../data/wk.txt")),
    ("data/wv.txt", include_str!("../../data/wv.txt")),
    ("data/wo.txt", include_str!("../../data/wo.txt")),
    ("data/fc1.txt", include_str!("../../data/fc1.txt")),
    ("data/fc2.txt", include_str!("../../data/fc2.txt")),
];

/// Names per set, letters (uniform numbers) per name, tokens.
pub const NAMES: usize = 10;
pub const STEPS: usize = 16;
pub const TOKENS: usize = 27;

/// The letters, a to z.
pub fn chars() -> Vec<char> {
    DATA[0].1.trim().chars().collect()
}

/// The program's head: the model, its forward pass, the generator.
pub fn head() -> &'static str {
    section(SOURCE, "\"nn:\" u_se< \"NN\"", "# -- end of the core")
}

/// Starting letters as they may stand in the program: a to z only, at most 8.
pub fn clean(start: &str) -> String {
    start.to_lowercase().chars().filter(|c| c.is_ascii_lowercase()).take(8).collect()
}

/// A run's own lines: the uniform numbers, NAMES names at temperature
/// t from the start letters, then for the name numbered `pick` its
/// probabilities at every step and each head's attention.
pub fn tail(t: f64, start: &str, u: &[f64], pick: usize) -> String {
    let s = match clean(start).as_str() {
        "" => "1 r_eshape v".to_string(),
        letters => format!("v c_at chars i_ndexOf \"{letters}\""),
    };
    let us: Vec<String> = u.iter().map(|&x| lit(x)).collect();
    format!(
        "U := {NAMES} {STEPS} r_eshape {}\n\
         names := '{{ i -> ({}, {s}) u:n_ame i s_elect U }} m_ap r_ange {NAMES}\n\
         \",\" t:j_oin names\n\
         ids := v c_at chars i_ndexOf d_isclose {pick} s_elect names\n\
         r_avel {} sm:p_robs u:g_pt ids\n\
         n := t_ally ids\n\
         h := u:r_ms u:r_ms (ids s_elect wte) + (r_ange n) s_elect wpe\n\
         u:l_ooks := {{ j -> r := (4 * j - 1) + r_ange 4; (at:c_ausal n) at:w_eights (o_\\ r s_elect Wq, o_\\ r s_elect Wk) at:s_cores h }}\n\
         r_avel u:l_ooks 1\nr_avel u:l_ooks 2\nr_avel u:l_ooks 3\nr_avel u:l_ooks 4\n",
        us.join(" "),
        lit(t),
        lit(t)
    )
}

pub fn program(t: f64, start: &str, u: &[f64], pick: usize) -> String {
    format!("{}{}", head(), tail(t, start, u, pick))
}

/// A set of names, as X_eTaL generated them, and one looked into.
#[derive(Clone, Debug, PartialEq)]
pub struct Names {
    pub names: Vec<String>,
    /// For the picked name: its tokens' count n (the start, then its letters).
    pub n: usize,
    /// The probability of each token after each of the n, n x 27.
    pub probs: Vec<f64>,
    /// Each head's attention, n x n.
    pub heads: Vec<Vec<f64>>,
}

pub fn run(t: f64, start: &str, u: &[f64], pick: usize) -> Result<Names, String> {
    for (path, text) in DATA {
        microscope::libs::add(path, text);
    }
    let out = output(&program(t, start, u, pick), 6)?;
    let names: Vec<String> = out[0].split(',').map(str::to_string).collect();
    let n = 1 + names.get(pick.saturating_sub(1)).map_or(0, |w| w.chars().count());
    Ok(Names {
        probs: numbers(&out[1], n * TOKENS)?,
        heads: (2..6).map(|i| numbers(&out[i], n * n)).collect::<Result<_, _>>()?,
        n,
        names,
    })
}

/// Uniform numbers for a set of names: the browser's own randomness on
/// the page; a fixed sequence natively (the tests).
pub fn uniforms(seed: u64) -> Vec<f64> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = seed;
        (0..NAMES * STEPS).map(|_| js_sys::Math::random()).collect()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..NAMES * STEPS)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((x >> 11) as f64) / (1u64 << 53) as f64
            })
            .collect()
    }
}
