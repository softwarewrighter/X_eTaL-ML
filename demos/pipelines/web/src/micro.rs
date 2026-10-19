//! The model: the demo's two programs (pipelines.xtl, rosetta.xtl), each
//! example's pipeline and plain X_eTaL taken from them, the program the
//! page runs for one example, what it printed and what the pipeline
//! became. Nothing here knows about the browser.

use microscope::run::{expanded, numbers, output, section};

/// The command-line programs.
pub const PIPELINES: &str = include_str!("../../pipelines.xtl");
pub const ROSETTA: &str = include_str!("../../rosetta.xtl");

/// One example: its section of a program, the names its pipeline and
/// its plain X_eTaL are bound to, and the checks that they agree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Example {
    pub title: &'static str,
    pub src: &'static str,
    /// The section: from this heading up to `to`.
    pub from: &'static str,
    pub to: &'static str,
    pub pipe: &'static str,
    pub plain: &'static str,
    pub checks: &'static [&'static str],
    /// The equation's call, for the example written as math too.
    pub math: Option<&'static str>,
}

/// The examples, the CNN layer first.
pub const EXAMPLES: [Example; 5] = [
    Example { title: "A CNN layer, four ways", src: ROSETTA, from: "### As a paper prints it", to: "\nk:r_eport r1", pipe: "pipe", plain: "plain", checks: &["r1", "r2"], math: Some("eq") },
    Example { title: "Dense, its bias", src: PIPELINES, from: "### Dense layer with its bias", to: "### Stable softmax", pipe: "dense", plain: "dense0", checks: &["c1"], math: None },
    Example { title: "Stable softmax", src: PIPELINES, from: "### Stable softmax", to: "### Layer norm", pipe: "soft", plain: "soft0", checks: &["c2"], math: None },
    Example { title: "Layer norm", src: PIPELINES, from: "### Layer norm", to: "### Attention", pipe: "norm", plain: "norm0", checks: &["c3"], math: None },
    Example { title: "Attention, one head", src: PIPELINES, from: "### Attention", to: "\nk:r_eport", pipe: "head", plain: "head0", checks: &["c4"], math: None },
];

/// What a program starts with: its comment, the imports and its data
/// (up to the first section after `### Data`).
pub fn head(src: &str) -> &str {
    let data = src.find("### Data").unwrap_or(0);
    let next = src[data + 1..].find("\n### ").map_or(src.len(), |i| data + 2 + i);
    &src[..next]
}

/// The statement binding `name` in `src`: its line and the lines that
/// continue it (indented, or a pipeline's `|>` stages).
pub fn statement(src: &str, name: &str) -> String {
    let start = src.find(&format!("\n{name} := ")).map_or(0, |i| i + 1);
    let mut lines = src[start..].lines();
    let mut out = vec![lines.next().unwrap_or("")];
    out.extend(lines.take_while(|l| l.starts_with(' ') || l.starts_with("|>")));
    out.join("\n")
}

/// The program the page runs for `e`: the head, the example's section,
/// then its pipeline's value (r_avel) and the report of its checks.
pub fn program(e: &Example) -> String {
    let body = section(e.src, e.from, e.to).trim_end();
    format!("{}{body}\n\n# -- the run --------------------------------------------\nr_avel {}\nk:r_eport {}\n", head(e.src), e.pipe, e.checks.join(" k:a_nd "))
}

/// What one run gave.
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    /// The equation's LaTeX (the CNN layer), as e:t_ex< wrote it.
    pub tex: Option<String>,
    /// The pipeline's value, its items in order.
    pub value: Vec<f64>,
    /// The checks' lines, then their summary.
    pub checks: Vec<String>,
    /// What the pipeline's statement became (`xetal expand`).
    pub expansion: String,
}

/// Run `e`'s program and expand it.
pub fn run(e: &Example) -> Result<Run, String> {
    let src = program(e);
    let tex = usize::from(e.math.is_some());
    let out = output(&src, tex + 2 + e.checks.len())?;
    let all = expanded(&src)?;
    let from = format!("{} := ", e.pipe);
    // Up to the plain statement, less the comments that head its section.
    let cut = section(&all, &from, &format!("\n{} := ", e.plain));
    let lines: Vec<&str> = cut.lines().collect();
    let keep = lines.iter().rposition(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#')).map_or(0, |i| i + 1);
    let expansion = lines[..keep].join("\n");
    Ok(Run {
        tex: (tex == 1).then(|| out[0].clone()),
        value: numbers(&out[tex], out[tex].split_whitespace().count())?,
        checks: out[tex + 1..].to_vec(),
        expansion,
    })
}

/// The example's calls as written in its program: the equation (the
/// CNN layer's), the pipeline, the plain X_eTaL.
pub fn calls(e: &Example) -> (Option<String>, String, String) {
    (e.math.map(|m| statement(e.src, m)), statement(e.src, e.pipe), statement(e.src, e.plain))
}
