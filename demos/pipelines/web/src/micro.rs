//! The model: the demo's four programs (pipelines.xtl, rosetta.xtl,
//! diffusion.xtl, equations.xtl), each example's pipeline and plain X_eTaL taken from
//! them, the program the page runs for one example, what it printed and
//! what the pipeline became. Nothing here knows about the browser.

use microscope::run::{expanded, numbers, output, section};

/// The command-line programs.
pub const PIPELINES: &str = include_str!("../../pipelines.xtl");
pub const ROSETTA: &str = include_str!("../../rosetta.xtl");
pub const DIFFUSION: &str = include_str!("../../diffusion.xtl");
pub const EQUATIONS: &str = include_str!("../../equations.xtl");

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
    /// The statements holding the equations' calls, for an example
    /// written as math too: the section prints one LaTeX line per
    /// equation, in this order.
    pub math: &'static [&'static str],
    /// A sentence about the example, and a demo it links to (title, path), shown above it.
    pub about: &'static str,
    pub link: Option<(&'static str, &'static str)>,
}

/// The examples, the two Rosetta Stones first, then the pipelines of equations.
pub const EXAMPLES: [Example; 8] = [
    Example { title: "A CNN layer, four ways", src: ROSETTA, from: "### As a paper prints it", to: "\nk:r_eport r0", pipe: "pipe", plain: "plain", checks: &["r0", "r1", "r2"], math: &["eq", "drawn"], about: "", link: None },
    Example { title: "Diffusion sampling, four ways", src: DIFFUSION, from: "### As a paper prints it", to: "\nk:r_eport r0", pipe: "pipe", plain: "plain", checks: &["r0", "r1", "r2"], math: &["u:s_tep", "eq", "drawn"], about: "", link: None },
    Example { title: "Attention as equations", src: EQUATIONS, from: "### Attention as equations", to: "### A BERT layer", pipe: "head", plain: "head0", checks: &["r1"], math: &["drawn"], about: "A pipeline whose stages are equations: each stage an Equations call reading the arrays it names (it, the previous value, or an earlier stage's label inside its strings), the whole drawn as math, one row a stage.", link: None },
    Example { title: "A BERT layer", src: EQUATIONS, from: "### A BERT layer", to: "\nk:r_eport", pipe: "layer", plain: "layer0", checks: &["r2"], math: &["shown"], about: "One encoder layer of BERT with made-up weights, every stage an equation: one head of attention, added back and layer-normed (mean, variance, normalized), the feed-forward layer with GELU written with exp, added back and layer-normed; the residuals are labels, the shapes checked.", link: Some(("Tiny BERT", "../micro-bert/")) },
    Example { title: "Dense, its bias", src: PIPELINES, from: "### Dense layer with its bias", to: "### Stable softmax", pipe: "dense", plain: "dense0", checks: &["c1", "c1b"], math: &["denseM"], about: "", link: None },
    Example { title: "Stable softmax", src: PIPELINES, from: "### Stable softmax", to: "### Layer norm", pipe: "soft", plain: "soft0", checks: &["c2", "c2b"], math: &["softM"], about: "", link: None },
    Example { title: "Layer norm", src: PIPELINES, from: "### Layer norm", to: "### Attention", pipe: "norm", plain: "norm0", checks: &["c3", "c3b"], math: &["normM"], about: "", link: None },
    Example { title: "Attention, one head", src: PIPELINES, from: "### Attention", to: "\nk:r_eport", pipe: "head", plain: "head0", checks: &["c4", "c4b"], math: &["headM"], about: "", link: None },
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
    /// The equations' LaTeX, one per `math` statement, as e:t_ex< wrote it.
    pub tex: Vec<String>,
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
    let tex = e.math.len();
    let out = output(&src, tex + 2 + e.checks.len())?;
    let all = expanded(&src)?;
    // The pipeline's statement, its name at the start of a line (eqpipe
    // := is not pipe :=): through the block's closing line.
    let from = format!("\n{} := ", e.pipe);
    let start = all.find(&from).map_or(0, |i| i + 1);
    let end = all[start..].find("\n} @)").map_or(all.len(), |i| start + i + 5);
    let expansion = all[start..end].to_string();
    Ok(Run {
        tex: out[..tex].to_vec(),
        value: numbers(&out[tex], out[tex].split_whitespace().count())?,
        checks: out[tex + 1..].to_vec(),
        expansion,
    })
}

/// The example's calls as written in its program: the equations (the
/// CNN layer's, the diffusion sampler's two), the pipeline, the plain
/// X_eTaL.
pub fn calls(e: &Example) -> (Vec<String>, String, String) {
    (e.math.iter().map(|m| statement(e.src, m)).collect(), statement(e.src, e.pipe), statement(e.src, e.plain))
}

/// What Compose starts from: an equation with its LaTeX (the dense
/// layer with its bias), and a pipeline (the stable softmax).
pub const STARTERS: [(&str, &str); 2] = [
    ("An equation", "\"e:\" u_se< \"Equations\"\nW := f_loat 2 3 r_eshape 1 2 3 4 5 6\nx := f_loat 1 1 1\nb := f_loat 10 20\n\"$$\" c_at (\"z[i] | 2\" e:t_ex< \"(sum j:=0..2 | W[i,j] * x[j]) + b[i]\") c_at \"$$\"\n\"z[i] | 2\" e:e_q< \"(sum j:=0..2 | W[i,j] * x[j]) + b[i]\"\n"),
    ("A pipeline", "\"flow:\" u_se< \"Flow\"\n\"nn:\" u_se< \"NN\"\nz := 1.0 2.0 3.0\n@ flow:p_ipe< c\"\n   z\n|> 'm_ax r_/ it $m\n|> e_xp z - m $e\n|> e / '+ r_/ e\"\n"),
];
