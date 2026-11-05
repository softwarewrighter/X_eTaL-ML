//! The page: a rank chosen on a slider; the adapted layer y = W x + s B
//! (A x) and a backward step through it, each a pipeline of equations
//! drawn as math, a stage picked to see its value and the arrays it
//! reads; the update s B A, its rank and the merge; what adapters cost
//! on a large mixture-of-experts model; LoRA and LoRA-FA trained to fit
//! a known update; the rank sweep against the truncated SVD; and the
//! programs the page ran.

use std::rc::Rc;

use gloo_timers::callback::Timeout;
use web_sys::HtmlInputElement;
use yew::prelude::*;

use microscope::chrome::{footer, header, panel};
use microscope::color::{ramp, DIVERGE};
use microscope::source::{block, NONE};
use microscope::stages::clickable;
use microscope::stageview::{no_stage, row_picker, stage_view, Split, Turn};

use crate::micro::{backward, pipeline, program, run, sweep, sweep_program, Explored, Matrix, Settings, Sweep};

/// Cell size of a heat map, in pixels.
const CELL: f64 = 22.0;

/// A matrix as a heat map, colored as the 3D stage view colors its
/// cubes (red positive, blue negative, near white about 0), the depth by
/// the size against `scale` (the largest size shown beside
/// it, so maps side by side compare); each cell's value on hover.
fn heat(m: &Matrix, scale: f64, label: &str) -> Html {
    let (w, h) = (m.cols as f64 * CELL, m.rows as f64 * CELL);
    let cells = m.items.iter().enumerate().map(|(k, v)| {
        let (y, x) = ((k / m.cols) as f64 * CELL, (k % m.cols) as f64 * CELL);
        let c = ramp(&DIVERGE, 0.5 + 0.5 * if scale > 0.0 { (v / scale).clamp(-1.0, 1.0) } else { 0.0 });
        let fill = format!("rgb({}, {}, {})", c[0], c[1], c[2]);
        html! { <rect x={x.to_string()} y={y.to_string()} width={CELL.to_string()} height={CELL.to_string()} fill={fill} class="cell"><title>{format!("{v:.4}")}</title></rect> }
    });
    html! {
        <figure class="heat">
            <svg width={w.to_string()} height={h.to_string()} viewBox={format!("0 0 {w} {h}")} role="img" aria-label={label.to_string()}>
                { for cells }
                <rect x="0" y="0" width={w.to_string()} height={h.to_string()} class="frame" />
            </svg>
            <figcaption>{label}</figcaption>
        </figure>
    }
}

/// The largest size in some matrices.
fn largest(ms: &[&Matrix]) -> f64 {
    ms.iter().flat_map(|m| m.items.iter()).fold(0.0, |a: f64, v| a.max(v.abs()))
}

/// A vector as a one-column matrix.
fn column(items: &[f64]) -> Matrix {
    Matrix { rows: items.len(), cols: 1, items: items.to_vec() }
}

/// Stage k of the pipeline (from 0, the data first) in 2D: what it
/// computes, from what, the matrices it multiplies by included.
fn arrays(e: &Explored, k: usize, s: Settings) -> Html {
    let v = |i: usize| column(&e.stages[i].items);
    let (x, h, u, y) = (v(0), v(1), v(2), v(3));
    let op = |t: &str| html! { <span class="op">{t.to_string()}</span> };
    match k {
        0 => html! { <div class="row">{ heat(&x, largest(&[&x]), "x: the input, 8 numbers") }</div> },
        1 => {
            let sc = largest(&[&e.a, &x, &h]);
            html! { <div class="row">{ heat(&e.a, sc, &format!("A: {} x 8, trained", s.r)) }{ op("\u{00d7}") }{ heat(&x, sc, "x") }{ op("=") }{ heat(&h, sc, &format!("h = A x: {} numbers", s.r)) }</div> }
        }
        2 => {
            let sc = largest(&[&e.b, &h, &u]);
            html! { <div class="row">{ heat(&e.b, sc, &format!("B: 8 x {}, trained", s.r)) }{ op("\u{00d7}") }{ heat(&h, sc, "h") }{ op("=") }{ heat(&u, sc, "u = B h: 8 numbers") }</div> }
        }
        _ => {
            let sc = largest(&[&e.w, &x, &u, &y]);
            html! { <div class="row">{ heat(&e.w, sc, "W: 8 x 8, frozen") }{ op("\u{00d7}") }{ heat(&x, sc, "x") }{ op("+ s \u{00d7}") }{ heat(&u, sc, "u") }{ op("=") }{ heat(&y, sc, "y") }</div> }
        }
    }
}

/// Stage k of the backward step (from 0, g first) in 2D.
fn back_arrays(e: &Explored, k: usize, s: Settings) -> Html {
    let v = |i: usize| column(&e.back_stages[i].items);
    let m = |i: usize, rows: usize, cols: usize| Matrix { rows, cols, items: e.back_stages[i].items.clone() };
    let (x, h) = (column(&e.stages[0].items), column(&e.stages[1].items));
    let (g, b, da, db, dx) = (v(0), v(1), m(2, s.r, 8), m(3, 8, s.r), v(4));
    let op = |t: &str| html! { <span class="op">{t.to_string()}</span> };
    let bt = Matrix { rows: s.r, cols: 8, items: (0..s.r * 8).map(|k| e.b.items[(k % 8) * s.r + k / 8]).collect() };
    let xt = Matrix { rows: 1, cols: 8, items: x.items.clone() };
    let ht = Matrix { rows: 1, cols: s.r, items: h.items.clone() };
    match k {
        0 => html! { <div class="row">{ heat(&g, largest(&[&g]), "g = y - (W + T) x: the loss's gradient at the output") }</div> },
        1 => {
            let sc = largest(&[&bt, &g, &b]);
            html! { <div class="row">{ heat(&bt, sc, &format!("B': {} x 8", s.r)) }{ op("\u{00d7}") }{ heat(&g, sc, "g") }{ op("=") }{ heat(&b, sc, &format!("b = B' g: {} numbers", s.r)) }</div> }
        }
        2 => {
            let sc = largest(&[&b, &xt, &da]);
            html! { <div class="row">{ op("s \u{00d7}") }{ heat(&b, sc, "b") }{ op("\u{00d7}") }{ heat(&xt, sc, "x'") }{ op("=") }{ heat(&da, sc, &format!("dA: {} x 8", s.r)) }</div> }
        }
        3 => {
            let sc = largest(&[&g, &ht, &db]);
            html! { <div class="row">{ op("s \u{00d7}") }{ heat(&g, sc, "g") }{ op("\u{00d7}") }{ heat(&ht, sc, "h' = (A x)'") }{ op("=") }{ heat(&db, sc, &format!("dB: 8 x {}", s.r)) }</div> }
        }
        _ => {
            let sc = largest(&[&g, &b, &dx]);
            html! { <div class="row">{ op("W' \u{00d7}") }{ heat(&g, sc, "g") }{ op("+ s A' \u{00d7}") }{ heat(&b, sc, "b") }{ op("=") }{ heat(&dx, sc, "dx: back to the layer before") }</div> }
        }
    }
}

/// What each stage means, for its button: the forward pass, the backward step.
const STAGES: [&str; 4] = ["x: the input", "h = A x", "u = B h", "y = W x + s u"];
const BACK: [&str; 5] = ["g = y - wanted", "b = B' g", "dA = s b x'", "dB = s g h'", "dx = W' g + s A' b"];

/// Width and height of a chart, and its margins for the axes' labels.
const CW: f64 = 520.0;
const CH: f64 = 240.0;
const ML: f64 = 54.0;
const MB: f64 = 28.0;

/// Lines on a log scale: each series' points at x = 0, 1, ...; `xs` the
/// label under each point (some left empty); values under 1e-8 drawn
/// at 1e-8 (log 0 has no place).
fn chart(label: &str, series: &[(&str, &[f64], &str)], xs: &[String]) -> Html {
    let floor = 1e-8_f64;
    let all = series.iter().flat_map(|(_, v, _)| v.iter()).map(|v| v.max(floor));
    let (lo, hi) = all.fold((f64::MAX, f64::MIN), |(a, b), v| (a.min(v.log10()), b.max(v.log10())));
    let (lo, hi) = (lo.floor(), hi.ceil().max(lo.floor() + 1.0));
    let n = series.iter().map(|(_, v, _)| v.len()).max().unwrap_or(1).max(2) - 1;
    let px = |i: usize| ML + (CW - ML - 10.0) * i as f64 / n as f64;
    let py = |v: f64| 8.0 + (CH - MB - 8.0) * (hi - v.max(floor).log10()) / (hi - lo);
    let lines = series.iter().map(|(name, v, class)| {
        let pts: Vec<String> = v.iter().enumerate().map(|(i, &y)| format!("{:.1},{:.1}", px(i), py(y))).collect();
        html! { <polyline points={pts.join(" ")} class={format!("line {class}")}><title>{name.to_string()}</title></polyline> }
    });
    let ticks = (lo as i32..=hi as i32).map(|e| {
        let y = py(10f64.powi(e));
        html! { <g><line x1={ML.to_string()} x2={(CW - 10.0).to_string()} y1={y.to_string()} y2={y.to_string()} class="grid" /><text x={(ML - 6.0).to_string()} y={(y + 4.0).to_string()} class="tick end">{format!("1e{e}")}</text></g> }
    });
    let under = xs.iter().enumerate().filter(|(_, t)| !t.is_empty()).map(|(i, t)| html! { <text x={px(i).to_string()} y={(CH - 8.0).to_string()} class="tick mid">{t.clone()}</text> });
    let keys = series.iter().map(|(name, _, class)| html! { <span class="key"><span class={format!("swatch {class}")}></span>{name.to_string()}</span> });
    html! {
        <figure class="chart">
            <svg width={CW.to_string()} height={CH.to_string()} viewBox={format!("0 0 {CW} {CH}")} role="img" aria-label={label.to_string()}>
                { for ticks }{ for under }{ for lines }
            </svg>
            <figcaption>{ for keys }</figcaption>
        </figure>
    }
}

/// Which pipeline the math shows.
#[derive(Clone, Copy, PartialEq)]
enum Pass {
    Forward,
    Backward,
}

/// A whole number typed in a box, kept within `lo..=hi`.
fn number(label: &str, value: usize, (lo, hi): (usize, usize), set: Callback<usize>) -> Html {
    let change = Callback::from(move |e: Event| {
        let i: HtmlInputElement = e.target_unchecked_into();
        if let Ok(v) = i.value().trim().parse::<usize>() {
            set.emit(v.clamp(lo, hi));
        }
    });
    html! { <label class="slider">{label}
        <input type="number" min={lo.to_string()} max={hi.to_string()} value={value.to_string()} onchange={change} />
    </label> }
}

/// A number with thousands separated by commas.
fn commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[function_component(App)]
pub fn app() -> Html {
    let settings = use_state(Settings::default);
    // The stage chosen (from 1, the data first), by a row of the math or
    // a button; how the 3D blocks are turned.
    let stage = use_state(|| None::<usize>);
    let pass = use_state(|| Pass::Forward);
    let turn = use_state(Turn::default);
    // The rank sweep, run once, after the first run is drawn.
    let swept = use_state(|| None::<Rc<Result<Sweep, String>>>);
    {
        let swept = swept.clone();
        use_effect_with((), move |_| {
            let t = Timeout::new(900, move || swept.set(Some(Rc::new(sweep()))));
            move || drop(t)
        });
    }
    // A run, once per settings, after the spinner is drawn.
    let ran = use_state(|| None::<(Settings, Rc<Result<Explored, String>>)>);
    {
        let ran = ran.clone();
        use_effect_with(*settings, move |&s| {
            let t = Timeout::new(40, move || ran.set(Some((s, Rc::new(run(s))))));
            move || drop(t)
        });
    }
    let s = *settings;
    let set = |f: fn(Settings, usize) -> Settings| {
        let settings = settings.clone();
        Callback::from(move |v: usize| settings.set(f(*settings, v)))
    };
    let rank = {
        let set_r = set(|s, r| Settings { r, ..s });
        let change = Callback::from(move |e: Event| {
            let i: HtmlInputElement = e.target_unchecked_into();
            if let Ok(v) = i.value().parse::<usize>() {
                set_r.emit(v);
            }
        });
        html! { <div class="controls"><label class="slider">{format!("rank r = {}", s.r)}
            <input type="range" min="1" max="8" step="1" value={s.r.to_string()} onchange={change} />
        </label><span class="muted">{format!("A is {} x 8, B is 8 x {}, s = alpha / r = 6 / {} = {}", s.r, s.r, s.r, 6.0 / s.r as f64)}</span></div> }
    };

    let result = ran.as_ref().filter(|(k, _)| *k == s).map(|(_, r)| r.clone());
    // The chosen stage on the right: its numbers, and in 3D what it reads
    // and gives (the shared stage view, as the pipelines demo shows it).
    let right = match (result.as_deref(), *stage, *pass) {
        (Some(Ok(e)), Some(k), Pass::Forward) => stage_view(&e.stages, pipeline(), k, "of the adapted layer", &turn),
        (Some(Ok(e)), Some(k), Pass::Backward) => stage_view(&e.back_stages, backward(), k, "of the backward step", &turn),
        _ => no_stage(),
    };
    let body = match result {
        None => html! { <div class="running"><span class="spinner"></span>{format!("X_eTaL is running the adapted layer at rank {} in your browser\u{2026}", s.r)}</div> },
        Some(r) => match r.as_ref() {
            Err(e) => html! { <pre class="err">{ e.clone() }</pre> },
            Ok(e) => {
                let names: &[&str] = match *pass { Pass::Forward => &STAGES, Pass::Backward => &BACK };
                let buttons = names.iter().enumerate().map(|(k, t)| {
                    let st = stage.clone();
                    html! { <button class={classes!((*stage == Some(k + 1)).then_some("on"))} onclick={Callback::from(move |_| st.set(Some(k + 1)))}>{*t}</button> }
                });
                let sc = largest(&[&e.w, &e.delta, &e.merged]);
                let st = stage.clone();
                let pick = row_picker(Callback::from(move |k| st.set(Some(k))));
                let shown = clickable(if *pass == Pass::Forward { &e.tex } else { &e.back_tex }, *stage);
                let switch = |p: Pass, t: &str| {
                    let (ps, st) = (pass.clone(), stage.clone());
                    html! { <button class={classes!((*pass == p).then_some("on"))} onclick={Callback::from(move |_| { ps.set(p); st.set(None); })}>{t.to_string()}</button> }
                };
                let r = s.r;
                let at = |v: &[f64]| *v.last().unwrap_or(&0.0);
                let tsc = largest(&[&e.t, &e.learned, &e.learned_fa]);
                let steps: Vec<String> = (0..e.loss.len()).map(|i| if i % 5 == 0 { (10 * i).to_string() } else { String::new() }).collect();
                let best = vec![e.best[r - 1]; e.loss.len()];
                let sweep_panel = match swept.as_deref() {
                    None => html! { <div class="running"><span class="spinner"></span>{"X_eTaL is training LoRA and LoRA-FA at every rank, 1 to 8, in your browser\u{2026}"}</div> },
                    Some(Err(err)) => html! { <pre class="err">{ err.clone() }</pre> },
                    Some(Ok(w)) => {
                        let ranks: Vec<String> = (1..=8).map(|k| k.to_string()).collect();
                        let rows = (0..8).map(|k| html! { <tr><td>{k + 1}</td><td>{format!("{:.6}", e.best[k])}</td><td>{format!("{:.6}", w.lora[k])}</td><td>{format!("{:.6}", w.fa[k])}</td></tr> });
                        html! { <>
                            { chart("the loss at each rank", &[("best possible (truncated SVD)", &e.best, "best"), ("LoRA", &w.lora, "lora"), ("LoRA-FA", &w.fa, "fa")], &ranks) }
                            <table class="facts sweep"><tr><th>{"rank"}</th><th>{"best possible"}</th><th>{"LoRA"}</th><th>{"LoRA-FA"}</th></tr>{ for rows }</table>
                        </> }
                    }
                };
                html! { <>
                    { panel(if *pass == Pass::Forward { "The adapted layer" } else { "A backward step" }, if *pass == Pass::Forward { "y := @ flow:p_ipe< ..." } else { "back := @ flow:p_ipe< ..." }, "x goes two ways: through the frozen W, and through the adapter, A first (r numbers), then B (back to 8); the two are added. The factors stay apart: the 8 x 8 matrix B A is never made. Backward, the loss's gradient g at the output goes back the same two ways: B' g once (r numbers), then the gradients of A and B (outer products) and of x. Click a row of the math, or a button, to see the stage: on the right its numbers and, in 3D, the stages it reads and what it gives (drag to turn); here what it computes from what, the matrices too (red positive, blue negative, near white about 0, as in 3D; hover a cell for its value).", false, html! { <>
                        <div class="stages">{ switch(Pass::Forward, "Forward: y = W x + s B (A x)") }{ switch(Pass::Backward, "Backward: the gradients") }</div>
                        <div class="tex" onclick={pick} data-tex={shown.clone()}>{ shown.clone() }</div>
                        <div class="stages">{ for buttons }</div>
                        { match (*stage, *pass) { (Some(k), Pass::Forward) => arrays(e, k - 1, s), (Some(k), Pass::Backward) => back_arrays(e, k - 1, s), (None, _) => html! {} } }
                        <p class="muted">{ if e.back_same { "The backward pipeline's dx is the library's lo:g_rads (within 1e-12); every gradient of lo:g_rads is checked against finite differences in the library's tests." } else { "The backward pipeline's dx differs from lo:g_rads!" } }</p>
                    </> }) }
                    { panel("The update and the merge", "lo:m_erge (W, A, B, s)", "The update s B A is an 8 x 8 matrix of rank r at most; added to W it is the merged matrix, one matrix again, as fast as the base, for deploying one task.", false, html! { <>
                        <div class="row">{ heat(&e.w, sc, "W, frozen") }<span class="op">{"+"}</span>{ heat(&e.delta, sc, &format!("s B A, rank {}", e.rank)) }<span class="op">{"="}</span>{ heat(&e.merged, sc, "W + s B A, merged") }</div>
                        <table class="facts">
                            <tr><td>{"the update's rank (lo:r_ank, by Gram-Schmidt)"}</td><td>{ e.rank }</td></tr>
                            <tr><td>{"the merged layer gives the factored layer's output (within 1e-12)"}</td><td>{ if e.merged_same { "yes" } else { "no" } }</td></tr>
                            <tr><td>{"numbers trained (A and B), against W's"}</td><td>{ format!("{} against {}{}", e.trained.0, e.trained.1, if e.trained.0 >= e.trained.1 { ": no saving at this rank" } else { "" }) }</td></tr>
                            <tr><td>{"multiply-adds per input: B (A x), then making B A first"}</td><td>{ format!("{} and {}", e.costs.0, e.costs.1) }</td></tr>
                        </table>
                    </> }) }
                    { panel("On a large model", "lo:e_xpert", "Adapters at the same rank on a mixture-of-experts model: each expert a SwiGLU block of three matrices (gate and up, f x d; down, d x f). Change the sizes; the rank is the slider's. At rank 16 with d 4096 and f 14336 an expert's adapters train 884,736 numbers, 1.69 MiB in BF16.", false, html! { <>
                        <div class="controls sizes">
                            { number("d", s.d, (1, 65536), set(|s, d| Settings { d, ..s })) }
                            { number("f", s.f, (1, 262144), set(|s, f| Settings { f, ..s })) }
                            { number("experts a layer", s.experts, (1, 1024), set(|s, experts| Settings { experts, ..s })) }
                            { number("layers", s.layers, (1, 256), set(|s, layers| Settings { layers, ..s })) }
                        </div>
                        <table class="facts">
                            <tr><td>{"numbers trained per expert, 3 r (d + f)"}</td><td>{ commas(e.expert) }</td></tr>
                            <tr><td>{"one expert, BF16"}</td><td>{ format!("{:.2} MiB", e.mib.0) }</td></tr>
                            <tr><td>{"a layer of experts"}</td><td>{ format!("{:.1} MiB", e.mib.1) }</td></tr>
                            <tr><td>{"every layer"}</td><td>{ format!("{:.2} GiB", e.mib.2 / 1024.0) }</td></tr>
                        </table>
                    </> }) }
                    { panel("Training: LoRA beside LoRA-FA", "lr lo:a_dam (state, (I, Y))", "Both learn the same known update T (8 x 8, of rank 4) from the same start, B at 0 and the same seeded A: the inputs are the 8 unit vectors, so the loss is half the squared distance between s B A and T. LoRA trains A and B; LoRA-FA freezes A and trains B alone, half the numbers. 300 Adam steps each, at the rank on the slider. No update of rank r can do better than the truncated SVD of T (Eckart-Young): LoRA reaches it; LoRA-FA cannot, as its update keeps its rows in A's row space, wherever B goes.", false, html! { <>
                        { chart("the loss as each trains", &[("LoRA", &e.loss, "lora"), ("LoRA-FA", &e.loss_fa, "fa"), (&format!("best possible at rank {r}"), &best, "best")], &steps) }
                        <p class="muted">{"Loss after each 10 Adam steps (log scale); steps along the bottom."}</p>
                        <table class="facts">
                            <tr><td>{format!("LoRA's loss after 300 steps (trains {} numbers)", 16 * r)}</td><td>{format!("{:.6}", at(&e.loss))}</td></tr>
                            <tr><td>{format!("LoRA-FA's (trains {} numbers)", 8 * r)}</td><td>{format!("{:.6}", at(&e.loss_fa))}</td></tr>
                            <tr><td>{format!("the best any rank-{r} update can do: half T's squared singular values past the {r}th")}</td><td>{format!("{:.6}", e.best[r - 1])}</td></tr>
                        </table>
                        <div class="row">{ heat(&e.t, tsc, "T: the update to learn, rank 4") }{ heat(&e.learned, tsc, &format!("LoRA's s B A, rank {r}")) }{ heat(&e.learned_fa, tsc, &format!("LoRA-FA's s B A, rank {r}")) }</div>
                    </> }) }
                    { panel("The rank sweep", "'u:l_ora e_ach r_ange 8", "LoRA and LoRA-FA trained at every rank from 1 to 8 (300 Adam steps each), against the best possible at that rank, from T's singular values (found by power iteration). T is of rank 4, so from rank 4 on the best is 0. LoRA follows the best at every rank; LoRA-FA needs a larger rank for the same loss. This runs once, apart from the slider.", false, sweep_panel) }
                </> }
            }
        },
    };

    html! {
        <>
        <header>
            { header("LoRA explorer", "Low-rank adaptation, computed by X_eTaL in your browser: a frozen 8 x 8 weight matrix W adapted by two small trainable ones, A (r x 8) and B (8 x r), the layer y = W x + s B (A x). Choose the rank; see each stage forward and backward, the update and its rank, the merge, what adapters cost on a large mixture-of-experts model, and LoRA and LoRA-FA trained to fit a known update against the best possible at each rank.") }
            { rank }
        </header>
        <main><div class="wide">
            <Split idle={stage.is_none()} close={{ let st = stage.clone(); Callback::from(move |_| st.set(None)) }} {right} left={html! { <>
            { body }
            <section class="panel code">
                <h2>{"The program"}</h2>
                <p class="note">{"Everything the page runs, in your browser, at each setting: lora-explorer.xtl with the rank and sizes you set, up to its rank sweep, then what the page reads back (the two pipelines drawn as math, every stage's value, the matrices)."}</p>
                { block(&program(s), NONE) }
                <p class="note">{"And once, for the rank sweep: the whole of lora-explorer.xtl at its defaults (the sweep's two lines are its last)."}</p>
                { block(sweep_program(), NONE) }
            </section>
            </> }} />
        </div></main>
        { footer() }
        </>
    }
}
