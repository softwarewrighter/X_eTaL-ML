//! The page: a rank chosen on a slider; the adapted layer y = W x + s B
//! (A x) as a pipeline of equations drawn as math, a stage picked to
//! see its value and the arrays it reads; the update s B A, its rank and
//! the merge; what adapters cost on a large mixture-of-experts model;
//! and the program the page ran.

use std::rc::Rc;

use gloo_timers::callback::Timeout;
use web_sys::HtmlInputElement;
use yew::prelude::*;

use microscope::chrome::{footer, header, panel};
use microscope::color::{ramp, DIVERGE};
use microscope::source::{block, NONE};
use microscope::stages::clickable;
use microscope::stageview::{no_stage, row_picker, stage_view, Split, Turn};

use crate::micro::{pipeline, program, run, Explored, Matrix, Settings};

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

/// What each stage means, for its button.
const STAGES: [&str; 4] = ["x: the input", "h = A x", "u = B h", "y = W x + s u"];

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
    let turn = use_state(Turn::default);
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
    let right = match (result.as_deref(), *stage) {
        (Some(Ok(e)), Some(k)) => stage_view(&e.stages, pipeline(), k, "of the adapted layer", &turn),
        _ => no_stage(),
    };
    let body = match result {
        None => html! { <div class="running"><span class="spinner"></span>{format!("X_eTaL is running the adapted layer at rank {} in your browser\u{2026}", s.r)}</div> },
        Some(r) => match r.as_ref() {
            Err(e) => html! { <pre class="err">{ e.clone() }</pre> },
            Ok(e) => {
                let buttons = STAGES.iter().enumerate().map(|(k, t)| {
                    let st = stage.clone();
                    html! { <button class={classes!((*stage == Some(k + 1)).then_some("on"))} onclick={Callback::from(move |_| st.set(Some(k + 1)))}>{*t}</button> }
                });
                let sc = largest(&[&e.w, &e.delta, &e.merged]);
                let st = stage.clone();
                let pick = row_picker(Callback::from(move |k| st.set(Some(k))));
                let shown = clickable(&e.tex, *stage);
                html! { <>
                    { panel("The adapted layer", "y := @ flow:p_ipe< ...", "x goes two ways: through the frozen W, and through the adapter, A first (r numbers), then B (back to 8); the two are added. The factors stay apart: the 8 x 8 matrix B A is never made. Click a row of the math, or a button, to see the stage: on the right its numbers and, in 3D, the stages it reads and what it gives (drag to turn); here what it computes from what, the matrices too (red positive, blue negative, near white about 0, as in 3D; hover a cell for its value).", false, html! { <>
                        <div class="tex" onclick={pick} data-tex={shown.clone()}>{ shown.clone() }</div>
                        <div class="stages">{ for buttons }</div>
                        { match *stage { Some(k) => arrays(e, k - 1, s), None => html! {} } }
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
                </> }
            }
        },
    };

    html! {
        <>
        <header>
            { header("LoRA explorer", "Low-rank adaptation, computed by X_eTaL in your browser: a frozen 8 x 8 weight matrix W adapted by two small trainable ones, A (r x 8) and B (8 x r), the layer y = W x + s B (A x). Choose the rank; see each stage, the update and its rank, the merge, and what adapters cost on a large mixture-of-experts model. Training comes next.") }
            { rank }
        </header>
        <main><div class="wide">
            <Split idle={stage.is_none()} close={{ let st = stage.clone(); Callback::from(move |_| st.set(None)) }} {right} left={html! { <>
            { body }
            <section class="panel code">
                <h2>{"The program"}</h2>
                <p class="note">{"Everything the page runs, in your browser: lora-explorer.xtl with the rank and sizes you set, then what the page reads back (the pipeline drawn as math, every stage's value, the matrices)."}</p>
                { block(&program(s), NONE) }
            </section>
            </> }} />
        </div></main>
        { footer() }
        </>
    }
}
