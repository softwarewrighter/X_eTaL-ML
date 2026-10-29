//! The page: an example to pick; its pipeline, what the pipeline became
//! and the same layer in plain X_eTaL side by side (for the CNN layer,
//! the equation as a paper prints it first); the checks that they
//! agree; and the whole program the page ran.

use wasm_bindgen::JsCast;
use yew::prelude::*;

use microscope::chrome::{footer, header, panel};
use microscope::color::{ramp, DIVERGE};
use microscope::palette::Composer;
use microscope::source::{between, block, code, NONE};

use crate::blocks::{inputs, picture, SIZE};
use crate::micro::{calls, chosen_row, clickable, expansion_ranges, program, run, stage_ranges, Example, Run, Value, EXAMPLES, STARTERS};

/// The stage chosen: of the pipeline of equations (drawn as math) or of
/// the pipeline, its number from 1 (the data first).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Chosen {
    Drawn(usize),
    Piped(usize),
}

fn picker(picked: &UseStateHandle<usize>, chosen: &UseStateHandle<Option<Chosen>>) -> Html {
    html! {
        <div class="controls">
            <span>{"An example: "}</span>
            { for EXAMPLES.iter().map(|e| e.title).chain(["Compose your own"]).enumerate().map(|(i, title)| {
                let (p, c) = (picked.clone(), chosen.clone());
                html! { <button class={classes!((**picked == i).then_some("on"))} onclick={Callback::from(move |_| { p.set(i); c.set(None); })}>{title}</button> }
            }) }
        </div>
    }
}

/// The equations, each drawn as math where KaTeX has loaded
/// (index.html), its LaTeX as text where it has not, above the call
/// that reads it.
fn math(e: &Example, r: &Run, chosen: &UseStateHandle<Option<Chosen>>) -> Html {
    let calls = calls(e).0;
    let on = match **chosen {
        Some(Chosen::Drawn(k)) => Some(k),
        _ => None,
    };
    // A click on a drawn row chooses its stage (its class st-K), the row
    // by where the click falls (chosen_row).
    let c = chosen.clone();
    let pick = Callback::from(move |ev: MouseEvent| {
        let Some(tex) = ev.current_target().and_then(|t| t.dyn_into::<web_sys::Element>().ok()) else { return };
        let Ok(found) = tex.query_selector_all(".st") else { return };
        let rows: Vec<(usize, f64, f64)> = (0..found.length())
            .filter_map(|i| found.item(i)?.dyn_into::<web_sys::Element>().ok())
            .filter_map(|el| {
                let k = el.class_name().split_whitespace().find_map(|w| w.strip_prefix("st-").and_then(|n| n.parse::<usize>().ok()))?;
                let r = el.get_bounding_client_rect();
                let x = ev.client_x() as f64;
                (r.left() <= x && x <= r.right()).then(|| (k, r.top(), r.bottom()))
            })
            .collect();
        if let Some(k) = chosen_row(&rows, ev.client_y() as f64) {
            c.set(Some(Chosen::Drawn(k)));
        }
    });
    if calls.is_empty() {
        return html! {};
    }
    let paper = calls.iter().any(|c| !c.contains("flow:m_ath<"));
    let note = format!("{}The layer as a pipeline of equations, drawn by Flow's m_ath< from the very text that computes it: a row a stage, an arrow between, each label after the stage it names (a triangle, as X_eTaL draws $, with its shape), each equation as X_eTaL-libraries' Equations draws it; checked equal to the pipeline below as the page runs.",
        if paper { "The equation as a paper prints it first, read by Equations (e:e_q< its value, e:t_ex< its LaTeX, one reading of the same text; a sum whose term uses the one before is one step, applied by p_ower). " } else { "" });
    panel("As math", "flow:m_ath<", &note,
        false, html! { <>
            { for calls.iter().zip(&r.tex).map(|(call, tex)| {
                let shown = if call.contains("flow:m_ath<") { clickable(tex, on) } else { tex.clone() };
                html! { <>
                    <div class="tex" style="overflow-x: auto" onclick={pick.clone()} data-tex={shown.clone()}>{shown.clone()}</div>
                    { block(call, NONE) }
                </> }
            }) }
            <p class="note">{"Click a row of the pipeline drawn as math, or a stage under the pipeline below, to see what it gives."}</p>
        </> })
}

/// The example's sentence, and the demo it links to.
fn about(e: &Example) -> Html {
    if e.about.is_empty() {
        return html! {};
    }
    html! { <p class="note">{e.about}{ for e.link.iter().map(|(title, href)| html! { <>{" The whole model: "}<a href={*href} style="color: var(--accent)">{*title}</a>{"."}</> }) }</p> }
}

fn columns(e: &Example, r: &Run, chosen: &UseStateHandle<Option<Chosen>>) -> Html {
    let (_, pipe, plain) = calls(e);
    let on = match **chosen {
        Some(Chosen::Piped(k)) => Some(k),
        _ => None,
    };
    let texts = stage_ranges(&pipe);
    let lines = expansion_ranges(&r.expansion);
    let at = |ranges: &[(usize, usize)]| on.and_then(|k| ranges.get(k - 1).copied()).unwrap_or(NONE);
    let chips = html! { <div class="stages">
        { for (1..=texts.len()).map(|k| {
            let c = chosen.clone();
            let label = r.stages.iter().filter(|v| v.stage == k).map(|v| v.label.clone()).filter(|l| !l.is_empty()).collect::<Vec<_>>().join(", ");
            let title = if label.is_empty() { format!("stage {k}") } else { format!("stage {k}: {label}") };
            html! { <button class={classes!((on == Some(k)).then_some("on"))} onclick={Callback::from(move |_| c.set(Some(Chosen::Piped(k))))}>{title}</button> }
        }) }
    </div> };
    html! {
        <div class="layout three">
            <div class="col">{ panel("The pipeline", "flow:p_ipe<",
                "Data flow, left to right: the data, then each stage, |> between them; it names the previous value, $ labels a stage's value for later stages.",
                true, html! { <>{ block(&pipe, at(&texts)) }{ chips }</> }) }</div>
            <div class="col">{ panel("What it became", "xetal expand",
                "What the Flow macro wrote in its place when the program was compiled: one line per stage, each value named, type-checked like any code.",
                false, block(&r.expansion, at(&lines))) }</div>
            <div class="col">{ panel("Plain X_eTaL", "the library",
                "The same layer as this repository's libraries write it, without a pipeline.",
                false, block(&plain, NONE)) }</div>
        </div>
    }
}

/// How the blocks are turned (yaw, pitch), and a drag in progress
/// (where it started, the turn then).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Turn {
    pub yaw: f64,
    pub pitch: f64,
    pub from: Option<(f64, f64, f64, f64)>,
}

impl Default for Turn {
    fn default() -> Self {
        Turn { yaw: -0.55, pitch: 0.5, from: None }
    }
}

/// What the chosen stage gives: each value (a tuple's parts each), its
/// shape, its numbers as a table colored by sign and size when it fits
/// (64 items at most, slices of a rank-3 array), else its range and its
/// first numbers; then in 3D, the stages it reads and what it gives.
fn gives(e: &Example, r: &Run, chosen: &UseStateHandle<Option<Chosen>>, turn: &UseStateHandle<Turn>) -> Html {
    let (math, pipe, _) = calls(e);
    let (trace, k, whose, text) = match **chosen {
        Some(Chosen::Drawn(k)) => (&r.drawn, k, "of the pipeline of equations", math.into_iter().find(|c| c.contains("flow:m_ath<")).unwrap_or_default()),
        Some(Chosen::Piped(k)) => (&r.stages, k, "of the pipeline", pipe),
        None => return html! {},
    };
    let vs: Vec<&Value> = trace.iter().filter(|v| v.stage == k).collect();
    let labels: Vec<(usize, String)> = trace.iter().filter(|v| !v.label.is_empty()).map(|v| (v.stage, v.label.clone())).collect();
    let reads = stage_ranges(&text).get(k - 1).map(|&(a, b)| inputs(&text[a..b], k, &labels)).unwrap_or_default();
    let title = format!("Stage {k} {whose} gives");
    panel(&title, "flow:s_tages<", "Computed as the page runs, the same values the pipeline passes on (Flow's s_tages< keeps every stage's value).", false, html! { <>
        { for vs.iter().map(|v| value(v)) }
        { solids(trace, &reads, &vs, turn) }
    </> })
}

/// The stage in 3D: the values of the stages it reads, an arrow, what
/// it gives; a cube an item, colored as the tables are; drag to turn.
fn solids(trace: &[Value], reads: &[usize], out: &[&Value], turn: &UseStateHandle<Turn>) -> Html {
    let t = **turn;
    let (t1, t2, t3) = (turn.clone(), turn.clone(), turn.clone());
    let down = Callback::from(move |e: PointerEvent| {
        let t = *t1;
        t1.set(Turn { from: Some((e.client_x() as f64, e.client_y() as f64, t.yaw, t.pitch)), ..t });
    });
    let mv = Callback::from(move |e: PointerEvent| {
        let t = *t2;
        if let Some((x, y, yaw, pitch)) = t.from {
            let (dx, dy) = (e.client_x() as f64 - x, e.client_y() as f64 - y);
            t2.set(Turn { yaw: yaw + 0.01 * dx, pitch: (pitch + 0.01 * dy).clamp(-1.5, 1.5), from: t.from });
        }
    });
    let up = Callback::from(move |_: PointerEvent| t3.set(Turn { from: None, ..*t3 }));
    let figure = |v: &Value, role: &str| {
        let p = picture(&v.shape, v.items.len(), t.yaw, t.pitch);
        let size = v.items.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-300);
        let name = if v.label.is_empty() { format!("stage {}", v.stage) } else { format!("stage {} ({})", v.stage, v.label) };
        let shape = if v.shape.is_empty() { "one number".to_string() } else { v.shape.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(" \u{00d7} ") };
        let shown = if v.items.len() < v.count() { format!(", its first {} items", v.items.len()) } else { String::new() };
        html! { <figure class="solid">
            <svg viewBox={format!("0 0 {SIZE} {SIZE}")} width={SIZE.to_string()} height={SIZE.to_string()} role="img" aria-label={format!("{role} {name} in 3-D")}>
                { for p.edges.iter().map(|&((x1, y1), (x2, y2))| html! { <line x1={format!("{x1:.1}")} y1={format!("{y1:.1}")} x2={format!("{x2:.1}")} y2={format!("{y2:.1}")} class="edge" /> }) }
                { for p.faces.iter().map(|f| {
                    let c = ramp(&DIVERGE, 0.5 + 0.5 * v.items[f.item] / size);
                    let pts = f.pts.iter().map(|(x, y)| format!("{x:.1},{y:.1}")).collect::<Vec<_>>().join(" ");
                    let fill = format!("rgb({:.0}, {:.0}, {:.0})", c[0] as f64 * f.light, c[1] as f64 * f.light, c[2] as f64 * f.light);
                    html! { <polygon points={pts} fill={fill} /> }
                }) }
                { for p.labels.iter().map(|(x, y, s)| html! { <text x={format!("{x:.1}")} y={format!("{y:.1}")} text-anchor="middle" dominant-baseline="middle">{s.clone()}</text> }) }
            </svg>
            <figcaption>{format!("{role} {name}: {shape}{shown}")}</figcaption>
        </figure> }
    };
    let ins: Vec<&Value> = reads.iter().flat_map(|&s| trace.iter().filter(move |v| v.stage == s)).collect();
    html! { <div class="solids" onpointerdown={down} onpointermove={mv} onpointerup={up.clone()} onpointerleave={up}>
        <p class="note">{"In 3D, a cube an item: the stages it reads, then what it gives. Drag to turn."}</p>
        <div class="row">
            { for ins.iter().map(|v| figure(v, "reads")) }
            { if ins.is_empty() { html! {} } else { html! { <span class="arrow">{"\u{2192}"}</span> } } }
            { for out.iter().map(|v| figure(v, "gives")) }
        </div>
    </div> }
}

fn value(v: &Value) -> Html {
    let name = if v.label.is_empty() { String::new() } else { format!("{}: ", v.label) };
    let shape = if v.shape.is_empty() { "one number".to_string() } else { v.shape.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(" \u{00d7} ") };
    let size = v.items.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-300);
    let cell = |x: f64| {
        let c = ramp(&DIVERGE, 0.5 + 0.5 * x / size);
        html! { <td style={format!("background: rgb({}, {}, {})", c[0], c[1], c[2])}>{format!("{x:.3}")}</td> }
    };
    let table = |rows: usize, cols: usize, from: usize| html! { <table class="items">
        { for (0..rows).map(|i| html! { <tr>{ for (0..cols).map(|j| cell(v.items[from + i * cols + j])) }</tr> }) }
    </table> };
    let fits = v.count() <= 64 && v.items.len() == v.count();
    let body = match (fits, v.shape.as_slice()) {
        (true, []) => table(1, 1, 0),
        (true, [n]) => table(1, *n, 0),
        (true, [r, c]) => table(*r, *c, 0),
        (true, [s, r, c]) => html! { <>{ for (0..*s).map(|i| table(*r, *c, i * r * c)) }</> },
        _ => html! { <p class="calc">{format!("from {:.4} to {:.4}; the first numbers: {}", v.min, v.max, v.items.iter().take(8).map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(" "))}</p> },
    };
    html! { <div class="value"><p class="calc">{ code(&format!("{name}{shape}")) }</p>{ body }</div> }
}

fn agree(e: &Example, r: &Run) -> Html {
    let shown: Vec<String> = r.value.iter().take(12).map(|v| format!("{v:.4}")).collect();
    let more = if r.value.len() > 12 { " ..." } else { "" };
    panel("They agree", "k:n_ear", "Checked as the program runs, in your browser: the values equal within a tolerance (Check's n_ear).", false, html! { <>
        { for r.checks.iter().map(|c| html! { <p class="calc">{c.clone()}</p> }) }
        <p class="calc">{code(&format!("r_avel {}", e.pipe))}{format!(" = {}{more}", shown.join(" "))}</p>
    </> })
}

#[function_component(App)]
pub fn app() -> Html {
    let picked = use_state(|| 0usize);
    let chosen = use_state(|| None::<Chosen>);
    let turn = use_state(Turn::default);
    // The last choice is composing one's own, from the palette.
    let main = if *picked >= EXAMPLES.len() {
        html! { <Composer starters={STARTERS.to_vec()} /> }
    } else {
        let e = EXAMPLES[*picked];
        let src = program(&e);
        let body = match run(&e) {
            Ok(r) => {
                // What a stage gives, beside where it was chosen.
                let drawn = matches!(*chosen, Some(Chosen::Drawn(_)));
                html! { <>{ about(&e) }{ math(&e, &r, &chosen) }{ if drawn { gives(&e, &r, &chosen, &turn) } else { html! {} } }{ columns(&e, &r, &chosen) }{ if drawn { html! {} } else { gives(&e, &r, &chosen, &turn) } }{ agree(&e, &r) }</> }
            }
            Err(why) => html! { <p class="error">{format!("X_eTaL stopped: {why}")}</p> },
        };
        let from = format!("\n{} := ", e.pipe);
        html! { <>
            { body }
            <section class="panel code">
                <h2>{"The program"}</h2>
                <p class="note">{"Everything the page runs, in your browser: the head of the program (its imports and data), this example's section, then the pipeline's value and the checks. The pipeline is highlighted."}</p>
                { block(&src, between(&src, &from, &from)) }
            </section>
        </> }
    };
    html! {
        <>
        <header>
            { header("Pipelines", "A layer written as data flow: Flow's pipelines, left to right, |> between the stages, on this repository's libraries, each beside what it became and the same layer in plain X_eTaL. A CNN layer and a diffusion sampler are shown four ways: as a paper prints them, as a pipeline, as its expansion and as plain X_eTaL, all computing the same values. Every example is also written as a pipeline of equations, drawn as math a row a stage. Compose your own from a palette of the notations' parts.") }
            { picker(&picked, &chosen) }
        </header>
        <main>
            { main }
        </main>
        { footer() }
        </>
    }
}
