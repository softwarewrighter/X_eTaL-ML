//! A pipeline's stages on a page: the pipeline drawn as math, a row a
//! stage, a click choosing a row; what the chosen stage gives, its
//! numbers and in 3D the stages it reads; two panes with a divider
//! dragged to resize; and `Stages`, all of it for the first pipeline of
//! a program. Moved here from the pipelines demo.

use wasm_bindgen::JsCast;
use yew::prelude::*;

use crate::chrome::panel;
use crate::color::{ramp, DIVERGE};
use crate::source::{block, code, NONE};
use crate::stages::{chosen_row, clickable, dragged, explore, inputs, picture, stage_ranges, Value, SIDE, SIZE};

/// A click on a drawn pipeline's row: its stage (its class st-K) given
/// to `chose`, the row by where the click falls (chosen_row).
pub fn row_picker(chose: Callback<usize>) -> Callback<MouseEvent> {
    Callback::from(move |ev: MouseEvent| {
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
            chose.emit(k);
        }
    })
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

/// The hint the right pane shows until a stage is chosen.
pub fn no_stage() -> Html {
    panel("What a stage gives", "flow:s_tages<", "Click a row of the pipeline drawn as math, or a stage under the pipeline, to see here what it gives: its shape, its numbers, and in 3D the stages it reads and what it gives.", false, html! {})
}

/// Stage `k` of a pipeline (`text`, its call as written) with its
/// values `trace` (flow:s_tages<): what it gives, then in 3D what it
/// reads and gives.
pub fn stage_view(trace: &[Value], text: &str, k: usize, whose: &str, turn: &UseStateHandle<Turn>) -> Html {
    let vs: Vec<&Value> = trace.iter().filter(|v| v.stage == k).collect();
    let labels: Vec<(usize, String)> = trace.iter().filter(|v| !v.label.is_empty()).map(|v| (v.stage, v.label.clone())).collect();
    let reads = stage_ranges(text).get(k - 1).map(|&(a, b)| inputs(&text[a..b], k, &labels)).unwrap_or_default();
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

#[derive(Properties, PartialEq)]
pub struct SplitProps {
    pub left: Html,
    pub right: Html,
    /// No stage chosen: on a narrow screen the right pane is hidden.
    pub idle: bool,
    /// The right pane's close button (shown when a stage is chosen).
    pub close: Callback<()>,
}

/// Two panes: `left`, and `right` in view as the page scrolls; the
/// divider between them dragged to resize the right (`dragged`).
#[function_component(Split)]
pub fn split(p: &SplitProps) -> Html {
    let side = use_state(|| SIDE);
    let drag = use_state(|| None::<(f64, f64)>);
    let (s1, d1, s2, d2, d3) = (side.clone(), drag.clone(), side.clone(), drag.clone(), drag.clone());
    let down = Callback::from(move |e: PointerEvent| {
        if let Some(el) = e.target_dyn_into::<web_sys::Element>() {
            let _ = el.set_pointer_capture(e.pointer_id());
        }
        d1.set(Some((e.client_x() as f64, *s1)));
    });
    let mv = Callback::from(move |e: PointerEvent| {
        if let Some((x, w)) = *d2 {
            s2.set(dragged(w, x, e.client_x() as f64));
        }
    });
    let up = Callback::from(move |_: PointerEvent| d3.set(None));
    let close = p.close.reform(|_: MouseEvent| ());
    html! {
        <div class={classes!("work", p.idle.then_some("idle"))} style={format!("--side: {:.0}px", *side)}>
            <div class="left">{ p.left.clone() }</div>
            <div class={classes!("divider", drag.is_some().then_some("drag"))} title="drag to resize"
                onpointerdown={down} onpointermove={mv} onpointerup={up.clone()} onpointercancel={up}></div>
            <aside class="side">
                { if p.idle { html! {} } else { html! { <button class="close" title="close" aria-label="close the stage" onclick={close}>{"\u{00d7}"}</button> } } }
                { p.right.clone() }
            </aside>
        </div>
    }
}

#[derive(Properties, PartialEq)]
pub struct Props {
    /// The program that was run.
    pub src: String,
}

#[function_component(Stages)]
pub fn stages(p: &Props) -> Html {
    // Run once per program, not on every render.
    let found = use_memo(p.src.clone(), |src| explore(src));
    let chosen = use_state(|| None::<usize>);
    let turn = use_state(Turn::default);
    {
        // A new program, a new choice.
        let chosen = chosen.clone();
        use_effect_with(p.src.clone(), move |_| chosen.set(None));
    }
    let x = match found.as_ref() {
        None => return html! {},
        Some(Err(why)) => {
            return panel("The stages", "flow:s_tages<", "Its first pipeline's stages could not be shown: run at the top of the program, a pipeline's names must be bound there (not inside a function).", false, html! { <pre class="err">{ why.clone() }</pre> })
        }
        Some(Ok(x)) => x,
    };
    let on = *chosen;
    let c = chosen.clone();
    let pick = row_picker(Callback::from(move |k| c.set(Some(k))));
    let texts = stage_ranges(&x.call);
    let at = on.and_then(|k| texts.get(k - 1).copied()).unwrap_or(NONE);
    let chips = html! { <div class="stages">
        { for (1..=texts.len()).map(|k| {
            let c = chosen.clone();
            let label = x.trace.iter().filter(|v| v.stage == k).map(|v| v.label.clone()).filter(|l| !l.is_empty()).collect::<Vec<_>>().join(", ");
            let title = if label.is_empty() { format!("stage {k}") } else { format!("stage {k}: {label}") };
            html! { <button class={classes!((on == Some(k)).then_some("on"))} onclick={Callback::from(move |_| c.set(Some(k)))}>{title}</button> }
        }) }
    </div> };
    let shown = clickable(&x.tex, on);
    let left = panel("The stages", "flow:m_ath<", "The program's first pipeline, drawn as math by Flow's m_ath< (a row a stage) and as written: click a row or a stage to see on the right what it gives, its numbers and in 3D the stages it reads.", false, html! { <>
        <div class="tex" style="overflow-x: auto" onclick={pick} data-tex={shown.clone()}>{shown.clone()}</div>
        { block(&x.call, at) }
        { chips }
    </> });
    let right = match on {
        Some(k) => stage_view(&x.trace, &x.call, k, "of your pipeline", &turn),
        None => no_stage(),
    };
    let c = chosen.clone();
    html! { <Split idle={on.is_none()} close={Callback::from(move |_| c.set(None))} {left} {right} /> }
}
