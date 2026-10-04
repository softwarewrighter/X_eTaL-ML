//! The page: tokens routed to a 4 x 4 grid of experts, the routing
//! arrays, one token's arithmetic, the load, the program.

use std::rc::Rc;

use gloo_timers::callback::Interval;
use web_sys::{HtmlInputElement, HtmlSelectElement};
use yew::prelude::*;

use microscope::canvas::Canvas;
use microscope::chrome::{chip, footer, header, notice, panel};
use microscope::colour;
use microscope::source::code;

use crate::micro::{expert_name, tokens, words, Anatomy, Nudge, Nudged, EXPERTS, FEATURES, SENTENCES, SIDE, STEPS};
use crate::model::{Action, Model};
use crate::view::{expert_box, slice, slice_cell, source, strip, token_y, Stage, CELL_H, CELL_W, GAP, GRID_X, STAGES, TOK_X, TOP};

fn act(m: &UseReducerHandle<Model>, a: impl Fn() -> Action + 'static) -> Callback<MouseEvent> {
    let d = m.dispatcher();
    Callback::from(move |_| d.dispatch(a()))
}

fn num(x: f64) -> String {
    if x == 0.0 || !x.is_finite() {
        return "0.0".into();
    }
    let digits = (3 - x.abs().log10().floor() as i32).clamp(1, 8) as usize;
    format!("{x:.digits$}")
}

fn stage_chip(m: &UseReducerHandle<Model>, s: Stage) -> Html {
    let n = m.last.as_ref().map_or(0, |a| a.n());
    let (name, src, dims, meaning): (&str, &str, Vec<usize>, &str) = match s {
        Stage::Embed => ("embed", "ids s_elect E", vec![n, 8], "each token's 8 features"),
        Stage::Scores => ("scores", "u:s_cores x", vec![n, EXPERTS], "every token against every expert"),
        Stage::Softmax => ("softmax", "u:s_oftmax", vec![n, EXPERTS], "each token's probabilities over the experts"),
        Stage::Top2 => ("top-2", "u:t_op2 p", vec![n, EXPERTS], "the gates: two per token, 0 elsewhere"),
        Stage::Load => ("load", "u:l_oad gates", vec![EXPERTS], "how many tokens each expert got"),
        Stage::Nudge => ("nudge", "gs ; gg", vec![STEPS, EXPERTS], "the gates of one token pushed along a direction, step by step"),
    };
    chip(name, src, &dims, meaning, m.focus == s, act(m, move || Action::Focus(s)))
}

fn controls(m: &UseReducerHandle<Model>) -> Html {
    let d = m.dispatcher();
    let on_preset = Callback::from(move |e: Event| {
        let i = e.target_unchecked_into::<HtmlSelectElement>().selected_index();
        d.dispatch(Action::Sentence(SENTENCES[i.max(0) as usize % SENTENCES.len()].to_string()));
    });
    let d = m.dispatcher();
    let on_text = Callback::from(move |e: Event| {
        d.dispatch(Action::Sentence(e.target_unchecked_into::<HtmlInputElement>().value()));
    });
    html! {
        <div class="controls">
            <button onclick={act(m, || Action::TogglePlay)}>{ if m.playing { "Pause" } else { "Play" } }</button>
            <select onchange={on_preset} aria-label="Sentence">
                { for SENTENCES.iter().map(|s| html! { <option selected={*s == m.sentence}>{*s}</option> }) }
            </select>
            <input class="sentence" type="text" value={m.sentence.clone()} onchange={on_text} aria-label="Your sentence" />
            <span class="gen">{format!("X_eTaL routed {} tokens in {:.0} ms", m.last.as_ref().map_or(0, |a| a.n()), m.ms)}</span>
        </div>
    }
}

/// The routing picture: tokens on the left, the 16 experts on the
/// right, a curve from each token to its two experts as wide as the
/// gate; the selected token's curves in colour.
fn picture(model: &UseReducerHandle<Model>, a: &Anatomy) -> Html {
    let n = a.n();
    let sel = model.token;
    let curves = (0..n).flat_map(|t| a.top2(t).into_iter().map(move |e| (t, e))).map(|(t, e)| {
        let g = a.gates[t * EXPERTS + e];
        let (x1, y1) = (TOK_X + 6.0, token_y(t, n));
        let (bx, by) = expert_box(e);
        let (x2, y2) = (bx, by + CELL_H / 2.0);
        let d = format!("M {x1:.1} {y1:.1} C {:.1} {y1:.1}, {:.1} {y2:.1}, {x2:.1} {y2:.1}", x1 + 110.0, x2 - 110.0);
        html! { <path class={classes!("route", (t == sel).then_some("sel"))} {d} stroke-width={format!("{:.2}", 1.0 + 7.0 * g)} /> }
    });
    let labels = (0..n).map(|t| {
        let d = model.dispatcher();
        let onclick = Callback::from(move |_: MouseEvent| d.dispatch(Action::Token(t)));
        html! { <text class={classes!("tok", (t == sel).then_some("sel"))} x={TOK_X.to_string()} y={format!("{:.1}", token_y(t, n) + 5.0)} text-anchor="end" {onclick}>{a.words[t].clone()}</text> }
    });
    let picked = a.top2(sel);
    let max_load = a.load.iter().cloned().fold(1.0, f64::max);
    let boxes = (0..EXPERTS).map(|e| {
        let (x, y) = expert_box(e);
        let heat = a.load[e] / max_load;
        let style = format!("fill: hsl(258 70% {}%)", 97.0 - 30.0 * heat);
        html! { <g class={classes!("expert", picked.contains(&e).then_some("sel"))}>
            <rect x={x.to_string()} y={y.to_string()} width={CELL_W.to_string()} height={CELL_H.to_string()} rx="8" {style} />
            <text x={(x + 8.0).to_string()} y={(y + 18.0).to_string()} class="ename">{format!("{} {}", e + 1, FEATURES[e / 4])}</text>
            <text x={(x + 8.0).to_string()} y={(y + 34.0).to_string()} class="ename">{format!("\u{00d7} {}", FEATURES[4 + e % 4])}</text>
            <text x={(x + 8.0).to_string()} y={(y + 54.0).to_string()} class="eload">{format!("{} token{}", a.load[e], if a.load[e] == 1.0 { "" } else { "s" })}</text>
        </g> }
    });
    let w = GRID_X + 4.0 * (CELL_W + GAP);
    let h = TOP * 2.0 + 4.0 * (CELL_H + GAP);
    let body = html! {
        <svg class="routes" viewBox={format!("0 0 {w} {h}")} role="img" aria-label="tokens routed to experts">
            { for curves }{ for boxes }{ for labels }
        </svg>
    };
    panel("Routing:", "gates := u:t_op2 u:s_oftmax u:s_cores x",
        "Each token goes to its two most likely experts; a curve is as wide as its gate. Experts are shaded by load. Click a token to inspect it.", false, body)
}

fn word_select(label: &str, chosen: usize, on: impl Fn(usize) -> Nudge + 'static, model: &UseReducerHandle<Model>) -> Html {
    let d = model.dispatcher();
    let onchange = Callback::from(move |e: Event| {
        let i = e.target_unchecked_into::<HtmlSelectElement>().selected_index();
        d.dispatch(Action::Nudge(on(i.max(0) as usize + 1)));
    });
    html! { <select {onchange} aria-label={label.to_string()}>
        { for words().iter().enumerate().map(|(i, w)| html! { <option selected={i + 1 == chosen}>{*w}</option> }) }
    </select> }
}

/// Nudge a token: one word's embedding pushed towards another, the
/// experts chosen along the way, and a slice of the space around it.
fn nudge(model: &UseReducerHandle<Model>, n: &Nudged) -> Html {
    let w = words();
    let nd = model.nudge;
    let i = model.cursor.min(STEPS - 1);
    let d = model.dispatcher();
    let on_eps = Callback::from(move |e: InputEvent| {
        let v = e.target_unchecked_into::<HtmlInputElement>().value();
        d.dispatch(Action::Cursor(v.parse().unwrap_or(0)));
    });
    let d = model.dispatcher();
    let on_strip = Some(Callback::from(move |(_, x): (usize, usize)| d.dispatch(Action::Cursor(x))));
    let (f, s) = (n.first[i] as usize, n.second[i] as usize);
    let (g1, g2) = (n.gates[i * EXPERTS + f - 1], n.gates[i * EXPERTS + s - 1]);
    let changes = n.changes();
    let list = changes.iter().map(|&c| html! { <li>{format!(
        "between \u{03b5} = {:.2} and {:.2}: experts {} and {} \u{2192} {} and {}",
        n.eps[c], n.eps[c + 1], n.first[c], n.second[c], n.first[c + 1], n.second[c + 1]
    )}</li> });
    let (cr, cc) = slice_cell(n.eps[i], 0.0);
    let controls = html! { <div class="controls">
        {"push "}{ word_select("Word", nd.w0, move |k| Nudge { w0: k, ..nd }, model) }
        {" towards "}{ word_select("Towards", nd.wa, move |k| Nudge { wa: k, ..nd }, model) }
        {" (the slice also towards "}{ word_select("Also towards", nd.wb, move |k| Nudge { wb: k, ..nd }, model) }{")"}
        <label class="slider">{"\u{03b5}"}
            <input type="range" min="0" max={(STEPS - 1).to_string()} value={i.to_string()} oninput={on_eps} />
            <b>{format!("{:.2}", n.eps[i])}</b>
        </label>
        <span class="gen">{format!("X_eTaL routed {} points in {:.0} ms", STEPS + SIDE * SIDE, model.nudge_ms)}</span>
    </div> };
    let body = html! { <>
        { controls }
        <p class="calc">{format!("At \u{03b5} = {:.2}, \u{201c}{}\u{201d} + \u{03b5} (\u{201c}{}\u{201d} \u{2212} \u{201c}{}\u{201d}) goes to expert {} ({}), gate {:.3}, and expert {} ({}), gate {:.3}.",
            n.eps[i], w[nd.w0 - 1], w[nd.wa - 1], w[nd.w0 - 1], f, expert_name(f - 1), g1, s, expert_name(s - 1), g2)}</p>
        <div class="pair">
            <figure class="wide">
                <Canvas rows={16} cols={STEPS} rgba={Rc::new(strip(&n.gates, i))} mark={None} onclick={on_strip} class="strip" />
                <figcaption>{code("gs")}{" : experts 1 to 16 down, \u{03b5} from 0 to 1.2 across; bright where an expert has a gate (the cursor column in orange)"}</figcaption>
            </figure>
            <figure>
                <Canvas rows={SIDE} cols={SIDE} rgba={Rc::new(slice(&n.slice_first, &n.slice_second))} mark={Some((cr, cc))} class="mid" />
                <figcaption>{code("gg")}{format!(" : x0 + a d1 + b d2, a across, b up; one colour per pair; dark dots: \u{201c}{}\u{201d}, \u{201c}{}\u{201d}, \u{201c}{}\u{201d}", w[nd.w0 - 1], w[nd.wa - 1], w[nd.wb - 1])}</figcaption>
            </figure>
        </div>
        <p class="note">{ if changes.is_empty() { "No change of experts along this path.".to_string() } else { format!("The pair of experts changes {} time{} along the path:", changes.len(), if changes.len() == 1 { "" } else { "s" }) } }</p>
        <ul class="changes">{ for list }</ul>
        <p class="note">{"The research point: inputs that are almost the same can go to different experts. The scores are linear in the input, so the regions where a pair of experts wins are cut by straight lines (where two experts' scores tie): a small step across a line switches experts abruptly, while a large step inside a region changes nothing. "}
            <a href="https://github.com/sw-ml-study/moe-microscope" target="_blank">{"moe-microscope"}</a>{" (live: "}<a href="https://sw-ml-study.github.io/moe-microscope/" target="_blank">{"lessons"}</a>{") builds and inspects whole mixture-of-experts models at this scale."}</p>
    </> };
    panel("Nudge a token:", "xs := (eps '* t_able d1) + (o_ffsets k) 'r_ight t_able x0",
        "One word's embedding pushed towards another's: the same router decides each step. Move \u{03b5}, or pick the words.", model.focus == Stage::Nudge, body)
}

fn pic(model: &UseReducerHandle<Model>, rows: usize, cols: usize, rgba: Vec<u8>, caption: &str) -> Html {
    let d = model.dispatcher();
    let onclick = Some(Callback::from(move |(y, _): (usize, usize)| d.dispatch(Action::Token(y))));
    html! { <figure>
        <Canvas {rows} {cols} rgba={Rc::new(rgba)} mark={None} {onclick} class="mid" />
        <figcaption>{code(caption)}</figcaption>
    </figure> }
}

fn arrays(model: &UseReducerHandle<Model>, a: &Anatomy) -> Html {
    let n = a.n();
    let f = model.focus;
    html! { <>
        { panel("1. Embed:", "x := ids s_elect E", "Each word's row of the embedding table: its 8 features (animal, number, colour, action, place, food, function word, time) plus a small ripple. One row per token.",
            f == Stage::Embed, html! { <div class="pair">{pic(model, n, 8, colour::signed(&a.x), "x")}</div> }) }
        { panel("2. Scores:", "u:s_cores x", "One matrix product: every token against every expert's weights (8 x 16).",
            f == Stage::Scores, html! { <div class="pair">{pic(model, n, EXPERTS, colour::signed(&a.scores), "u:s_cores x")}</div> }) }
        { panel("3. Softmax:", "p := u:s_oftmax u:s_cores x", "Each row's scores as probabilities: e to each score, over the row's sum (the row's largest taken off first so nothing overflows).",
            f == Stage::Softmax, html! { <div class="pair">{pic(model, n, EXPERTS, colour::field(&a.p, 0.0, 1.0), "p")}</div> }) }
        { panel("4. Top-2:", "gates := u:t_op2 p", "The largest of each row marks the first expert; with it taken out, the largest left marks the second. The two probabilities over their sum are the gates; the other 14 are 0.",
            f == Stage::Top2, html! { <div class="pair">{pic(model, n, EXPERTS, colour::field(&a.gates, 0.0, 1.0), "gates")}</div> }) }
        { panel("5. Load:", "u:l_oad gates", "Down each column, how many tokens have a gate there: each expert's share of the work.",
            f == Stage::Load, load(a)) }
    </> }
}

fn load(a: &Anatomy) -> Html {
    let max = a.load.iter().cloned().fold(1.0, f64::max);
    let bars = (0..EXPERTS).map(|e| html! {
        <div class="bar" title={expert_name(e)}>
            <div class="fill" style={format!("height: {:.0}%", 100.0 * a.load[e] / max)}></div>
            <span>{e + 1}</span>
        </div>
    });
    html! { <div class="bars">{ for bars }</div> }
}

fn grid4(v: &[f64], marked: [usize; 2]) -> Html {
    html! { <table class="egrid">{ for (0..4).map(|r| html! { <tr>{ for (0..4).map(|c| {
        let e = 4 * r + c;
        html! { <td class={classes!(marked.contains(&e).then_some("on"))}>{num(v[e])}</td> }
    }) }</tr> }) }</table> }
}

fn inspector(model: &UseReducerHandle<Model>, a: &Anatomy) -> Html {
    let t = model.token.min(a.n() - 1);
    let r = t * EXPERTS..(t + 1) * EXPERTS;
    let [e1, e2] = a.top2(t);
    let p = &a.p[r.clone()];
    let feats = (0..8).map(|k| html! { <tr><td>{FEATURES[k]}</td><td>{code(&num(a.x[t * 8 + k]))}</td></tr> });
    let g = format!("{} / ({} + {})", num(p[e1]), num(p[e1]), num(p[e2]));
    html! {
        <section class="panel inspector">
            <h2>{format!("Token {}: \u{201c}{}\u{201d}", t + 1, a.words[t])}</h2>
            <div class="crosses">
                <div><p class="note">{"its embedding"}</p><table class="zs">{ for feats }</table></div>
                <div><p class="note">{"its scores, on the expert grid"}</p>{grid4(&a.scores[r.clone()], [e1, e2])}</div>
                <div><p class="note">{"its probabilities"}</p>{grid4(p, [e1, e2])}</div>
            </div>
            <p class="calc">{format!("first: expert {} ({}), second: expert {} ({})", e1 + 1, expert_name(e1), e2 + 1, expert_name(e2))}</p>
            <p class="calc">{"gate 1 = "}{code(&g)}{" = "}{code(&num(a.gates[t * EXPERTS + e1]))}{"; gate 2 = "}{code(&num(a.gates[t * EXPERTS + e2]))}</p>
        </section>
    }
}

#[function_component(App)]
pub fn app() -> Html {
    let model = use_reducer(Model::new);
    {
        let d = model.dispatcher();
        use_effect_with(model.playing, move |&playing| {
            let timer = playing.then(|| Interval::new(1200, move || d.dispatch(Action::Tick)));
            move || drop(timer)
        });
    }
    let ids: Vec<usize> = tokens(&model.sentence).into_iter().map(|t| t.1).collect();
    let body = match &model.last {
        Some(a) => html! { <>
            { picture(&model, a) }
            { for model.nudged.as_ref().map(|n| nudge(&model, n)) }
            <div class="layout even">
                <div class="col">{ inspector(&model, a) }{ arrays(&model, a) }</div>
                <div class="col">
                    <section class="panel code">
                        <h2>{"The program"}</h2>
                        <p class="note">{"Everything the page runs, in your browser: the head of moe-router.xtl (the features read from data/, the router, the core), then the run for your sentence and the run for the nudge; the stage you pick is highlighted."}</p>
                        { source(&ids, &model.nudge, model.focus) }
                    </section>
                </div>
            </div>
        </> },
        None => html! {},
    };
    html! {
        <>
        <header>
            { header("MoE routing microscope", "A mixture-of-experts layer sends each token to only two of its 16 experts. Here every token is scored against every expert by one matrix product, the scores become probabilities, each token keeps its top two, and the experts' load is counted. Type a sentence; words from the vocabulary route by meaning.") }
            <nav class="timeline">{ for STAGES.iter().map(|&s| stage_chip(&model, s)) }</nav>
            { controls(&model) }
            { notice(&model.notice) }
        </header>
        <main>{body}</main>
        { footer() }
        </>
    }
}
