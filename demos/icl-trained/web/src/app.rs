//! The page: a model and a prompt, what the model answers and which
//! heads answer it, a slot swapped, the gate found and pushed, the
//! training curves, the command line's measures and the program.

use gloo_timers::callback::Timeout;
use web_sys::HtmlSelectElement;
use yew::prelude::*;

use microscope::chrome::{footer, header, notice, panel};
use microscope::source::{between, block};

use crate::micro::{class_of, gate_tail, head, log, measured, prompt, tail, token, words, Anatomy, Edit, Set, CLASS, MODELS, N};
use crate::model::{Action, Job, Model};

const COLORS: [&str; 8] = ["#1c7ed6", "#e8590c", "#2f9e44", "#ae3ec9", "#f08c00", "#0c8599", "#c2255c", "#5c940d"];

fn act(m: &UseReducerHandle<Model>, a: impl Fn() -> Action + 'static) -> Callback<MouseEvent> {
    let d = m.dispatcher();
    Callback::from(move |_| d.dispatch(a()))
}

fn controls(m: &UseReducerHandle<Model>) -> Html {
    let d = m.dispatcher();
    let on_prompt = Callback::from(move |e: Event| {
        let v: usize = e.target_unchecked_into::<HtmlSelectElement>().value().parse().unwrap_or(0);
        let set = if v < N { Set::Unique } else { Set::Recurring };
        d.dispatch(Action::Prompt(set, v % N + 1))
    });
    let current = if m.set == Set::Unique { m.k - 1 } else { N + m.k - 1 };
    html! {
        <div class="controls">
            { for MODELS.iter().enumerate().map(|(i, (_, name, _))| html! {
                <button class={classes!((m.model == i).then_some("on"))} onclick={act(m, move || Action::Model(i))}>{name}</button>
            }) }
            <label>{" prompt "}
                <select onchange={on_prompt} aria-label="Prompt">
                    { for (0..2 * N).map(|v| html! { <option value={v.to_string()} selected={v == current}>{ if v < N { format!("unique labels {}", v + 1) } else { format!("recurring labels {}", v - N + 1) } }</option> }) }
                </select>
            </label>
            <span class="gen">{ if m.queue.is_empty() { format!("X_eTaL ran it in {:.0} ms", m.ms) } else if m.queue.contains(&Job::Gate) { "finding the gate: 24 runs\u{2026}".into() } else { "running\u{2026}".into() } }</span>
        </div>
    }
}

/// The prompt, a demonstration a group: each sentence's class, its
/// words, the arrow, the label.
fn shown_prompt(m: &Model) -> Html {
    let cls = words();
    let full = prompt(m.set, m.k);
    let mut groups: Vec<Html> = vec![];
    let mut i = 1;
    while i + 4 <= full.len() {
        let ws = &full[i..i + 3];
        let c = class_of(ws, &cls);
        let chip = c.map_or(html! {}, |c| html! { <span class="cls" style={format!("background: {}", COLORS[c])}>{CLASS[c]}</span> });
        let last = i + 5 >= full.len();
        let label = full[i + 4];
        groups.push(html! { <span class="demo">{chip}{ ws.iter().map(|&w| token(w)).collect::<Vec<_>>().join(" ") }{" \u{2192} "}
            <span class={classes!("lab", last.then_some("answer"))}>{ if last { format!("{}?", token(label)) } else { token(label) } }</span></span> });
        i += 6;
    }
    let body = html! { <>
        <p class="prompt">{ for groups }</p>
        <p class="note">{"Each word (w0 to w47) is in two of the eight classes; a sentence is three words sharing one class, its chip. Labels (L0 to L31) are drawn afresh for every prompt, so the model can only learn them from the prompt. The last line is the query; the model is to give its label (orange, with a question mark), not shown to it."}</p>
    </> };
    panel("The prompt:", "u:p_rompt (P, k)", MODELS[m.model].2, false, body)
}

fn answer(a: &Anatomy, intact: Option<&Anatomy>) -> Html {
    let mut shown: Vec<usize> = a.ids.iter().copied().filter(|&t| t >= 52).collect();
    shown.sort();
    shown.dedup();
    if !shown.contains(&a.answer) {
        shown.push(a.answer);
    }
    let top = shown.iter().fold(1e-9_f64, |b, &l| b.max(a.logits[l].abs()));
    let bars = shown.iter().map(|&l| {
        let v = a.logits[l];
        html! { <div class={classes!("bar", (l == a.answer).then_some("best"))} title={format!("{v:.3}")}>
            <b>{format!("{v:.1}")}</b>
            <div class="fill" style={format!("height: {:.1}%", 100.0 * v.max(0.0) / top)}></div>
            <span>{token(l)}</span>
        </div> }
    });
    let change = match (a.edit, intact) {
        (Edit::None, _) | (_, None) => String::new(),
        (_, Some(i)) => format!(" Intact, it was {:+.2}.", i.margin),
    };
    let body = html! { <>
        <div class="bars digits">{ for bars }</div>
        <p class="calc">{format!("The answer is {}: its logit less the best other label shown ({}), the margin, is {:+.2}.{change}", token(a.answer), token(a.alt), a.margin)}</p>
    </> };
    panel("The answer:", "u:m_easure", "The logits at the query's arrow of the labels shown in the prompt (and the answer). The paper's margin: the answer's logit less the best other label's, in nats.", false, body)
}

/// A 4 x 4 table of per-head numbers, layers as rows.
fn heads(v: &[f64]) -> Html {
    let top = v.iter().fold(1e-9_f64, |b, x| b.max(x.abs()));
    html! { <table class="zs heads">
        <tr><th></th>{ for (1..=4).map(|h| html! { <th>{format!("head {h}")}</th> }) }</tr>
        { for (0..4).map(|l| html! { <tr><th>{format!("layer {}", l + 1)}</th>
            { for (0..4).map(|h| { let x = v[l * 4 + h]; let a = (x.abs() / top).min(1.0);
                let bg = if x >= 0.0 { format!("rgba(232, 89, 12, {a:.2})") } else { format!("rgba(28, 126, 214, {a:.2})") };
                html! { <td style={format!("background: {bg}")}>{format!("{x:+.2}")}</td> } }) }
        </tr> }) }
    </table> }
}

fn who(a: &Anatomy) -> Html {
    let (l, h) = a.top();
    let look = a.look(l, h);
    let bars = (0..a.n()).map(|i| {
        let w = look[i];
        let t = a.ids[i];
        let slot = t >= 52 && i > 0 && a.ids[i - 1] == 2;
        let kind = if slot && t == a.answer { "match" } else if slot { "slot" } else { "rest" };
        html! { <div class={classes!("bar", kind)} title={format!("{}: {w:.3}", token(t))}>
            <b>{ if w >= 0.05 { format!("{w:.2}") } else { String::new() } }</b>
            <div class="fill" style={format!("height: {:.1}%", 100.0 * w)}></div>
            <span>{token(t)}</span>
        </div> }
    });
    let body = html! { <>
        {heads(&a.dla)}
        <p class="calc">{format!("Head {h} of layer {l} adds the most: where it looks from the arrow, the slots holding the answer in orange, other labels in purple.")}</p>
        <div class="bars icl">{ for bars }</div>
    </> };
    panel("Which heads answer:", "Wt u:d_la run", "Each head's direct contribution to the margin: its write at the arrow, through the last norm and the unembedding, the answer's column less the alternative's (the paper's direct logit attribution).", false, body)
}

fn experiments(m: &UseReducerHandle<Model>) -> Html {
    let g = m.gates[m.model].clone();
    let swaps = (0..4).map(|l| {
        let on = m.edit == Edit::Swap(l);
        html! { <button class={classes!(on.then_some("on"))} onclick={act(m, move || Action::Edit(Edit::Swap(l)))}>{ if l == 0 { "at the embedding".into() } else { format!("after layer {l}") } }</button> }
    });
    let gate_body = match &g {
        None => html! { <p><button onclick={act(m, || Action::Gate)} disabled={m.queue.contains(&Job::Gate)}>{ if m.queue.contains(&Job::Gate) { "finding\u{2026}" } else { "Find the gate" } }</button>
            {format!(" {N} unique-label and {N} recurring-label prompts and the recurring ones' class-flip twins: 24 runs (about half a minute here).")}</p> },
        Some(g) => {
            let gl = g.layer();
            let steer = [("toward unique", Edit::Steer(gl, -1.0)), ("toward recurring", Edit::Steer(gl, 1.0)), ("random, -1", Edit::Random(gl, -1.0)), ("random, +1", Edit::Random(gl, 1.0))];
            let gap = g.gaps[gl - 1].0.abs();
            html! { <>
                <p class="calc">{format!("Margins on these prompts: unique {:.2}, recurring {:.2}.", g.unique.iter().sum::<f64>() / N as f64, g.recurring.iter().sum::<f64>() / N as f64)}</p>
                <table class="zs"><tr><th>{"the direction after"}</th><th>{"gap"}</th><th>{"sign accuracy, held out"}</th></tr>
                    { for g.gaps.iter().enumerate().map(|(i, (gp, ac))| html! { <tr><td>{format!("layer {}", i + 1)}</td><td>{format!("{gp:.2}")}</td><td>{format!("{ac:.2}")}</td></tr> }) }
                </table>
                <p class="note">{"The class-flip attribution of each head (the readers: those whose contribution follows the query's class):"}</p>
                {heads(&g.flip)}
                <p>{format!("Push the arrow by one gap ({gap:.2}) along the direction after layer {gl}: ")}
                    { for steer.iter().map(|(name, e)| { let e = *e; html! { <button class={classes!((m.edit == e).then_some("on"))} onclick={act(m, move || Action::Edit(e))}>{*name}</button> } }) }</p>
            </> }
        }
    };
    let body = html! { <>
        <p>{"Swap the answer's slot with another label's slot, each keeping its position: "}{ for swaps }
            <button class={classes!((m.edit == Edit::None).then_some("on"))} onclick={act(m, || Action::Edit(Edit::None))}>{"intact"}</button></p>
        <p class="note">{"Swapped before the writer heads have keyed the slots, the swapped-in label is answered; after, the answer follows its key. The answer panel shows the result."}</p>
        <h3>{"The gate"}</h3>
        <p class="note">{"The direction separating recurring-label from unique-label prompts at the query's arrow: the recurring prompts' mean less the unique ones', from half of each, its gap and sign accuracy on the other half. The paper found it only in models trained where retrieving on unique-label prompts does not pay (the gate mix)."}</p>
        {gate_body}
    </> };
    panel("Take it apart:", "(l, a, b, lv, v) u:e_dit", "Edits of the residual stream after a layer, then the rest of the forward pass.", false, body)
}

/// A training curve: the held-out margins of each kind over the steps.
fn curves(model: usize) -> Html {
    let pts = log(model);
    let (w, h) = (560.0, 200.0);
    let last = pts.last().map_or(1.0, |p| p.0);
    let (lo, hi) = (-14.0, 12.0);
    let x = |s: f64| 40.0 + (w - 50.0) * s / last;
    let y = |v: f64| 10.0 + (h - 30.0) * (hi - v.clamp(lo, hi)) / (hi - lo);
    let line = |f: fn(&(f64, f64, f64, f64)) -> f64| pts.iter().map(|p| format!("{:.1},{:.1}", x(p.0), y(f(p)))).collect::<Vec<_>>().join(" ");
    let series: [(&str, &str, fn(&(f64, f64, f64, f64)) -> f64); 3] = [("unique", "#1c7ed6", |p| p.1), ("recurring", "#e8590c", |p| p.2), ("fresh query label", "#868e96", |p| p.3)];
    let body = html! { <>
        <svg class="curve" viewBox={format!("0 0 {w} {h}")} role="img" aria-label="margins over training">
            <line x1="40" x2={(w - 10.0).to_string()} y1={y(0.0).to_string()} y2={y(0.0).to_string()} stroke="#adb5bd" stroke-dasharray="3 3" />
            { for [-10.0, 0.0, 10.0].iter().map(|v| html! { <text x="34" y={(y(*v) + 4.0).to_string()} font-size="10" text-anchor="end">{v.to_string()}</text> }) }
            <text x={(w - 10.0).to_string()} y={(h - 4.0).to_string()} font-size="10" text-anchor="end">{format!("{} steps", last as usize + 1)}</text>
            { for series.iter().map(|(_, c, f)| html! { <polyline points={line(*f)} fill="none" stroke={*c} stroke-width="1.6" /> }) }
            { for series.iter().enumerate().map(|(i, (n, c, _))| html! { <text x={(50.0 + 130.0 * i as f64).to_string()} y="22" font-size="11" fill={*c}>{*n}</text> }) }
        </svg>
    </> };
    panel("How it learned:", "train/", "The mean margin on 96 held-out prompts of each kind during training, in nats: the task is learned all at once, after a long flat start, as the paper found.", false, body)
}

fn table() -> Html {
    let ms = measured();
    let names = ["Mix", "Recurring only", "Gate"];
    let paper = [("8.8", "8.8", "0.60", "8.9 (8.8)", "8.7 (8.7)"), ("10.3", "10.4", "0.54", "10.4 (10.4)", "10.2 (10.3)"), ("1.8 to 2.3", "8.7 to 9.1", "0.81 to 1.00", "0.7 (7.0), 1.1 (5.8)", "8.0 (1.8), 4.0 (0.9)")];
    html! { <table class="zs">
        <tr><th>{"training"}</th><th>{"unique"}</th><th>{"recurring"}</th><th>{"direction"}</th><th>{"recurring \u{2192} unique"}</th><th>{"unique \u{2192} recurring"}</th><th>{"readers"}</th><th>{"slot swap: none, embedding, layers 1 to 3"}</th></tr>
        { for ms.iter().zip(names).zip(paper).map(|((m, n), p)| html! { <>
            <tr><td style="text-align: left">{n}</td><td>{format!("{:.2}", m.margins.0)}</td><td>{format!("{:.2}", m.margins.1)}</td>
                <td>{format!("{:.2} (layer {})", m.gaps[m.layer - 1].1, m.layer)}</td>
                <td>{format!("{:.2} ({:.2})", m.steer[0], m.steer[1])}</td><td>{format!("{:.2} ({:.2})", m.steer[2], m.steer[3])}</td>
                <td>{ m.readers.iter().map(|(l, h, a)| format!("{l}:{h} {a:.2}")).collect::<Vec<_>>().join(", ") }</td>
                <td>{ m.swap.iter().map(|x| format!("{x:.1}")).collect::<Vec<_>>().join(" ") }</td></tr>
            <tr class="paper"><td style="text-align: left">{"  the paper"}</td><td>{p.0}</td><td>{p.1}</td><td>{p.2}</td><td>{p.3}</td><td>{p.4}</td><td>{"1 or 2 in the last layer"}</td><td>{"7 to 11 at the writer layers"}</td></tr>
        </> }) }
    </table> }
}

fn source(m: &Model) -> Html {
    let src = match &m.gates[m.model] {
        Some(_) if matches!(m.edit, Edit::None) => format!("{}# -- the gate, found for this model --------------------\n{}# -- the run, for your prompt -------------------------\n{}", head(), gate_tail(m.model), tail(m.model, m.set, m.k, m.edit)),
        _ => format!("{}# -- the run, for your prompt -------------------------\n{}", head(), tail(m.model, m.set, m.k, m.edit)),
    };
    block(&src, between(&src, "u:l_ayer := ", "u:l_ayer := "))
}

#[function_component(App)]
pub fn app() -> Html {
    let model = use_reducer(Model::new);
    {
        // The next job once this frame has drawn.
        let d = model.dispatcher();
        let key = (model.queue.len(), model.queue.first().cloned());
        use_effect_with(key, move |(n, _)| {
            let t = (*n > 0).then(|| Timeout::new(30, move || d.dispatch(Action::Next)));
            move || drop(t)
        });
    }
    let body = match &model.anatomy {
        Some(a) => html! {
            <div class="layout even">
                <div class="col">{ shown_prompt(&model) }{ answer(a, model.intact.as_deref()) }{ who(a) }</div>
                <div class="col">
                    { experiments(&model) }
                    { curves(model.model) }
                    <section class="panel">
                        <h2>{"The paper's measures, taken by the command line"}</h2>
                        <p class="note">{"icl-trained.xtl on each model, 8 prompts of each kind (the paper's authors used about 100 to 200): the margins, the gate direction's held-out sign accuracy at its best layer, the margins pushed one gap along it (a random direction of the same length in parentheses), the two heads with the largest class-flip attribution (layer:head), and the slot swap (the swapped-in label's logit less the answer's)."}</p>
                        { table() }
                    </section>
                    <section class="panel code">
                        <h2>{"The program"}</h2>
                        <p class="note">{"Everything the page runs, in your browser: icl-trained.xtl up to its own runs (the models read from their files, the forward pass, the prompts, the measures, the analysis), then the lines for your prompt; the forward pass's layer is highlighted. The weights were learned by train/, offline; X_eTaL runs them."}</p>
                        { source(&model) }
                    </section>
                </div>
            </div>
        },
        None => html! { <div class="running"><div class="spinner"></div>{"X_eTaL is running the model\u{2026}"}</div> },
    };
    html! {
        <>
        <header>
            { header("In-context learning, trained", "A small transformer (4 layers, 64 wide, 180,288 weights) trained from scratch on Gao and Gao's task: demonstrations, a sentence of some class and a label, then a query sentence whose label it must give, the labels redrawn for every prompt. Three training mixes, each taken apart by X_eTaL in your browser as the paper took its model apart: the heads that read the answer, the slots the labels are kept in, and the gate the paper found forms only when retrieving on unique-label prompts does not pay.") }
            { controls(&model) }
            { notice(&model.notice) }
        </header>
        <main>{body}</main>
        { footer() }
        </>
    }
}
