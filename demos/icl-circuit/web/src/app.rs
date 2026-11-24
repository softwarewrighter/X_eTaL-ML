//! The page: a prompt of demonstrations, the circuit's parts to keep or
//! knock out, where the readers look, the answer, the residual stream's
//! named blocks, the paper's knockouts and the program.

use gloo_timers::callback::Timeout;
use web_sys::{HtmlInputElement, HtmlSelectElement};
use yew::prelude::*;

use microscope::chrome::{footer, header, notice, panel};
use microscope::source::{between, block};

use crate::micro::{presets, shown, strongest, Anatomy, Config, BLOCKS, CLASSES, KNOCKOUTS, LABELS};
use crate::model::{Action, Job, Model};

fn act(m: &UseReducerHandle<Model>, a: impl Fn() -> Action + 'static) -> Callback<MouseEvent> {
    let d = m.dispatcher();
    Callback::from(move |_| d.dispatch(a()))
}

/// A checkbox that flips one part of the configuration.
fn toggle(m: &UseReducerHandle<Model>, label: &str, on: bool, flip: fn(Config) -> Config) -> Html {
    let d = m.dispatcher();
    let c = m.config;
    let changed = Callback::from(move |_: Event| d.dispatch(Action::Config(flip(c))));
    html! { <label class="toggle"><input type="checkbox" checked={on} onchange={changed} />{label}</label> }
}

fn controls(m: &UseReducerHandle<Model>) -> Html {
    let d = m.dispatcher();
    let on_text = Callback::from(move |e: Event| d.dispatch(Action::Text(e.target_unchecked_into::<HtmlInputElement>().value())));
    let d = m.dispatcher();
    let c = m.config;
    let on_readers = Callback::from(move |e: Event| {
        let r = e.target_unchecked_into::<HtmlSelectElement>().value().parse().unwrap_or(1);
        d.dispatch(Action::Config(Config { readers: r, ..c }))
    });
    let c = m.config;
    html! {
        <>
        <div class="controls">
            { for presets().into_iter().enumerate().map(|(i, (name, p))| html! {
                <button class={classes!((m.prompt == p).then_some("on"))} onclick={act(m, move || Action::Preset(i))}>{name}</button>
            }) }
            <input class="sentence" type="text" value={m.text.clone()} onchange={on_text} aria-label="Prompt" />
        </div>
        <div class="controls">
            { toggle(m, " writers", c.writers, |c| Config { writers: !c.writers, ..c }) }
            { toggle(m, " duplicate detector", c.dup, |c| Config { dup: !c.dup, ..c }) }
            { toggle(m, " pooling head", c.pool, |c| Config { pool: !c.pool, ..c }) }
            { toggle(m, " gate forced open", c.gate_added, |c| Config { gate_added: !c.gate_added, ..c }) }
            { toggle(m, " gate kept out of the readers", c.gate_out, |c| Config { gate_out: !c.gate_out, ..c }) }
            <label class="toggle">{"readers "}
                <select onchange={on_readers} aria-label="Readers">
                    { for [1, 2, 3, 5, 8, 12, 20].iter().map(|&r| html! { <option value={r.to_string()} selected={r == c.readers}>{r}</option> }) }
                </select>
            </label>
            { toggle(m, " each capped", c.capped, |c| Config { capped: !c.capped, ..c }) }
            <span class="gen">{ if m.queue.is_empty() { format!("X_eTaL ran the circuit in {:.0} ms", m.ms) } else { "running\u{2026}".into() } }</span>
        </div>
        </>
    }
}

/// A token as the page writes it.
fn word(a: &Anatomy, i: usize) -> String {
    match a.tokens[i] {
        (0, _, _) => "start".into(),
        (1, c, _) => CLASSES[c].into(),
        (2, _, _) => "\u{2192}".into(),
        (_, _, l) => LABELS[l].into(),
    }
}

/// The readers' attention at the query, a bar a token.
fn looks(a: &Anatomy) -> Html {
    let bars = (0..a.n()).map(|i| {
        let w = a.attention[i];
        let kind = if i == 0 { "sink" } else if a.matches(i) { "match" } else if a.tokens[i].0 == 3 { "slot" } else { "rest" };
        html! { <div class={classes!("bar", kind)} title={format!("{}: {w:.4}", word(a, i))}>
            <b>{ if w >= 0.005 { format!("{w:.2}") } else { String::new() } }</b>
            <div class="fill" style={format!("height: {:.1}%", 100.0 * w)}></div>
            <span>{word(a, i)}</span>
        </div> }
    });
    let gate = if a.gate > 0.25 { "open" } else { "closed" };
    let body = html! { <>
        <div class="bars icl">{ for bars }</div>
        <p class="calc">{format!("On the sink {:.3}, on the slots matching the query's class {:.3}, on the other slots {:.3}; the gate at the query {:.3} ({gate}).", a.sink, a.matching, a.other, a.gate)}</p>
    </> };
    panel("Where the readers look:", "n s_elect A",
        "The query's attention in the last layer, a bar a token: the sink at position 0 (gray), the labels whose sentence has the query's class (orange), the other labels (purple). A match scores beta = 4; the sink scores 5 less 8 times the gate. With no repeated label the gate stays closed and the sink wins.",
        false, body)
}

/// The answer: each used label's logit, and the margin.
fn answer(a: &Anatomy) -> Html {
    let used = a.prompt.used();
    let top = used.iter().fold(1e-9_f64, |b, &l| b.max(a.logits[l].abs()));
    let target = a.prompt.target();
    let bars = used.iter().map(|&l| {
        let v = a.logits[l];
        html! { <div class={classes!("bar", (l == target).then_some("best"))} title={format!("{v:.4}")}>
            <b>{format!("{v:.2}")}</b>
            <div class="fill" style={format!("height: {:.1}%", 100.0 * v.max(0.0) / top)}></div>
            <span>{LABELS[l]}</span>
        </div> }
    });
    let best = strongest(&a.logits).map(|(l, _)| LABELS[l]).unwrap_or("nothing");
    let body = html! { <>
        <div class="bars digits">{ for bars }</div>
        <p class="calc">{format!("The answer should be \u{201c}{}\u{201d}: its logit less the best other label's, the margin, is {:+.3}. The readers copy most of \u{201c}{best}\u{201d}.", LABELS[target], a.margin)}</p>
        <p class="note">{format!("The read law (the paper's Proposition 1) from the three constants and the gate gives the matching mass {:.6}; the readers' softmax gave {:.6}.", a.law, a.matching)}</p>
    </> };
    panel("The answer:", "pr u:m_argin l",
        "Each label's logit at the query: the attention mass the readers put on its tokens (the readers' values are the labels), times the readers. The paper's margins are these, not log-probabilities.",
        false, body)
}

/// A block's contents as a few words.
fn cell(a: &Anatomy, i: usize, b: usize) -> String {
    let v = a.block(i, b);
    match b {
        3 | 4 => if v[0].abs() < 0.0005 { String::new() } else { format!("{:.3}", v[0]) },
        2 => ["start", "sentence", "arrow", "label"][strongest(v).map_or(0, |x| x.0)].into(),
        _ => strongest(v).map_or(String::new(), |(k, x)| {
            let name = if b == 1 { LABELS[k] } else { CLASSES[k] };
            if (x - 1.0).abs() < 0.005 { name.into() } else { format!("{name} {x:.2}") }
        }),
    }
}

fn residual(a: &Anatomy) -> Html {
    let body = html! {
        <table class="zs blocks">
            <tr><th>{"token"}</th>{ for BLOCKS.iter().map(|b| html! { <th>{b.0}</th> }) }</tr>
            { for (0..a.n()).map(|i| html! { <tr class={classes!((i + 1 == a.n()).then_some("query"))}>
                <th>{format!("{i} {}", word(a, i))}</th>
                { for (0..BLOCKS.len()).map(|b| html! { <td>{cell(a, i, b)}</td> }) }
            </tr> }) }
        </table>
    };
    panel("The residual stream:", "X",
        "Each token's row after the three layers, by named block (the position block left out). Class, label and type are the embedding; the writers fill each label's slot key and the arrows' query code, the duplicate detector a repeated label's flag, the pooling head the arrows' gate.",
        false, body)
}

fn knockouts(m: &UseReducerHandle<Model>) -> Html {
    let body = match &m.paper {
        None => html! {
            <p><button onclick={act(m, || Action::Paper)} disabled={m.queue.contains(&Job::Paper)}>
                { if m.queue.contains(&Job::Paper) { "running\u{2026}" } else { "Run the paper's knockouts" } }
            </button>{" seven runs of the circuit: the paper's Section 12, on its two prompts."}</p>
        },
        Some(rows) => html! {
            <table class="zs">
                <tr><th>{"prompt"}</th><th>{"the circuit"}</th><th>{"margin"}</th><th>{"paper"}</th><th>{"sink"}</th><th>{"paper"}</th></tr>
                { for KNOCKOUTS.iter().zip(rows.iter()).map(|((name, pr, m, s), (mx, sx))| html! { <tr>
                    <td>{pr}</td><td style="text-align: left">{name}</td>
                    <td>{format!("{mx:.2}")}</td><td>{format!("{m:.2}")}</td>
                    <td>{format!("{sx:.2}")}</td><td>{s.map_or("".into(), |s| format!("{s:.2}"))}</td>
                </tr> }) }
            </table>
        },
    };
    panel("The paper's knockouts:", "u:c_ase (two, ...)",
        "Unique labels: six classes, one demonstration each. Recurring: three classes, two each. The paper's sink 'emptied' is about 0.001 here.",
        false, body)
}

fn source(a: &Anatomy) -> Html {
    let src = shown(&a.prompt, &a.config);
    block(&src, between(&src, "u:r_un := ", "u:r_un := "))
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
                <div class="col">{ looks(a) }{ answer(a) }{ knockouts(&model) }</div>
                <div class="col">
                    { residual(a) }
                    <section class="panel code">
                        <h2>{"The program"}</h2>
                        <p class="note">{"Everything the page runs, in your browser: the core of icl-circuit.xtl (the blocks, the prompt, one head, the three layers, the readout), then the lines for your prompt; the circuit is highlighted. No weight is learned: each is a block of 0s and 1s, set by hand."}</p>
                        { source(a) }
                    </section>
                </div>
            </div>
        },
        None => html! { <div class="running"><div class="spinner"></div>{"X_eTaL is running the circuit\u{2026}"}</div> },
    };
    html! {
        <>
        <header>
            { header("In-context learning circuit", "A prompt shows demonstrations, a sentence of some class then its label, and asks for the label of a query sentence. Three attention layers, set by hand and run by X_eTaL in your browser, answer it with no weight trained: writers stamp each label with its sentence's class, a duplicate detector marks a label seen before, a pooling head turns those marks into a gate, and readers match the query's class and copy the label, unless the closed gate leaves them resting on the sink.") }
            { controls(&model) }
            { notice(&model.notice) }
        </header>
        <main>{body}</main>
        { footer() }
        </>
    }
}
