//! The page: a sentence, the attention weights as a heatmap of words
//! against words, the scores, one word's query against every key, its
//! output, and the program.

use web_sys::HtmlInputElement;
use yew::prelude::*;

use microscope::chrome::{chip, footer, header, notice, panel};
use microscope::color::{ramp, DIVERGE, GLOW};
use microscope::source::code;

use crate::micro::{tokens, Anatomy, DIMS, FEATURES};
use crate::model::{Action, Model, PRESETS};
use crate::view::{source, Stage, STAGES};

fn act(m: &UseReducerHandle<Model>, a: impl Fn() -> Action + 'static) -> Callback<MouseEvent> {
    let d = m.dispatcher();
    Callback::from(move |_| d.dispatch(a()))
}

/// A cell's background, and text that stays readable on it.
fn rgb(c: [u8; 3]) -> String {
    let light = 0.299 * c[0] as f64 + 0.587 * c[1] as f64 + 0.114 * c[2] as f64 > 140.0;
    format!("background: rgb({}, {}, {}); color: {}", c[0], c[1], c[2], if light { "#111" } else { "#eee" })
}

fn stage_chip(m: &UseReducerHandle<Model>, s: Stage, n: usize) -> Html {
    let (name, src, dims, meaning): (&str, &str, Vec<usize>, &str) = match s {
        Stage::Embed => ("embed", "ids s_elect F", vec![n, 8], "each word's 8 features"),
        Stage::QueryKey => ("queries, keys", "X '+ '* i_nner Wq", vec![n, 2], "what each word looks for, and what it offers"),
        Stage::Scores => ("scores", "u:s_cores X", vec![n, n], "every query against every key: Q K^T / sqrt 2"),
        Stage::Weights => ("weights", "X u:a_ttend M", vec![n, n], "softmax of each row (masked words get nothing)"),
        Stage::Output => ("output", "A '+ '* i_nner X", vec![n, 8], "each word, the weighted mix of what it looks at"),
    };
    chip(name, src, &dims, meaning, m.focus == s, act(m, move || Action::Focus(s)))
}

fn controls(m: &UseReducerHandle<Model>) -> Html {
    let d = m.dispatcher();
    let on_text = Callback::from(move |e: Event| d.dispatch(Action::Sentence(e.target_unchecked_into::<HtmlInputElement>().value())));
    let d = m.dispatcher();
    let causal = m.causal;
    let on_mask = Callback::from(move |_: Event| d.dispatch(Action::Causal(!causal)));
    html! {
        <div class="controls">
            <input class="sentence" type="text" value={m.sentence.clone()} onchange={on_text} aria-label="Sentence" />
            { for PRESETS.iter().map(|p| { let p = p.to_string(); let label = p.split_whitespace().last().unwrap_or("").to_string(); html! {
                <button class={classes!((m.sentence == p).then_some("on"))} onclick={act(m, move || Action::Sentence(p.clone()))}>{format!("\u{2026} {label}")}</button>
            } }) }
            <label><input type="checkbox" checked={m.causal} onchange={on_mask} />{" causal mask"}</label>
            <span class="gen">{format!("X_eTaL ran the head in {:.0} ms", m.ms)}</span>
        </div>
    }
}

/// An n x n table of words against words, each cell colored by its value.
fn matrix(m: &UseReducerHandle<Model>, a: &Anatomy, values: &[f64], color: impl Fn(f64) -> [u8; 3], digits: usize) -> Html {
    let n = a.n();
    html! {
        <table class="zs heat">
            <tr><th></th>{ for a.words.iter().map(|w| html! { <th class="ch" title={w.clone()}>{w.chars().take(3).collect::<String>()}</th> }) }</tr>
            { for (0..n).map(|i| {
                let d = m.dispatcher();
                html! { <tr class={classes!("pick", (i == m.row).then_some("sel"))} onclick={Callback::from(move |_| d.dispatch(Action::Row(i)))}>
                    <th>{&a.words[i]}</th>
                    { for (0..n).map(|j| { let v = values[i * n + j]; html! { <td style={rgb(color(v))} title={format!("{v:.3}")}>{format!("{v:.digits$}")}</td> } }) }
                </tr> }
            }) }
        </table>
    }
}

fn weights(m: &UseReducerHandle<Model>, a: &Anatomy) -> Html {
    let body = matrix(m, a, &a.a, |v| ramp(&GLOW, v), 2);
    panel("Weights:", "A := X u:a_ttend M",
        "Each row is one word's attention: the softmax of its scores, so a row adds up to 1. Masked words (the causal mask: words after it) get nothing. Click a row to follow that word.",
        m.focus == Stage::Weights, body)
}

fn scores(m: &UseReducerHandle<Model>, a: &Anatomy) -> Html {
    let big = a.s.iter().fold(1e-9_f64, |b, v| b.max(v.abs()));
    let body = matrix(m, a, &a.s, move |v| ramp(&DIVERGE, 0.5 + v / (2.0 * big)), 1);
    panel("Scores:", "S := u:s_cores X",
        "Every query against every key in one matrix product with a transpose, Q (o_\\ K), divided by the square root of their 2 dimensions.",
        m.focus == Stage::Scores, body)
}

fn inspector(m: &UseReducerHandle<Model>, a: &Anatomy) -> Html {
    let (n, i) = (a.n(), m.row);
    let w = &a.words[i];
    let rows = (0..n).map(|j| html! { <tr class={classes!((a.a[i * n + j] >= 0.5).then_some("best"))}>
        <td>{&a.words[j]}</td>
        <td>{code(&format!("{} {}", a.k[2 * j], a.k[2 * j + 1]))}</td>
        <td>{format!("{:.3}", a.s[i * n + j])}</td>
        <td>{format!("{:.3}", a.a[i * n + j])}</td>
    </tr> });
    let top = a.top(i).map_or("no word gets half its weight: it spreads evenly".to_string(), |j| format!("it looks at \u{201c}{}\u{201d}", a.words[j]));
    let feats = (0..8).filter(|&f| a.y[i * 8 + f].abs() >= 0.005).map(|f| format!("{} {:.2}", FEATURES[f], a.y[i * 8 + f])).collect::<Vec<_>>().join(", ");
    html! { <section class="panel">
        <h2>{format!("\u{201c}{w}\u{201d}: its query against every key")}</h2>
        <p class="note">{format!("Its query (looks for {} {}, {} {}): ", DIMS[0], a.q[2 * i], DIMS[1], a.q[2 * i + 1])}{top}{"."}</p>
        <table class="zs outs"><tr><th>{"word"}</th><th>{"key"}</th><th>{"score"}</th><th>{"weight"}</th></tr>{ for rows }</table>
        <p class="calc">{"Its output, the weighted mix of the words' features: "}{if feats.is_empty() { "nothing".into() } else { feats }}</p>
    </section> }
}

#[function_component(App)]
pub fn app() -> Html {
    let model = use_reducer(Model::new);
    let n = model.anatomy.as_ref().map_or(0, |a| a.n());
    let ids: Vec<usize> = tokens(&model.sentence).iter().map(|t| t.1).collect();
    let body = match &model.anatomy {
        Some(a) => html! { <>
            <div class="layout even">
                <div class="col">{ weights(&model, a) }{ inspector(&model, a) }</div>
                <div class="col">
                    { scores(&model, a) }
                    <section class="panel code">
                        <h2>{"The program"}</h2>
                        <p class="note">{"Everything the page runs, in your browser: the vocabulary and head read from data/ and the core of attention.xtl, then the lines for your sentence; the stage you pick is highlighted."}</p>
                        { source(&ids, model.causal, model.focus) }
                    </section>
                </div>
            </div>
        </> },
        None => html! {},
    };
    html! {
        <>
        <header>
            { header("Attention microscope", "One head of attention, run by X_eTaL in your browser: every word's query meets every word's key in one matrix product, softmax turns each row into weights, and each word's output is the weighted mix of the words it looks at. The head is set by hand so that a word looks for what it can describe: tired for the animal, wide for the street.") }
            <nav class="timeline">{ for STAGES.iter().map(|&s| stage_chip(&model, s, n)) }</nav>
            { controls(&model) }
            { notice(&model.notice) }
        </header>
        <main>{body}</main>
        { footer() }
        </>
    }
}
