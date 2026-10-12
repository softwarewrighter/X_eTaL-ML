//! The page: invent names, the temperature, the starting letters; the
//! names; for the one picked, what the model thought at each letter and
//! where each head looked; the program, its forward pass highlighted.

use web_sys::HtmlInputElement;
use yew::prelude::*;

use microscope::chrome::{footer, header, notice, panel};
use microscope::color::{ramp, GLOW};
use microscope::source::code;

use crate::micro::{chars, Names, TOKENS};
use crate::model::{Action, Model};
use crate::view::source;

/// A token as a letter; the start and end token as a mark.
fn letter(tok: usize) -> String {
    match chars().get(tok.wrapping_sub(1)) {
        Some(c) => c.to_string(),
        None => "\u{2022}".to_string(),
    }
}

fn controls(m: &UseReducerHandle<Model>) -> Html {
    let (d1, d2, d3) = (m.dispatcher(), m.dispatcher(), m.dispatcher());
    let invent = Callback::from(move |_: MouseEvent| d1.dispatch(Action::Invent));
    let on_t = Callback::from(move |e: Event| d2.dispatch(Action::Temp(e.target_unchecked_into::<HtmlInputElement>().value().parse().unwrap_or(0.5))));
    let on_s = Callback::from(move |e: Event| d3.dispatch(Action::Start(e.target_unchecked_into::<HtmlInputElement>().value())));
    html! {
        <div class="controls">
            <button class="on" onclick={invent}>{"Invent names"}</button>
            <label class="slider">{"temperature "}{code("t")}
                <input type="range" min="0.1" max="1.5" step="0.05" value={m.t.to_string()} onchange={on_t} />
                <b>{format!("{:.2}", m.t)}</b>
            </label>
            <span>{" starting with "}</span>
            <input class="sentence short" type="text" value={m.start.clone()} onchange={on_s} aria-label="Starting letters" placeholder="any letters" />
            <span class="gen">{format!("X_eTaL: {:.0} ms", m.ms)}</span>
        </div>
    }
}

fn names(m: &UseReducerHandle<Model>, r: &Names) -> Html {
    html! { <div class="names">
        { for r.names.iter().enumerate().map(|(i, w)| {
            let d = m.dispatcher();
            let pick = Callback::from(move |_: MouseEvent| d.dispatch(Action::Pick(i + 1)));
            html! { <button class={classes!((m.pick == i + 1).then_some("on"))} onclick={pick}>{ if w.is_empty() { "(nothing)".to_string() } else { w.clone() } }</button> }
        }) }
    </div> }
}

/// For each place of the picked name: the letter so far and the five
/// likeliest next tokens with their probabilities.
fn next(r: &Names, word: &str) -> Html {
    let typed: Vec<String> = std::iter::once("\u{2022}".to_string()).chain(word.chars().map(|c| c.to_string())).collect();
    html! { <table class="next">
        { for r.probs.chunks(TOKENS).enumerate().map(|(i, row)| {
            let mut top: Vec<(usize, f64)> = row.iter().copied().enumerate().map(|(k, p)| (k + 1, p)).collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            let chosen = word.chars().nth(i).map_or(28, |c| chars().iter().position(|&x| x == c).map_or(28, |p| p + 1));
            let chosen = if i + 1 == r.n { TOKENS } else { chosen };
            html! { <tr>
                <th>{ typed[..=i].concat() }</th>
                { for top.iter().take(5).copied().chain(top.iter().skip(5).filter(|t| t.0 == chosen).copied()).map(|(k, p)| html! {
                    <td class={classes!((k == chosen).then_some("chosen"))}>
                        <span class="tok">{letter(k)}</span>
                        <span class="bar" style={format!("width: {:.0}px", 4.0 + 60.0 * p)}></span>
                        <span class="p">{format!("{:.0}%", 100.0 * p)}</span>
                    </td>
                }) }
            </tr> }
        }) }
    </table> }
}

/// A head's attention as a small table: rows look, columns are looked at.
fn head_table(h: &[f64], n: usize, word: &str, j: usize) -> Html {
    let labels: Vec<String> = std::iter::once("\u{2022}".to_string()).chain(word.chars().map(|c| c.to_string())).collect();
    html! { <figure class="arr">
        <table class="zs heat look">
            <tr><th></th>{ for labels.iter().map(|l| html! { <th>{l.clone()}</th> }) }</tr>
            { for (0..n).map(|i| html! { <tr><th>{labels[i].clone()}</th>{ for (0..n).map(|k| {
                let w = h[i * n + k];
                let c = ramp(&GLOW, w);
                html! { <td style={format!("background: rgb({}, {}, {})", c[0], c[1], c[2])} title={format!("{w:.3}")}></td> }
            }) }</tr> }) }
        </table>
        <figcaption>{format!("head {j}")}</figcaption>
    </figure> }
}

#[function_component(App)]
pub fn app() -> Html {
    let model = use_reducer(Model::new);
    let body = match &model.names {
        Some(r) => {
            let word = r.names.get(model.pick - 1).cloned().unwrap_or_default();
            html! {
                <div class="layout even">
                    <div class="col">
                        { panel("Names it invents:", "(t, s) u:n_ame u", "Ten names, a letter at a time: at each step the model scores the 27 tokens (26 letters and the end), and one is drawn by a uniform number from your browser. Click a name to look into it.", true, names(&model, r)) }
                        { panel(&format!("What it thought, letter by letter ({word}):"), "t sm:p_robs u:g_pt ids", "After the start (\u{2022}) and each letter, the five likeliest next tokens at this temperature; the one drawn is outlined, added at the end when it was not among them (\u{2022} again ends the name).", false, next(r, &word)) }
                        { panel("Where each head looks:", "at:w_eights at:s_cores", "The 4 attention heads over the picked name: each row is a place, bright where it looks (only back, never ahead: the causal mask).", false, html! {
                            <div class="arrs">{ for r.heads.iter().enumerate().map(|(j, h)| head_table(h, r.n, &word, j + 1)) }</div>
                        }) }
                    </div>
                    <div class="col">
                        <section class="panel code">
                            <h2>{"The program"}</h2>
                            <p class="note">{"Everything the page runs, in your browser: the head of micro-gpt.xtl (the weights read from data/, the forward pass on the Norm, Attention and Sample libraries, the generator), then each set's lines; the forward pass is highlighted."}</p>
                            { source(model.t, &model.start, model.pick) }
                        </section>
                    </div>
                </div>
            }
        }
        None => html! {},
    };
    html! {
        <>
        <header>
            { header("Tiny GPT", "Karpathy's microgpt in X_eTaL: one transformer layer, 16 numbers wide, 4 attention heads, 4,192 weights trained on 32,033 baby names. It invents new names a letter at a time, in your browser; the forward pass gives the same scores as the Rust model it was trained in.") }
            { controls(&model) }
            { notice(&model.notice) }
        </header>
        <main>{body}</main>
        { footer() }
        </>
    }
}
