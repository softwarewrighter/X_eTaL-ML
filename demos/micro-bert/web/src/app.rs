//! The page: a name with a blank; the letters the model would put in
//! it; where each head of each layer looks; the model as math; the
//! program, its encoder layer (a pipeline) highlighted.

use web_sys::HtmlInputElement;
use yew::prelude::*;

use microscope::chrome::{footer, header, notice, panel};
use microscope::color::{ramp, GLOW};
use microscope::source::code;

use crate::micro::{accuracy, chars, Fill};
use crate::model::{Action, Model};
use crate::view::source;

const SAMPLES: [&str; 6] = ["ma_y", "em_a", "_lice", "sophi_", "jo_n", "_"];

fn controls(m: &UseReducerHandle<Model>) -> Html {
    let d = m.dispatcher();
    let on_name = Callback::from(move |e: Event| d.dispatch(Action::Name(e.target_unchecked_into::<HtmlInputElement>().value())));
    html! {
        <div class="controls">
            <span>{"A name with a blank "}{code("_")}{": "}</span>
            <input class="sentence short" type="text" value={m.name.clone()} onchange={on_name} aria-label="A name with a blank" placeholder="ma_y" />
            { for SAMPLES.iter().map(|s| {
                let d = m.dispatcher();
                let pick = Callback::from(move |_: MouseEvent| d.dispatch(Action::Name(s.to_string())));
                html! { <button class={classes!((m.name == *s).then_some("on"))} onclick={pick}>{*s}</button> }
            }) }
            <span class="gen">{format!("X_eTaL: {:.0} ms", m.ms)}</span>
        </div>
    }
}

/// The letters for the blank, likeliest first: the eight likeliest.
fn letters(f: &Fill) -> Html {
    let cs = chars();
    let mut top: Vec<(usize, f64)> = f.probs.iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    html! { <table class="next">
        { for top.iter().take(8).map(|&(k, p)| html! { <tr><td>
            <span class="tok" style="display: inline-block; min-width: 9ch">{ f.name.replace('_', &cs[k].to_string()) }</span>
            <span class="bar" style={format!("width: {:.0}px", 4.0 + 160.0 * p)}></span>
            <span class="p">{format!("{:.0}%", 100.0 * p)}</span>
        </td></tr> }) }
    </table> }
}

/// A head's attention: rows look, columns are looked at, both ways.
fn head_table(h: &[f64], name: &str, caption: String) -> Html {
    let labels: Vec<String> = name.chars().map(|c| c.to_string()).collect();
    let n = labels.len();
    html! { <figure class="arr">
        <table class="zs heat look">
            <tr><th></th>{ for labels.iter().map(|l| html! { <th>{l.clone()}</th> }) }</tr>
            { for (0..n).map(|i| html! { <tr><th>{labels[i].clone()}</th>{ for (0..n).map(|k| {
                let w = h[i * n + k];
                let c = ramp(&GLOW, w);
                html! { <td style={format!("background: rgb({}, {}, {})", c[0], c[1], c[2])} title={format!("{w:.3}")}></td> }
            }) }</tr> }) }
        </table>
        <figcaption>{caption}</figcaption>
    </figure> }
}

fn math(f: &Fill) -> Html {
    html! { <>
        { for f.tex.iter().map(|t| html! { <div class="tex" style="overflow-x: auto" data-tex={t.clone()}>{t.clone()}</div> }) }
    </> }
}

#[function_component(App)]
pub fn app() -> Html {
    let model = use_reducer(Model::new);
    let [right, three, total, guess] = accuracy();
    let pct = |k: usize| 100.0 * k as f64 / total.max(1) as f64;
    let body = match &model.fill {
        Some(f) => html! {
            <div class="layout even">
                <div class="col">
                    { panel("The blank, filled:", "u:f_ill s", &format!("The probability of each letter for the blank, from the letters on both sides of it: the scores at the blank's place after both layers, through softmax. On 1,000 names it never saw, its likeliest letter is right {:.0}% of the time and the right one is in its top three {:.0}% (always guessing the commonest letter, a: {:.0}%).", pct(right), pct(three), pct(guess)), true, letters(f)) }
                    { panel("Where each head looks:", "at:w_eights at:s_cores", "Each layer's 2 heads over the name: each row is a place, bright where it looks. Every place sees every other, before and after it (at:a_ll): the blank looks at its neighbors on both sides, which a GPT's causal mask would forbid.", false, html! {
                        <div class="arrs">{ for f.heads.iter().enumerate().map(|(j, h)| head_table(h, &f.name, format!("layer {}, head {}", j / 2 + 1, j % 2 + 1))) }</div>
                    }) }
                    { panel("BERT in one line, and its attention as equations:", "e:t_ex<", "The model as a paper prints it: L encoder layers, each attention (MHA) then the feed-forward layer (FFN), each added back (I +) and layer-normed (LN), on the letters' and places' embeddings; written as LaTeX by hand, since Equations has no composition of functions. Below it, one head's attention as Equations draws it from the strings it also computes (the command-line program checks they give head 1's values). The layer itself is the pipeline highlighted in the program.", false, math(f)) }
                </div>
                <div class="col">
                    <section class="panel code">
                        <h2>{"The program"}</h2>
                        <p class="note">{"Everything the page runs, in your browser: the head of micro-bert.xtl (the weights read from data/, the encoder layer as a Flow pipeline on the NN, Activations, Norm and Attention libraries, applied L times by p_ower), then each name's lines; the encoder layer is highlighted."}</p>
                        { source(&model.name) }
                    </section>
                </div>
            </div>
        },
        None => html! {},
    };
    html! {
        <>
        <header>
            { header("Tiny BERT", "The encoder half of a transformer in X_eTaL: 2 layers, 16 numbers wide, 2 attention heads, 7,594 weights, trained on 31,033 baby names to fill a blank letter. Every letter looks both ways, before and after it. One encoder layer is a pipeline, applied twice; the forward pass gives the same scores as the Rust model it was trained in.") }
            { controls(&model) }
            { notice(&model.notice) }
        </header>
        <main>{body}</main>
        { footer() }
        </>
    }
}
