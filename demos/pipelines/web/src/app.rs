//! The page: an example to pick; its pipeline, what the pipeline became
//! and the same layer in plain X_eTaL side by side (for the CNN layer,
//! the equation as a paper prints it first); the checks that they
//! agree; and the whole program the page ran.

use yew::prelude::*;

use microscope::chrome::{footer, header, panel};
use microscope::palette::Composer;
use microscope::source::{between, block, code, NONE};

use crate::micro::{calls, program, run, Example, Run, EXAMPLES, STARTERS};

fn picker(picked: &UseStateHandle<usize>) -> Html {
    html! {
        <div class="controls">
            <span>{"An example: "}</span>
            { for EXAMPLES.iter().map(|e| e.title).chain(["Compose your own"]).enumerate().map(|(i, title)| {
                let p = picked.clone();
                html! { <button class={classes!((**picked == i).then_some("on"))} onclick={Callback::from(move |_| p.set(i))}>{title}</button> }
            }) }
        </div>
    }
}

/// The equations, each drawn as math where KaTeX has loaded
/// (index.html), its LaTeX as text where it has not, above the call
/// that reads it.
fn math(e: &Example, r: &Run) -> Html {
    let calls = calls(e).0;
    if calls.is_empty() {
        return html! {};
    }
    panel("As a paper prints it", "e:e_q<",
        "The equation, read by X_eTaL-libraries' Equations: e:e_q< gives its value, e:t_ex< its LaTeX, from one reading of the same text. A sum whose term uses the one before (a recurrence) is one step, applied by p_ower.",
        false, html! { <>
            { for calls.iter().zip(&r.tex).map(|(call, tex)| html! { <>
                <div class="tex" data-tex={tex.clone()}>{tex.clone()}</div>
                { block(call, NONE) }
            </> }) }
        </> })
}

fn columns(e: &Example, r: &Run) -> Html {
    let (_, pipe, plain) = calls(e);
    html! {
        <div class="layout three">
            <div class="col">{ panel("The pipeline", "flow:p_ipe<",
                "Data flow, left to right: the data, then each stage, |> between them; it names the previous value, $ labels a stage's value for later stages.",
                true, block(&pipe, NONE)) }</div>
            <div class="col">{ panel("What it became", "xetal expand",
                "What the Flow macro wrote in its place when the program was compiled: one line per stage, each value named, type-checked like any code.",
                false, block(&r.expansion, NONE)) }</div>
            <div class="col">{ panel("Plain X_eTaL", "the library",
                "The same layer as this repository's libraries write it, without a pipeline.",
                false, block(&plain, NONE)) }</div>
        </div>
    }
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
    // The last choice is composing one's own, from the palette.
    let main = if *picked >= EXAMPLES.len() {
        html! { <Composer starters={STARTERS.to_vec()} /> }
    } else {
        let e = EXAMPLES[*picked];
        let src = program(&e);
        let body = match run(&e) {
            Ok(r) => html! { <>{ math(&e, &r) }{ columns(&e, &r) }{ agree(&e, &r) }</> },
            Err(why) => html! { <p class="error">{format!("X_eTaL stopped: {why}")}</p> },
        };
        let from = format!("{} := ", e.pipe);
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
            { header("Pipelines", "A layer written as data flow: Flow's pipelines, left to right, |> between the stages, on this repository's libraries, each beside what it became and the same layer in plain X_eTaL. A CNN layer and a diffusion sampler are shown four ways: as a paper prints them, as a pipeline, as its expansion and as plain X_eTaL, all computing the same values. Compose your own from a palette of the notations' parts.") }
            { picker(&picked) }
        </header>
        <main>
            { main }
        </main>
        { footer() }
        </>
    }
}
