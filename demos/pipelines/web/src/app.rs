//! The page: an example to pick; its pipeline, what the pipeline became
//! and the same layer in plain X_eTaL side by side (for the CNN layer,
//! the equation as a paper prints it first); the checks that they
//! agree; and the whole program the page ran.

use std::rc::Rc;

use gloo_timers::callback::Timeout;
use yew::prelude::*;

use microscope::chrome::{footer, header, panel};
use microscope::palette::Composer;
use microscope::source::{between, block, code, NONE};

use microscope::stageview::{no_stage, row_picker, stage_view, Split, Turn};
use crate::explore::Stages;
use crate::micro::{calls, clickable, expansion_ranges, program, run, stage_ranges, Example, Run, EXAMPLES, STARTERS};

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
    let c = chosen.clone();
    let pick = row_picker(Callback::from(move |k| c.set(Some(Chosen::Drawn(k)))));
    if calls.is_empty() {
        return html! {};
    }
    let paper = calls.iter().any(|c| !c.contains("flow:m_ath<") && !c.contains(":= r#\""));
    let typed = calls.iter().any(|c| c.contains(":= r#\""));
    let note = format!("{}The layer as a pipeline of equations, drawn by Flow's m_ath< from the very text that computes it: a row a stage, an arrow between, each label after the stage it names (a triangle, as X_eTaL draws $, with its shape), each equation as X_eTaL-libraries' Equations draws it; checked equal to the pipeline below as the page runs.",
        if paper { "The equation as a paper prints it first, read by Equations (e:e_q< its value, e:t_ex< its LaTeX, one reading of the same text; a sum whose term uses the one before is one step, applied by p_ower). " } else if typed { "The rule as a paper writes it first, typed as LaTeX (Equations has no derivatives yet, ask E8). " } else { "" });
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

/// What the chosen stage gives: each value (a tuple's parts each), its
/// shape, its numbers as a table colored by sign and size when it fits
/// (64 items at most, slices of a rank-3 array), else its range and its
/// first numbers; then in 3D, the stages it reads and what it gives.
fn gives(e: &Example, r: &Run, chosen: &UseStateHandle<Option<Chosen>>, turn: &UseStateHandle<Turn>) -> Html {
    let (math, pipe, _) = calls(e);
    let (trace, k, whose, text) = match **chosen {
        Some(Chosen::Drawn(k)) => (&r.drawn, k, "of the pipeline of equations", math.into_iter().find(|c| c.contains("flow:m_ath<")).unwrap_or_default()),
        Some(Chosen::Piped(k)) => (&r.stages, k, "of the pipeline", pipe),
        None => return no_stage(),
    };
    stage_view(trace, &text, k, whose, turn)
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
    // The example's run, once per example (X_eTaL runs for seconds on the
    // BERT layer): after the spinner is drawn, not on every render.
    let ran = use_state(|| None::<(usize, Rc<Result<Run, String>>)>);
    {
        let ran = ran.clone();
        use_effect_with(*picked, move |&p| {
            let t = (p < EXAMPLES.len()).then(|| Timeout::new(60, move || ran.set(Some((p, Rc::new(run(&EXAMPLES[p])))))));
            move || drop(t)
        });
    }
    // The last choice is composing one's own, from the palette.
    let main = if *picked >= EXAMPLES.len() {
        html! { <Composer starters={STARTERS.to_vec()} after={Some(Callback::from(|src: String| html! { <Stages {src} /> }))} /> }
    } else {
        let e = EXAMPLES[*picked];
        let src = program(&e);
        let from = format!("\n{} := ", e.pipe);
        let the_program = html! {
            <section class="panel code">
                <h2>{"The program"}</h2>
                <p class="note">{"Everything the page runs, in your browser: the head of the program (its imports and data), this example's section, then the pipeline's value and the checks. The pipeline is highlighted."}</p>
                { block(&src, between(&src, &from, &from)) }
            </section>
        };
        match ran.as_ref().filter(|(p, _)| *p == *picked).map(|(_, r)| r.clone()) {
            None => html! { <>
                <div class="running"><span class="spinner"></span>{format!("X_eTaL is running {} in your browser\u{2026}", e.title)}</div>
                { the_program }
            </> },
            Some(r) => match r.as_ref() {
                Ok(r) => html! {
                    <Split idle={chosen.is_none()} close={{ let c = chosen.clone(); Callback::from(move |_| c.set(None)) }}
                        left={html! { <>{ about(&e) }{ math(&e, r, &chosen) }{ columns(&e, r, &chosen) }{ agree(&e, r) }{ the_program }</> }}
                        right={gives(&e, r, &chosen, &turn)} />
                },
                Err(why) => html! { <><p class="error">{format!("X_eTaL stopped: {why}")}</p>{ the_program }</> },
            },
        }
    };
    html! {
        <>
        <header>
            { header("Pipelines", "A layer written as data flow: Flow's pipelines, left to right, |> between the stages, on this repository's libraries, each beside what it became and the same layer in plain X_eTaL. A CNN layer and a diffusion sampler are shown four ways: as a paper prints them, as a pipeline, as its expansion and as plain X_eTaL, all computing the same values. Every example is also written as a pipeline of equations, drawn as math a row a stage. Compose your own from a palette of the notations' parts.") }
            { picker(&picked, &chosen) }
        </header>
        <main>
            <div class="wide">{ main }</div>
        </main>
        { footer() }
        </>
    }
}
