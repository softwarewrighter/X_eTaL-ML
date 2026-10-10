//! The page: step, play, start again; k and the start; the map with the
//! points, the centers and their trails; the inertia drawn by Plot;
//! Learn's step; the program.

use std::rc::Rc;

use gloo_timers::callback::Timeout;
use yew::prelude::*;

use microscope::canvas::Canvas;
use microscope::chrome::{footer, header, notice, panel, picture};
use microscope::source::{block, NONE};

use crate::micro::{k_step, Start};
use crate::model::{Action, Model, LIMIT};
use crate::view::{map, source, PX};

fn act(m: &UseReducerHandle<Model>, a: impl Fn() -> Action + 'static) -> Callback<MouseEvent> {
    let d = m.dispatcher();
    Callback::from(move |_| d.dispatch(a()))
}

fn controls(m: &UseReducerHandle<Model>) -> Html {
    let playing = m.playing;
    let done = m.settled() || m.steps >= LIMIT;
    let status = match (m.settled(), m.steps) {
        (true, s) => format!("settled after {s} steps: no point changed center"),
        (false, 0) => "the start".to_string(),
        (false, s) => format!("step {s}"),
    };
    html! { <>
        <div class="controls">
            <button class="on" onclick={act(m, move || Action::Play(!playing))} disabled={done}>{ if playing { "Pause" } else { "Run" } }</button>
            <button onclick={act(m, || Action::Tick)} disabled={playing || done}>{"One step"}</button>
            <button onclick={act(m, || Action::Reset)}>{"Start again"}</button>
            <span class="gen">{format!("{status} \u{00b7} X_eTaL: {:.0} ms a step", m.ms)}</span>
        </div>
        <div class="controls">
            <span>{"Start: "}</span>
            <button class={classes!((m.start == Start::Poor).then_some("on"))} onclick={act(m, || Action::Start(Start::Poor))}>{"the first k points (a poor start)"}</button>
            <button class={classes!((m.start == Start::Farthest).then_some("on"))} onclick={act(m, || Action::Start(Start::Farthest))}>{"farthest-first"}</button>
            <span>{" k: "}</span>
            { for (2..=8).map(|k| html! { <button class={classes!((m.k == k).then_some("on"))} onclick={act(m, move || Action::K(k))}>{k.to_string()}</button> }) }
        </div>
    </> }
}

#[function_component(App)]
pub fn app() -> Html {
    let model = use_reducer(Model::new);
    {
        // While running, take the next step once this frame has drawn.
        let d = model.dispatcher();
        let (playing, steps) = (model.playing, model.steps);
        use_effect_with((playing, steps), move |&(playing, _)| {
            let t = playing.then(|| Timeout::new(250, move || d.dispatch(Action::Tick)));
            move || drop(t)
        });
    }
    let body = match &model.after {
        Some(a) => html! {
            <div class="layout even">
                <div class="col">
                    { panel("The points and the centers:", "C1 ml:a_ssign X", "300 points in five blobs, each in the color of its nearest center; the ground is the center nearest each place; the crosses are the centers, their paths so far in gray. From the poor start, watch three centers crowd one blob: k-means only ever moves downhill, and stops there.", true, html! {
                        <div class="padbox"><Canvas rows={PX} cols={PX} rgba={Rc::new(map(&a.map, &a.points, &a.points_in, &model.trail))} class="pad" /></div>
                    }) }
                    { panel("The inertia:", "C1 ml:i_nertia X", "The sum of each point's squared distance to its center, after each step: it never rises. Drawn by the Plot library in the program's last lines.", false, html! { <>
                        <p class="calc">{format!("{:.3} now", a.inertia)}{ if model.inertias.len() > 1 { format!(", from {:.3} at the start", model.inertias[0]) } else { String::new() } }</p>
                        { for a.pictures.iter().map(|svg| picture(svg, "the inertia after each step")) }
                    </> }) }
                    { panel("A step, in the Learn library:", "X ml:k_step C", "Each point's nearest center as one-hot rows O; the centers' new places are O's columns times the points, over how many points each has; a center with none stays. No loop over points or centers.", false, html! {
                        { block(k_step(), NONE) }
                    }) }
                </div>
                <div class="col">
                    <section class="panel code">
                        <h2>{"The program"}</h2>
                        <p class="note">{"Everything the page runs, in your browser: the head of k-means.xtl (the points, the map's grid), then each run's lines; the step (or the start) is highlighted."}</p>
                        { source(model.k, model.start, model.steps > 0, model.inertias.len()) }
                    </section>
                </div>
            </div>
        },
        None => html! {},
    };
    html! {
        <>
        <header>
            { header("k-means", "Clustering without labels: each step gives every point its nearest center, then moves each center to the mean of its points, one step of a few array expressions over all the points at once. From a poor start the centers travel across the plane and can settle wrong; from a farthest-first start they find the five blobs.") }
            { controls(&model) }
            { notice(&model.notice) }
        </header>
        <main>{body}</main>
        { footer() }
        </>
    }
}
