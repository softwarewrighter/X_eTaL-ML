//! The page: step, play, start again; the ball placed by a click and
//! its speed chosen, or a state the network never saw; the window's
//! three frames, the predicted frame, the dream; the program.

use std::rc::Rc;

use gloo_timers::callback::Timeout;
use yew::prelude::*;

use microscope::canvas::Canvas;
use microscope::chrome::{footer, header, notice, panel};

use crate::micro::{Place, SPEEDS};
use crate::model::{Action, Model, LIMIT};
use crate::view::{dream, frames, predicted, source, CELL, PX, WIDE};

fn act(m: &UseReducerHandle<Model>, a: impl Fn() -> Action + 'static) -> Callback<MouseEvent> {
    let d = m.dispatcher();
    Callback::from(move |_| d.dispatch(a()))
}

fn speed(m: &UseReducerHandle<Model>, axis: usize) -> Html {
    let v = m.start[2 + axis];
    html! { for SPEEDS.iter().map(|&s| {
        let on = (s == v).then_some("on");
        let a = move || if axis == 0 { Action::Vx(s) } else { Action::Vy(s) };
        html! { <button class={classes!(on)} onclick={act(m, a)}>{ format!("{s:+}") }</button> }
    }) }
}

fn controls(m: &UseReducerHandle<Model>) -> Html {
    let playing = m.playing;
    let done = m.steps >= LIMIT;
    let status = match m.steps {
        0 => "the start".to_string(),
        n => format!("step {n}: the dreamed ball {} of {n} steps where the true one is", m.on_track()),
    };
    html! { <>
        <div class="controls">
            <button class="on" onclick={act(m, move || Action::Play(!playing))} disabled={done}>{ if playing { "Pause" } else { "Dream" } }</button>
            <button onclick={act(m, || Action::Tick)} disabled={playing || done}>{"One step"}</button>
            <button onclick={act(m, || Action::Reset)}>{"Start again"}</button>
            <button onclick={act(m, || Action::Unseen)}>{"A state it never saw"}</button>
            <span class="gen">{format!("{status} \u{00b7} X_eTaL: {:.0} ms a step", m.ms)}</span>
        </div>
        <div class="controls">
            <span>{"Click the first frame to place the ball. Speed x: "}</span>{ speed(m, 0) }
            <span>{" y: "}</span>{ speed(m, 1) }
        </div>
    </> }
}

#[function_component(App)]
pub fn app() -> Html {
    let model = use_reducer(Model::new);
    {
        // While dreaming, take the next step once this frame has drawn.
        let d = model.dispatcher();
        let (playing, steps) = (model.playing, model.steps);
        use_effect_with((playing, steps), move |&(playing, _)| {
            let t = playing.then(|| Timeout::new(300, move || d.dispatch(Action::Tick)));
            move || drop(t)
        });
    }
    let place = {
        let d = model.dispatcher();
        Callback::from(move |(r, c): (usize, usize)| {
            if c < PX {
                d.dispatch(Action::Put((c / CELL) as i64 - 1, (r / CELL) as i64 - 1));
            }
        })
    };
    let stepped = model.steps > 0;
    let s = model.start;
    let start_note = format!("the ball at ({}, {}), moving ({:+}, {:+}){}", s[0], s[1], s[2], s[3], if model.is_unseen() { ": a state it never saw" } else { ": a state it trained on" });
    let window: [Place; 3] = model.window;
    let exact_note = match (&model.last, model.exact.last()) {
        (Some(_), Some(true)) => "the true next frame exactly".to_string(),
        (Some(st), Some(false)) => format!("not quite: the likeliest place is ({}, {})", st.place.0, st.place.1),
        _ => "press One step or Dream".to_string(),
    };
    let body = html! {
        <div class="layout even">
            <div class="col">
                { panel("What it sees:", "W := 1 432 r_eshape r_avel u:f_rames 3 2 r_eshape ...", "The last three frames, 432 pixels in a row. The ball moves 1 or 2 cells a step on each axis and bounces off the walls. While dreaming, these are the network's own last three predictions.", true, html! { <>
                    <p class="calc">{ start_note }</p>
                    <div class="padbox"><Canvas rows={PX} cols={WIDE} rgba={Rc::new(frames(&window))} onclick={place} class="pad" /></div>
                </> }) }
                { panel("What it predicts:", "pr := u:p_redict W", "The next frame, a probability per pixel (darker is likelier), through 64 ReLU units and a sigmoid: nn:d_ense twice. The orange outline is where the ball really goes.", false, html! { <>
                    <p class="calc">{ exact_note }</p>
                    <div class="padbox"><Canvas rows={PX} cols={PX} rgba={Rc::new(predicted(model.last.as_ref().map_or(&[][..], |s| &s.probs), model.truth))} class="pad" /></div>
                </> }) }
                { panel("Dreaming:", "r_avel u:p_lace pr", "Fed its own predictions, sharpened to the likeliest 2 x 2 block each step: the dreamed ball (orange outline) against the true one (gray), their paths as dots. It learned the bounces from pixels alone; where the dream drifts, it was unsure.", false, html! {
                    <div class="padbox"><Canvas rows={PX} cols={PX} rgba={Rc::new(dream(&model.dreamed, &model.real))} class="pad" /></div>
                }) }
            </div>
            <div class="col">
                <section class="panel code">
                    <h2>{"The program"}</h2>
                    <p class="note">{"Everything the page runs, in your browser: the head of world-model.xtl (the world, the network from data/, the likeliest place), then the first run's lines and each step's; the step is highlighted."}</p>
                    { source(model.start, &window, model.truth, stepped) }
                </section>
            </div>
        </div>
    };
    html! {
        <>
        <header>
            { header("World model", "A network that has watched a ball bounce in a box predicts the next frame from the last three: 432 pixels in, 144 out, trained offline on four fifths of the world's 1,936 states. Place the ball, pick its speed, and let it dream: fed its own predictions, it plays the ball forward, bounces included.") }
            { controls(&model) }
            { notice(&model.notice) }
        </header>
        <main>{body}</main>
        { footer() }
        </>
    }
}
