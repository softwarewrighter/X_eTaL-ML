//! The page: play, pause, start again, the learning rate; the decision
//! map as it learns; the loss and accuracy curves; Adam's step in the
//! program.

use std::rc::Rc;

use gloo_timers::callback::Timeout;
use web_sys::HtmlInputElement;
use yew::prelude::*;

use microscope::canvas::Canvas;
use microscope::chrome::{footer, header, notice, panel, picture};
use microscope::source::code;

use crate::micro::SIDE;
use crate::model::{Action, Model, CHUNK, LIMIT};
use crate::view::{map, source};

fn act(m: &UseReducerHandle<Model>, a: impl Fn() -> Action + 'static) -> Callback<MouseEvent> {
    let d = m.dispatcher();
    Callback::from(move |_| d.dispatch(a()))
}

fn controls(m: &UseReducerHandle<Model>) -> Html {
    let d = m.dispatcher();
    let on_lr = Callback::from(move |e: Event| d.dispatch(Action::Rate(e.target_unchecked_into::<HtmlInputElement>().value().parse().unwrap_or(0.02))));
    let playing = m.playing;
    html! {
        <div class="controls">
            <button class="on" onclick={act(m, move || Action::Play(!playing))}>{ if playing { "Pause" } else if m.steps == 0 { "Train" } else { "Go on" } }</button>
            <button onclick={act(m, || Action::Tick)} disabled={playing}>{format!("{CHUNK} steps")}</button>
            <button onclick={act(m, || Action::Reset)}>{"Start again"}</button>
            <label class="slider">{"learning rate "}{code("lr")}
                <input type="range" min="0.002" max="0.2" step="0.002" value={m.lr.to_string()} onchange={on_lr} />
                <b>{format!("{:.3}", m.lr)}</b>
            </label>
            <span class="gen">{format!("step {} of {LIMIT} \u{00b7} X_eTaL: {:.1} ms a step", m.steps, m.ms_per_step)}</span>
        </div>
    }
}

fn curves(m: &UseReducerHandle<Model>) -> Html {
    let (l, r) = m.history.last().map_or((0.0, 0.0), |x| (x.1, x.2));
    let first = m.history.first().map_or(0.0, |x| x.1);
    let charts = m.after.as_ref().map(|a| a.pictures.clone()).unwrap_or_default();
    let body = html! { <>
        <p class="calc">{"loss "}<b>{format!("{l:.3}")}</b>{format!(" (from {first:.3}) \u{00b7} points right ")}<b>{format!("{:.1}%", 100.0 * r)}</b></p>
        { for charts.iter().map(|svg| picture(svg, "the loss and the share of points right after each run")) }
        <p class="note">{"Drawn by the Plot library's p:c_hart!, in the program's last lines."}</p>
    </> };
    panel("The loss and the points right:", "p:c_hart!", "Over the steps so far, measured on the 300 training points after every run.", false, body)
}

#[function_component(App)]
pub fn app() -> Html {
    let model = use_reducer(Model::new);
    {
        // While playing, run the next steps once this frame has drawn.
        let d = model.dispatcher();
        let (playing, steps) = (model.playing, model.steps);
        use_effect_with((playing, steps), move |&(playing, _)| {
            let t = playing.then(|| Timeout::new(30, move || d.dispatch(Action::Tick)));
            move || drop(t)
        });
    }
    let body = match &model.after {
        Some(a) => html! { <>
            <div class="layout even">
                <div class="col">
                    { panel("What it decides:", "nn:a_rgmax u:f_orward grid", "The arm the network picks at every point of the plane, and the 300 training points in their own arm's color. At the start the weights are small and random; watch the regions bend to follow the spiral.", true, html! {
                        <div class="padbox"><Canvas rows={SIDE} cols={SIDE} rgba={Rc::new(map(&a.map, &a.points, &a.arms))} class="pad" /></div>
                    }) }
                    { curves(&model) }
                </div>
                <div class="col">
                    <section class="panel code">
                        <h2>{"The program"}</h2>
                        <p class="note">{"Everything the page runs, in your browser: the head of train-live.xtl (the spiral made in X_eTaL, the network, its gradient, Adam), then each run's lines; Adam's step is highlighted."}</p>
                        { source(model.lr, model.steps > 0, CHUNK) }
                    </section>
                </div>
            </div>
        </> },
        None => html! {},
    };
    html! {
        <>
        <header>
            { header("Training live", "X_eTaL trains a small network in your browser: 2 inputs, 16 tanh units, a softmax over 3 arms of a spiral. Each step is the backprop microscope's four gradient lines and Adam, written out in X_eTaL; the page runs 25 steps at a time and redraws what the network decides.") }
            { controls(&model) }
            { notice(&model.notice) }
        </header>
        <main>{body}</main>
        { footer() }
        </>
    }
}
