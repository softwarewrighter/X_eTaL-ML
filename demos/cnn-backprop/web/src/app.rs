//! The page: train, pause, start again; the filters as they learn, the
//! test digits as the network reads them, the test loss and share right
//! drawn by Plot; the program, its backward pass highlighted.

use std::rc::Rc;

use gloo_timers::callback::Timeout;
use yew::prelude::*;

use microscope::canvas::Canvas;
use microscope::chrome::{footer, header, notice, panel, picture};

use crate::micro::{test_digits, SHOWN};
use crate::model::{Action, Model, CHUNK, LIMIT};
use crate::view::{filters, source, strip, FILTER_COLS};

fn act(m: &UseReducerHandle<Model>, a: impl Fn() -> Action + 'static) -> Callback<MouseEvent> {
    let d = m.dispatcher();
    Callback::from(move |_| d.dispatch(a()))
}

fn controls(m: &UseReducerHandle<Model>) -> Html {
    let playing = m.playing;
    html! {
        <div class="controls">
            <button class="on" onclick={act(m, move || Action::Play(!playing))} disabled={m.steps >= LIMIT}>{ if playing { "Pause" } else if m.steps == 0 { "Train" } else { "Go on" } }</button>
            <button onclick={act(m, || Action::Tick)} disabled={playing || m.steps >= LIMIT}>{format!("{CHUNK} steps")}</button>
            <button onclick={act(m, || Action::Reset)}>{"Start again"}</button>
            <span class="gen">{format!("step {} of {LIMIT} ({} digits seen) \u{00b7} X_eTaL: {:.0} ms a step", m.steps, 20 * m.steps, m.ms_per_step)}</span>
        </div>
    }
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
    let digits = test_digits();
    let body = match &model.after {
        Some(a) => {
            let px: Vec<&[f64]> = digits.iter().map(|d| d.1.as_slice()).collect();
            let (first, _) = model.history.first().map_or((0.0, 0.0), |h| (h.1, h.2));
            html! {
                <div class="layout even">
                    <div class="col">
                        { panel("The filters:", "K", "The 8 filters of 3 x 3 the convolution slides over every digit: blue weights negative, red positive. They start as noise; watch them become edge and stroke detectors.", false, html! {
                            <Canvas rows={3} cols={FILTER_COLS} rgba={Rc::new(filters(&a.state[0]))} class="pic filters" />
                        }) }
                        { panel("What it reads:", "nn:a_rgmax (K, B, W) u:p_robs 20 t_ake TX", "The first 20 of the 200 test digits, which it never trains on, and the digit the network reads in each: green when right, red when not.", true, html! { <>
                            <Canvas rows={28} cols={SHOWN * 29 - 1} rgba={Rc::new(strip(&px))} class="pic digits" />
                            <div class="reads">{ for digits.iter().zip(&a.reads).map(|((l, _), r)| html! { <span class={classes!(if l == r { "right" } else { "wrong" })}>{r.to_string()}</span> }) }</div>
                            <p class="calc">{format!("On all 200 test digits: loss {:.3} (from {first:.3}), {:.1}% right.", a.loss, 100.0 * a.right)}</p>
                        </> }) }
                        { panel("The test loss and share right:", "p:c_hart!", "After every run, on the 200 test digits; drawn by the Plot library in the program's last lines.", false, html! {
                            { for a.pictures.iter().map(|svg| picture(svg, "the test loss and share right after each run")) }
                        }) }
                    </div>
                    <div class="col">
                        <section class="panel code">
                            <h2>{"The program"}</h2>
                            <p class="note">{"Everything the page runs, in your browser: the head of cnn-backprop.xtl (the digits, the network, its backward pass, Adam), then each run's lines; the backward pass is highlighted. The page keeps the state between runs in files the program reads, as it reads its digits from data/."}</p>
                            { source(model.steps > 0, CHUNK, model.history.len()) }
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
            { header("CNN training", "X_eTaL trains a tiny convolutional network in your browser, from random weights, on 600 handwritten digits: 8 filters of 3 x 3, ReLU, 2 x 2 max-pooling, a dense layer, softmax. The backward pass undoes each stage in turn, a few array expressions each, checked against finite differences; Adam takes 20 digits a step.") }
            { controls(&model) }
            { notice(&model.notice) }
        </header>
        <main>{body}</main>
        { footer() }
        </>
    }
}
