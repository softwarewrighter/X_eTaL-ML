//! The page: draw eight pictures from new noise, or the same noise
//! again; every picture's journey from noise to a shape; one of them
//! large; the program.

use std::rc::Rc;

use yew::prelude::*;

use microscope::canvas::Canvas;
use microscope::chrome::{footer, header, notice, panel};

use crate::micro::{gaussian, on, N, PICTURES, SIDE};
use crate::model::{Action, Model};
use crate::view::{journey, large, source, BH, BW, CELL, GAP, JH, JW};

fn act(m: &UseReducerHandle<Model>, a: impl Fn() -> Action + 'static) -> Callback<MouseEvent> {
    let d = m.dispatcher();
    Callback::from(move |_| d.dispatch(a()))
}

/// PICTURES x N Gaussian numbers from the browser's randomness.
fn fresh() -> Vec<f64> {
    let u = || js_sys::Math::random().max(1e-12);
    (0..PICTURES * N).map(|_| gaussian(u(), u())).collect()
}

fn controls(m: &UseReducerHandle<Model>) -> Html {
    html! {
        <div class="controls">
            <button class="on" onclick={act(m, || Action::Draw(fresh()))}>{"Draw eight more"}</button>
            <button onclick={act(m, || Action::Again)}>{"The same noise again"}</button>
            <span class="gen">{format!("run {} \u{00b7} X_eTaL: {:.0} ms for eight pictures, ten steps each", m.runs, m.ms)}</span>
        </div>
    }
}

#[function_component(App)]
pub fn app() -> Html {
    let model = use_reducer(Model::new);
    let pick = {
        let d = model.dispatcher();
        Callback::from(move |(r, _c): (usize, usize)| d.dispatch(Action::Pick(r / (SIDE * CELL + GAP))))
    };
    let body = match &model.journey {
        Some(j) => {
            let counts: Vec<String> = (0..PICTURES).map(|i| on(j.picture(10, i)).to_string()).collect();
            let this = on(j.picture(10, model.pick));
            html! {
                <div class="layout even">
                    <div class="col">
                        { panel("From noise to a shape:", "(_, _, L) := 10 'u:s_tep p_ower (E, 10, E)", "Each row is one picture: pure noise at the left, then the ten steps, the shape at the right. Every step the denoiser says what clean picture it sees, the noise that would make this one of it is inferred, and the picture steps down one level. No new noise is added, so the same noise always gives the same picture. Click a row.", true, html! { <>
                            <p class="calc">{ format!("pixels on in the eight shapes: {}", counts.join(", ")) }</p>
                            <div class="padbox"><Canvas rows={JH} cols={JW} rgba={Rc::new(journey(j, model.pick))} onclick={pick} class="pad" /></div>
                        </> }) }
                        { panel("One picture, level by level:", "C := t u:c_lean X", "The picked picture at every level, large. A disk, a hollow square or a cross should be there by the end; where it is not, the denoiser was torn between two shapes and averaged them. Its convolution sees each pixel's neighbors, which the earlier dense denoiser could not, so squares come out as squares more often.", false, html! { <>
                            <p class="calc">{ format!("picture {}: {this} pixels on", model.pick + 1) }</p>
                            <div class="padbox"><Canvas rows={BH} cols={BW} rgba={Rc::new(large(j, model.pick))} class="pad" /></div>
                        </> }) }
                    </div>
                    <div class="col">
                        <section class="panel code">
                            <h2>{"The program"}</h2>
                            <p class="note">{"Everything the page runs, in your browser: the head of diffusion.xtl (the schedule, the denoiser from data/, the step, the sampler), then each run's lines; the ten steps are highlighted."}</p>
                            { source() }
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
            { header("Learned denoiser", "Pictures from noise, step by step: a small convolutional network trained to see the clean picture in a noisy one, at ten levels of noise, run backward from pure noise. Eight pictures at a time, every level of the way shown; the shapes it learned are a disk, a hollow square and a cross.") }
            { controls(&model) }
            { notice(&model.notice) }
        </header>
        <main>{body}</main>
        { footer() }
        </>
    }
}
