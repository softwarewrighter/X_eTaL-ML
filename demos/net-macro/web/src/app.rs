//! The page: a spec to pick or type, the line that calls the macro, the
//! function the macro wrote, the parameter count, and for the program's
//! trained networks what they decide over the plane; and any spec from
//! 2 inputs to 3 softmax trained in the page, from random weights, by
//! the step the macro writes.

use std::rc::Rc;

use gloo_timers::callback::Timeout;
use web_sys::HtmlInputElement;
use yew::prelude::*;

use microscope::canvas::Canvas;
use microscope::chrome::{footer, header, notice, panel, picture};
use microscope::source::{block, code, NONE};

use crate::micro::{networks, trainable, weight_names, SIDE};
use crate::model::{Action, Model, Shown, CHUNK, LIMIT};
use crate::view::{map, source, train_source};

fn controls(m: &UseReducerHandle<Model>) -> Html {
    let d = m.dispatcher();
    let on_text = Callback::from(move |e: Event| d.dispatch(Action::Spec(e.target_unchecked_into::<HtmlInputElement>().value())));
    html! { <>
        <div class="controls">
            <span>{"A network: "}</span>
            { for networks().into_iter().map(|n| {
                let d = m.dispatcher();
                let spec = n.spec.clone();
                html! { <button class={classes!((m.spec == n.spec).then_some("on"))} onclick={Callback::from(move |_| d.dispatch(Action::Spec(spec.clone())))}>{n.spec.clone()}</button> }
            }) }
            <span>{" or type a spec: "}</span>
            <input class="sentence" type="text" value={m.spec.clone()} onchange={on_text} aria-label="Spec" />
            <span class="gen">{format!("X_eTaL: {:.0} ms", m.ms)}</span>
        </div>
        <div class="controls">
            { train_buttons(m) }
        </div>
    </> }
}

fn train_buttons(m: &UseReducerHandle<Model>) -> Html {
    let d = m.dispatcher();
    let train = Callback::from(move |_| d.dispatch(Action::Train));
    let can = trainable(&m.spec).is_ok();
    match &m.training {
        None => html! { <>
            <button class="on" onclick={train} disabled={!can}>{"Train it"}</button>
            <span class="gen">{ if can { "from random weights, in your browser" } else { "a spec from 2 inputs to 3 softmax trains here" } }</span>
        </> },
        Some(t) => {
            let d = m.dispatcher();
            let playing = t.playing;
            let play = Callback::from(move |_| d.dispatch(Action::Play(!playing)));
            html! { <>
                <button class="on" onclick={play} disabled={t.steps >= LIMIT}>{ if playing { "Pause" } else { "Go on" } }</button>
                <button onclick={train}>{"Start again"}</button>
                <span class="gen">{format!("step {} of {LIMIT} \u{00b7} X_eTaL: {:.0} ms a step", t.steps, t.ms_per_step)}</span>
            </> }
        }
    }
}

fn call(m: &UseReducerHandle<Model>) -> Html {
    let line = match &m.shown {
        Shown::Trained(n, _) => n.line(),
        _ => format!("u:n_et := \"{}\" net:n_etwork< \"{}\"", m.spec, weight_names(&m.spec)),
    };
    panel("1. The network, one line:", "net:m_odel<",
        "A spec is sizes and activations, the input's size first. For a network of the program the macro also loads the weights, one file per dense layer, shaped as the spec says; a typed spec has no weights, so it uses net:n_etwork< with names. The macro reads this text when the program is compiled.",
        false, html! { <p class="calc">{code(&line)}</p> })
}

fn wrote(m: &UseReducerHandle<Model>) -> Html {
    let body = match &m.shown {
        Shown::Trained(_, r) => html! { <>
            { block(&r.expansion, NONE) }
            <p class="calc">{code(&format!("\"{}\" net:p_arams< @", m.spec))}{" became "}{code(&r.params.to_string())}{format!(": {} numbers to learn.", r.params)}</p>
        </> },
        Shown::Typed(r) => html! { <>
            <p class="calc">{code(&r.expansion)}</p>
            <p class="calc">{code(&format!("\"{}\" net:p_arams< @", m.spec))}{" became "}{code(&r.params.to_string())}{format!(": {} numbers to learn.", r.params)}</p>
        </> },
        Shown::Refused(e) => html! { <>
            <p class="error">{"The macro refused this spec when the program was compiled:"}</p>
            <p class="calc">{e.clone()}</p>
        </> },
    };
    panel("2. What the macro wrote:", "xetal expand",
        "Ordinary X_eTaL, type-checked like any code: each dense layer's weights read from its file and checked for the count the spec implies (the program stops, naming the layer, if a file is wrong), then the function: each number after the first a dense layer, each word an activation of NN. The notation adds nothing the code does not say.",
        true, body)
}

fn computes(m: &UseReducerHandle<Model>) -> Html {
    let body = match &m.shown {
        Shown::Trained(n, r) => html! { <>
            <div class="padbox"><Canvas rows={SIDE} cols={SIDE} rgba={Rc::new(map(&r.map, &r.points))} class="pad" /></div>
            <p class="calc">{code(&format!("arm nn:a_ccuracy u:{} xy", n.name))}{format!(" = {:.1}% of the test points right.", 100.0 * r.accuracy)}</p>
            <p class="note">{"The ground is the arm the network decides at each point of the plane; the dots are the 120 test points, in their own arm's color. Trained offline; the weights are read from data/."}</p>
        </> },
        Shown::Typed(_) => html! { <p class="note">{"This spec has no trained weights here, so there is nothing to run: pick one of the three networks above to see what a network decides."}</p> },
        Shown::Refused(_) => html! {},
    };
    panel("3. What it computes:", "u:d_eep grid", "Which of three spiral arms is a point on? A line cannot follow a spiral; a small network nearly can; the deep one does.", false, body)
}

fn trains(m: &UseReducerHandle<Model>) -> Html {
    let note = "From the same spec the macro writes the network's backward pass and an Adam step, then p_ower repeats it: the 300 points of a spiral made in X_eTaL, the weights small and random at the start. Each run is a few steps from the state the page holds.";
    let body = match (&m.training, trainable(&m.spec)) {
        (Some(t), _) => {
            let first = t.history.first().map_or(0.0, |h| h.1);
            html! { <>
                <div class="padbox"><Canvas rows={SIDE} cols={SIDE} rgba={Rc::new(map(&t.after.map, &t.after.points))} class="pad" /></div>
                <p class="calc">{format!("After {} steps: loss {:.3} (from {first:.3}), {:.1}% of the 300 points right.", t.steps, t.after.loss, 100.0 * t.after.right)}</p>
                { for t.after.pictures.iter().zip(["The loss after each run, stretched to its own range (losses):", "The share of points right after each run (right):"]).map(|(svg, what)| html! { <>
                    <p class="note">{what}</p>
                    { picture(svg, what) }
                </> }) }
                <p class="note">{"Drawn by the Plot library's p:l_ine!, in the program's last lines, from the second run on."}</p>
                <p class="calc">{code(&format!("\"u:s_tep X Y lr\" net:t_rain< \"{}\"", m.spec))}{" became:"}</p>
                { block(&t.step, NONE) }
            </> }
        }
        (None, Ok(_)) => html! { <p class="note">{"Press Train it to train this spec here."}</p> },
        (None, Err(why)) => html! { <p class="note">{why}</p> },
    };
    panel("4. Train it here:", "net:t_rain<", note, false, body)
}

#[function_component(App)]
pub fn app() -> Html {
    let model = use_reducer(Model::new);
    {
        // While training plays, run the next steps once this frame has drawn.
        let d = model.dispatcher();
        let (playing, steps) = model.training.as_ref().map_or((false, 0), |t| (t.playing, t.steps));
        use_effect_with((playing, steps), move |&(playing, _)| {
            let t = playing.then(|| Timeout::new(30, move || d.dispatch(Action::Tick)));
            move || drop(t)
        });
    }
    let picked = match &model.shown {
        Shown::Trained(n, _) => Some(n.clone()),
        _ => None,
    };
    html! {
        <>
        <header>
            { header("Network macro", "A network written as one line. X_eTaL's macro libraries extend the language itself: the Net library reads a spec such as \u{201c}2 16 relu 16 relu 3 softmax\u{201d} when the program is compiled and writes the forward function in its place, as ordinary code you can read. Pick a network, or type a spec of your own, and see what the macro wrote.") }
            { controls(&model) }
            { notice(&model.notice) }
        </header>
        <main>
            <div class="layout even">
                <div class="col">{ call(&model) }{ wrote(&model) }{ computes(&model) }{ trains(&model) }</div>
                <div class="col">
                    <section class="panel code">
                        <h2>{"The program"}</h2>
                        { match &model.training {
                            Some(t) => html! { <>
                                <p class="note">{"Everything the page runs to train, in your browser: the head of train-it.xtl for this spec (the spiral, the training step the macro writes, the starting state), then each run's lines. The line that writes the training step is highlighted."}</p>
                                { train_source(&model.spec, &t.sizes, CHUNK, t.history.len()) }
                            </> },
                            None => html! { <>
                                <p class="note">{"Everything the page runs, in your browser: for a network of net-macro.xtl, the program's head (the data and the three networks) and this run's lines; for a spec you typed, the two small programs. The line that calls the macro is highlighted."}</p>
                                { source(picked.as_ref(), &model.spec) }
                            </> },
                        } }
                    </section>
                </div>
            </div>
        </main>
        { footer() }
        </>
    }
}
