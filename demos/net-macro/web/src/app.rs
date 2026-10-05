//! The page: a spec to pick or type, the line that calls the macro, the
//! function the macro wrote, the parameter count, and for the program's
//! trained networks what they decide over the plane.

use std::rc::Rc;

use web_sys::HtmlInputElement;
use yew::prelude::*;

use microscope::canvas::Canvas;
use microscope::chrome::{footer, header, panel};
use microscope::source::code;

use crate::micro::{networks, weight_names, SIDE};
use crate::model::{Action, Model, Shown};
use crate::view::{map, source};

fn controls(m: &UseReducerHandle<Model>) -> Html {
    let d = m.dispatcher();
    let on_text = Callback::from(move |e: Event| d.dispatch(Action::Spec(e.target_unchecked_into::<HtmlInputElement>().value())));
    html! {
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
    }
}

fn call(m: &UseReducerHandle<Model>) -> Html {
    let line = match &m.shown {
        Shown::Trained(n, _) => format!("u:{} := \"{}\" net:n_etwork< \"{}\"", n.name, n.spec, n.weights),
        _ => format!("u:n_et := \"{}\" net:n_etwork< \"{}\"", m.spec, weight_names(&m.spec)),
    };
    panel("1. The network, one line:", "net:n_etwork<",
        "A spec is sizes and activations, the input's size first; on the right, one weight array per dense layer. The macro reads this text when the program is compiled.",
        false, html! { <p class="calc">{code(&line)}</p> })
}

fn wrote(m: &UseReducerHandle<Model>) -> Html {
    let body = match &m.shown {
        Shown::Trained(_, r) => html! { <>
            <p class="calc">{code(&r.expansion)}</p>
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
        "Ordinary X_eTaL, type-checked like any code: each number after the first became a dense layer, each word an activation of the NN library. The notation adds nothing the code does not say.",
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

#[function_component(App)]
pub fn app() -> Html {
    let model = use_reducer(Model::new);
    let picked = match &model.shown {
        Shown::Trained(n, _) => Some(n.clone()),
        _ => None,
    };
    html! {
        <>
        <header>
            { header("Network macro", "A network written as one line. X_eTaL's macro libraries extend the language itself: the Net library reads a spec such as \u{201c}2 16 relu 16 relu 3 softmax\u{201d} when the program is compiled and writes the forward function in its place, as ordinary code you can read. Pick a network, or type a spec of your own, and see what the macro wrote.") }
            { controls(&model) }
        </header>
        <main>
            <div class="layout even">
                <div class="col">{ call(&model) }{ wrote(&model) }{ computes(&model) }</div>
                <div class="col">
                    <section class="panel code">
                        <h2>{"The program"}</h2>
                        <p class="note">{"Everything the page runs, in your browser: for a network of net-macro.xtl, the program's head (the data and the three networks) and this run's lines; for a spec you typed, the two small programs. The line that calls the macro is highlighted."}</p>
                        { source(picked.as_ref(), &model.spec) }
                    </section>
                </div>
            </div>
        </main>
        { footer() }
        </>
    }
}
