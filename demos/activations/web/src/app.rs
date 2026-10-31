//! The page: a family of activation functions picked, its two charts
//! (the functions, their slopes) drawn by X_eTaL as the parameters'
//! sliders move; then an activation of the reader's own, edited, run
//! and drawn beside a built-in; and every program the page ran.

use std::rc::Rc;

use gloo_timers::callback::Timeout;
use web_sys::{HtmlInputElement, HtmlSelectElement, HtmlTextAreaElement};
use yew::prelude::*;

use microscope::chrome::{footer, header, panel, picture};
use microscope::source::{block, NONE};

use crate::micro::{program, run, run_yours, yours, Drawn, Mine, Params, BUILTINS, FAMILIES, YOURS};

/// A slider: its label, its value, its range and step, and what moving
/// it (when it is let go) sets.
fn slider(label: &str, value: f64, (lo, hi, step): (f64, f64, f64), set: Callback<f64>) -> Html {
    let change = Callback::from(move |e: Event| {
        let i: HtmlInputElement = e.target_unchecked_into();
        if let Ok(v) = i.value().parse::<f64>() {
            set.emit(v);
        }
    });
    html! { <label class="slider">{format!("{label} {value:.2}")}
        <input type="range" min={lo.to_string()} max={hi.to_string()} step={step.to_string()} value={value.to_string()} onchange={change} />
    </label> }
}

#[function_component(App)]
pub fn app() -> Html {
    let family = use_state(|| 0usize);
    let params = use_state(Params::default);
    // A family's run, once per family and parameters, after the spinner
    // is drawn.
    let drawn = use_state(|| None::<((usize, Params), Rc<Result<Drawn, String>>)>);
    {
        let drawn = drawn.clone();
        use_effect_with((*family, *params), move |&key| {
            let t = Timeout::new(40, move || drawn.set(Some((key, Rc::new(run(key.0, key.1))))));
            move || drop(t)
        });
    }
    let text = use_state(|| YOURS.to_string());
    let builtin = use_state(|| 0usize);
    let mine = use_state(|| None::<Result<Mine, String>>);
    let ran = use_state(String::new);

    let fam = FAMILIES[*family];
    let picker = html! { <div class="controls"><span>{"A family: "}</span>
        { for FAMILIES.iter().enumerate().map(|(i, f)| {
            let s = family.clone();
            html! { <button class={classes!((*family == i).then_some("on"))} onclick={Callback::from(move |_| s.set(i))}>{f.title}</button> }
        }) }
    </div> };
    let p = *params;
    let set = |f: fn(Params, f64) -> Params| {
        let params = params.clone();
        Callback::from(move |v: f64| params.set(f(*params, v)))
    };
    let sliders = html! { <div class="controls sliders">
        { slider("leaky ReLU's slope", p.leak, (0.0, 0.5, 0.01), set(|p, v| Params { leak: v, ..p })) }
        { slider("ELU's alpha", p.alpha, (0.1, 3.0, 0.05), set(|p, v| Params { alpha: v, ..p })) }
        { slider("Swish's beta", p.beta, (0.1, 5.0, 0.05), set(|p, v| Params { beta: v, ..p })) }
    </div> };
    let charts = match drawn.as_ref().filter(|(k, _)| *k == (*family, p)).map(|(_, r)| r.clone()) {
        None => html! { <div class="running"><span class="spinner"></span>{format!("X_eTaL is drawing the {} family in your browser\u{2026}", fam.title.to_lowercase())}</div> },
        Some(r) => match r.as_ref() {
            Ok(d) => html! { <div class="pair">
                { picture(&d.charts[0], &format!("{}: the functions", fam.title)) }
                { picture(&d.charts[1], &format!("{}: their slopes", fam.title)) }
            </div> },
            Err(e) => html! { <pre class="err">{ e.clone() }</pre> },
        },
    };

    let edit = {
        let text = text.clone();
        Callback::from(move |e: InputEvent| {
            let a: HtmlTextAreaElement = e.target_unchecked_into();
            text.set(a.value());
        })
    };
    let choose = {
        let builtin = builtin.clone();
        Callback::from(move |e: Event| {
            let s: HtmlSelectElement = e.target_unchecked_into();
            builtin.set(s.selected_index().max(0) as usize);
        })
    };
    let go = {
        let (text, builtin, mine, ran) = (text.clone(), builtin.clone(), mine.clone(), ran.clone());
        Callback::from(move |_| {
            mine.set(Some(run_yours(&text, *builtin)));
            ran.set(yours(&text, *builtin));
        })
    };
    let gave = match mine.as_ref() {
        None => html! {},
        Some(Err(e)) => html! { <pre class="err">{ e.clone() }</pre> },
        Some(Ok(m)) if m.charts.len() == 2 => html! { <div class="pair">
            { picture(&m.charts[0], "your activation beside a built-in") }
            { picture(&m.charts[1], "their slopes") }
        </div> },
        Some(Ok(m)) => html! { <pre class="err">{ format!("Not drawn: {}. Plot draws finite values only.", m.status) }</pre> },
    };
    let rows = (text.lines().count() + 1).max(4).to_string();

    html! {
        <>
        <header>
            { header("Activation functions", "Every activation of this repository's Activations library and its slope, over x from -6 to 6, drawn by X_eTaL in your browser with X_eTaL-libraries' Plot (an x axis of real values). Move a slider to change a parameter; write an activation of your own and see it beside a built-in.") }
            { picker }
        </header>
        <main>
            { panel(fam.title, "p:f_unctionChart!", "The functions, then their slopes (Activations' derivatives; the gates' by central differences), 241 values from -6 to 6 each.", false, html! { <>{ sliders }{ charts }</> }) }
            { panel("Your own", "u:m_ine", "An activation of your own: a function of one number x, named u:m_ine, using any of Activations' functions (act:). Run draws it, applied one x at a time, beside a built-in, with its slope by central differences; an error is placed in your text.", false, html! { <>
                <textarea class="editor" spellcheck="false" value={(*text).clone()} oninput={edit} {rows} />
                <div class="controls">
                    <span>{"beside "}</span>
                    <select onchange={choose}>{ for BUILTINS.iter().enumerate().map(|(i, b)| html! { <option selected={i == *builtin}>{b.0}</option> }) }</select>
                    <button class="run" onclick={go}>{"Run"}</button>
                </div>
                { gave }
                if !ran.is_empty() { <details><summary>{"The program it ran"}</summary>{ block(&ran, NONE) }</details> }
            </> }) }
            <section class="panel code">
                <h2>{"The program"}</h2>
                <p class="note">{"Everything the page runs for this family, in your browser: activations.xtl's head (the imports, the x values, the parameters as the sliders set them, the helpers), the family's section, then its curves printed for the page to read back."}</p>
                { block(&program(*family, p), NONE) }
            </section>
        </main>
        { footer() }
        </>
    }
}
