//! The page: a digit to draw or pick, the filters, the feature maps and
//! pooled maps (click any feature-map cell to see its patch times its
//! filter), the probabilities, and the program.

use std::rc::Rc;

use web_sys::Element;
use yew::prelude::*;

use microscope::canvas::Canvas;
use microscope::chrome::{chip, footer, header, notice, panel};
use microscope::colour::{field, scaled, signed};
use microscope::source::code;

use crate::micro::{Anatomy, Input};
use crate::model::{Action, Model};
use crate::view::{source, Stage, STAGES};

fn act(m: &UseReducerHandle<Model>, a: impl Fn() -> Action + 'static) -> Callback<MouseEvent> {
    let d = m.dispatcher();
    Callback::from(move |_| d.dispatch(a()))
}

/// A number with three decimals (X_eTaL's own values are exact; this is
/// only how the page prints them).
fn num(x: f64) -> String {
    format!("{x:.3}")
}

fn stage_chip(m: &UseReducerHandle<Model>, s: Stage) -> Html {
    let (name, src, dims, meaning): (&str, &str, Vec<usize>, &str) = match s {
        Stage::Digit => ("digit", "x", vec![28, 28], "the picture, 0 (dark) to 1 (ink)"),
        Stage::Conv => ("convolution", "nn:r_elu u:c_onv x", vec![8, 26, 26], "8 filters over every 3 x 3 window, then ReLU"),
        Stage::Pool => ("pooling", "u:p_ool c", vec![8, 13, 13], "the largest of each 2 x 2 block"),
        Stage::Classify => ("classify", "u:c_lassify x", vec![10], "dense layer and softmax: a probability per digit"),
    };
    chip(name, src, &dims, meaning, m.focus == s, act(m, move || Action::Focus(s)))
}

fn controls(m: &UseReducerHandle<Model>) -> Html {
    let picked = match m.input {
        Input::Sample(d) => Some(d),
        Input::Drawn(_) => None,
    };
    html! {
        <div class="controls">
            <span>{"Test digits: "}</span>
            { for (0..10).map(|d| html! {
                <button class={classes!((picked == Some(d)).then_some("on"))} onclick={act(m, move || Action::Sample(d))}>{d}</button>
            }) }
            <button onclick={act(m, || Action::Clear)}>{"Clear and draw"}</button>
            <span class="gen">{format!("X_eTaL ran the network in {:.0} ms", m.ms)}</span>
        </div>
    }
}

/// The pad: the digit as X_eTaL last read it, or the strokes being
/// drawn; drawing runs the network when the stroke ends.
fn pad(m: &UseReducerHandle<Model>, node: &NodeRef, drawing: &UseStateHandle<bool>, last: &std::rc::Rc<std::cell::Cell<Option<(usize, usize)>>>) -> Html {
    let cell = {
        let node = node.clone();
        move |e: &PointerEvent| -> Option<(usize, usize)> {
            let r = node.cast::<Element>()?.get_bounding_client_rect();
            let x = (e.client_x() as f64 - r.left()) / r.width();
            let y = (e.client_y() as f64 - r.top()) / r.height();
            ((0.0..1.0).contains(&x) && (0.0..1.0).contains(&y)).then(|| ((y * 28.0) as usize, (x * 28.0) as usize))
        }
    };
    let down = {
        let (d, drawing, cell, last) = (m.dispatcher(), drawing.clone(), cell.clone(), last.clone());
        Callback::from(move |e: PointerEvent| {
            if let Some(el) = e.target_dyn_into::<Element>() {
                let _ = el.set_pointer_capture(e.pointer_id());
            }
            drawing.set(true);
            last.set(cell(&e));
            if let Some(p) = cell(&e) {
                d.dispatch(Action::Paint(p, p));
            }
        })
    };
    let moved = {
        let (d, drawing, cell, last) = (m.dispatcher(), drawing.clone(), cell.clone(), last.clone());
        Callback::from(move |e: PointerEvent| {
            if *drawing.clone() {
                if let Some(p) = cell(&e) {
                    d.dispatch(Action::Paint(last.get().unwrap_or(p), p));
                    last.set(Some(p));
                }
            }
        })
    };
    let up = {
        let (d, drawing) = (m.dispatcher(), drawing.clone());
        Callback::from(move |_: PointerEvent| {
            if *drawing.clone() {
                drawing.set(false);
                d.dispatch(Action::Done);
            }
        })
    };
    // The inspected patch is marked once the pad is the digit X_eTaL ran.
    let (_, r, c) = m.sel;
    let mark = m.anatomy.as_ref().filter(|a| a.x == m.pad).map(|_| (r + 1, c + 1));
    html! {
        <div ref={node.clone()} class="padbox" onpointerdown={down} onpointermove={moved} onpointerup={up.clone()} onpointercancel={up}>
            <Canvas rows={28} cols={28} rgba={Rc::new(field(&m.pad, 0.0, 1.0))} {mark} class="pad" />
        </div>
    }
}

/// Eight small images side by side (filters, maps).
fn row_of(n: usize, rows: usize, cols: usize, values: &[f64], colour: fn(&[f64]) -> Vec<u8>, mark: Option<(usize, usize, usize)>, pick: Option<Callback<(usize, usize, usize)>>, class: &'static str) -> Html {
    let size = rows * cols;
    html! {
        <div class={classes!("maps8", class)}>
            { for (0..n).map(|f| {
                let rgba = Rc::new(colour(&values[f * size..(f + 1) * size]));
                let onclick = pick.clone().map(|cb| Callback::from(move |(r, c): (usize, usize)| cb.emit((f, r, c))));
                let mark = mark.and_then(|(g, r, c)| (g == f).then_some((r, c)));
                html! { <figure>
                    <Canvas rows={rows} cols={cols} {rgba} {mark} {onclick} />
                    <figcaption>{format!("filter {}", f + 1)}</figcaption>
                </figure> }
            }) }
        </div>
    }
}

fn conv(m: &UseReducerHandle<Model>, a: &Anatomy) -> Html {
    let d = m.dispatcher();
    let pick = Some(Callback::from(move |(f, r, c): (usize, usize, usize)| d.dispatch(Action::Select(f, r, c))));
    let body = html! { <>
        <p class="note">{"The 8 filters (3 x 3; blue negative, red positive):"}</p>
        { row_of(8, 3, 3, &a.k, signed, None, None, "kerns") }
        <p class="note">{"What each finds over the digit, after ReLU (click any cell to see how it was computed):"}</p>
        { row_of(8, 26, 26, &a.c, scaled, Some(m.sel), pick, "fmaps") }
    </> };
    panel("2. Convolution and ReLU:", "c := nn:r_elu u:c_onv x",
        "The digit's nine shifted copies (every 3 x 3 window at once) times the 8 filters: one matrix product, then the biases, then ReLU keeps what is positive.",
        m.focus == Stage::Conv, body)
}

fn pool(m: &UseReducerHandle<Model>, a: &Anatomy) -> Html {
    let (f, r, c) = m.sel;
    panel("3. Max-pooling:", "m := u:p_ool c",
        "Each map reshaped into 2 x 2 blocks and the largest of each kept: 8 maps of 13 x 13, the 1352 numbers the dense layer reads.",
        m.focus == Stage::Pool, row_of(8, 13, 13, &a.m, scaled, Some((f, r / 2, c / 2)), None, "pmaps"))
}

fn probabilities(m: &UseReducerHandle<Model>, a: &Anatomy) -> Html {
    let best = a.digit();
    let bars = (0..10).map(|d| {
        let h = format!("height: {:.1}%", 100.0 * a.p[d]);
        html! { <div class={classes!("bar", (d == best).then_some("best"))}>
            <span>{format!("{:.0}%", 100.0 * a.p[d])}</span>
            <div class="fill" style={h}></div>
            <b>{d}</b>
        </div> }
    });
    let body = html! { <>
        <div class="bars digits">{ for bars }</div>
        <p class="calc">{"The network reads a "}<b>{best}</b>{format!(", {:.1}% sure.", 100.0 * a.p[best])}</p>
    </> };
    panel("4. Dense layer and softmax:", "p := u:c_lassify x",
        "The 1352 pooled values times the dense weights, plus the biases, then softmax: ten probabilities that add up to 1.",
        m.focus == Stage::Classify, body)
}

fn inspector(m: &UseReducerHandle<Model>, a: &Anatomy) -> Html {
    let (f, r, c) = m.sel;
    let p = a.patch(f, r, c);
    let grid = |v: &[f64; 9]| html! {
        <table class="zs patch">{ for (0..3).map(|i| html! { <tr>{ for (0..3).map(|j| html! { <td>{num(v[3 * i + j])}</td> }) }</tr> }) }</table>
    };
    let x = a.c_at(f, r, c);
    html! { <section class="panel">
        <h2>{format!("The patch: filter {}, row {}, column {}", f + 1, r + 1, c + 1)}</h2>
        <p class="note">{"The 3 x 3 pixels of the digit under this output (marked on the digit), times the filter, item by item:"}</p>
        <div class="patchrow">{ grid(&p.pixels) }<span class="op">{"\u{00d7}"}</span>{ grid(&p.weights) }</div>
        <p class="calc">{"sum of the products "}{code(&num(p.sum))}{" + bias "}{code(&num(p.bias))}{" = "}{code(&num(p.sum + p.bias))}{", ReLU keeps "}{code(&num(p.value))}</p>
        <p class="calc">{"X_eTaL's value in the map: "}{code(&num(x))}</p>
        { for ((x - p.value).abs() > 1e-9).then(|| html! { <p class="error">{"the page's sum differs from X_eTaL's"}</p> }) }
    </section> }
}

/// The last pad cell of the stroke being drawn, kept across renders.
#[hook]
fn use_mut_ref_cell() -> std::rc::Rc<std::cell::Cell<Option<(usize, usize)>>> {
    (*use_state(|| std::rc::Rc::new(std::cell::Cell::new(None)))).clone()
}

#[function_component(App)]
pub fn app() -> Html {
    let model = use_reducer(Model::new);
    let node = use_node_ref();
    let drawing = use_state(|| false);
    let last = use_mut_ref_cell();
    let body = match &model.anatomy {
        Some(a) => html! { <>
            <div class="layout">
                <div class="col">
                    { panel("1. The digit:", "x", "Pick a test digit, or clear the pad and draw one (large and centred, like MNIST's); the network runs when you lift the pen.", model.focus == Stage::Digit, pad(&model, &node, &drawing, &last)) }
                    { probabilities(&model, a) }
                    { inspector(&model, a) }
                </div>
                <div class="col">{ conv(&model, a) }{ pool(&model, a) }</div>
            </div>
            <section class="panel code">
                <h2>{"The program"}</h2>
                <p class="note">{"Everything the page runs, in your browser: the network read from data/ and the core of cnn-digits.xtl, then this run's lines; the stage you pick is highlighted."}</p>
                { source(&model.input, model.focus) }
            </section>
        </> },
        None => html! {},
    };
    html! {
        <>
        <header>
            { header("Tiny CNN", "A tiny convolutional network, trained on MNIST, reads a handwritten digit. X_eTaL runs every stage in your browser: 8 filters over the digit (one matrix product), ReLU, 2 x 2 max-pooling, a dense layer and softmax. Draw a digit, or pick one, and click any feature map to see the arithmetic.") }
            <nav class="timeline">{ for STAGES.iter().map(|&s| stage_chip(&model, s)) }</nav>
            { controls(&model) }
            { notice(&model.notice) }
        </header>
        <main>{body}</main>
        { footer() }
        </>
    }
}
