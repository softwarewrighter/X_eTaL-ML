//! The page: the four formats side by side (map, accuracy, agreement,
//! error, storage, operations), the ternary weights as glyphs, one
//! output of layer 2 as additions and subtractions, the program.

use std::rc::Rc;

use web_sys::{HtmlInputElement, HtmlSelectElement};
use yew::prelude::*;

use microscope::canvas::Canvas;
use microscope::chrome::{chip, footer, header, notice, panel};
use microscope::source::code;

use crate::micro::{at, Formats, Point, Ternary, BITS, FORMATS, OUTPUTS, SETS, SIDE, WEIGHTS};
use crate::model::{Action, Model};
use crate::view::{cell, map, source, Stage, PIC, STAGES};

fn act(m: &UseReducerHandle<Model>, a: impl Fn() -> Action + 'static) -> Callback<MouseEvent> {
    let d = m.dispatcher();
    Callback::from(move |_| d.dispatch(a()))
}

/// A number as a plain decimal with about four significant digits.
fn num(x: f64) -> String {
    if x == 0.0 || !x.is_finite() {
        return "0.0".into();
    }
    let digits = (3 - x.abs().log10().floor() as i32).clamp(1, 8) as usize;
    format!("{x:.digits$}")
}

fn stage_chip(m: &UseReducerHandle<Model>, s: Stage) -> Html {
    let n = SIDE * SIDE;
    let (name, src, dims, meaning): (&str, &str, Vec<usize>, &str) = match s {
        Stage::Inputs => ("inputs", "grid", vec![n, 16], "every map point as a row of 16: x, y, then 0s"),
        Stage::Network => ("network", "grid u:n_et m32", vec![n, 3], "three outputs per point, one per arm"),
        Stage::Formats => ("formats", "t u:t_ern w", vec![16, 3, 16], "the ternary weights: input, layer, output"),
        Stage::Additions => ("additions", "h u:a_dds k", vec![16], "layer 2's outputs before the scale: sums and differences of inputs"),
        Stage::Maps => ("maps", "y32 ; y16 ; y8 ; y2", vec![n, 3], "each format's outputs over the map"),
        Stage::Measures => ("measures", "k u:m_easures y", vec![3], "accuracy, agreement with FP32, mean output error"),
    };
    chip(name, src, &dims, meaning, m.focus == s, act(m, move || Action::Focus(s)))
}

fn controls(m: &UseReducerHandle<Model>) -> Html {
    let d = m.dispatcher();
    let on_set = Callback::from(move |e: Event| {
        let i = e.target_unchecked_into::<HtmlSelectElement>().selected_index();
        d.dispatch(Action::Set(i.max(0) as usize));
    });
    let d = m.dispatcher();
    let on_t = Callback::from(move |e: Event| {
        let v = e.target_unchecked_into::<HtmlInputElement>().value();
        d.dispatch(Action::Thresh(v.parse().unwrap_or(0.5)));
    });
    let (a, b, c) = m.ms;
    html! {
        <div class="controls">
            <select onchange={on_set} aria-label="Weights">
                { for SETS.iter().enumerate().map(|(i, s)| html! { <option selected={i == m.setup.set}>{s.0}</option> }) }
            </select>
            <label class="slider">{"ternary threshold "}{code("t")}
                <input type="range" min="0" max="1.5" step="0.05" value={m.setup.t.to_string()} onchange={on_t} />
                <b>{format!("{:.2}", m.setup.t)}</b>
            </label>
            <span class="gen">{format!("X_eTaL: formats {a:.0} ms, ternary {b:.0} ms, the point {c:.0} ms")}</span>
        </div>
    }
}

fn card(model: &UseReducerHandle<Model>, k: usize, arms: &[f64], meas: [f64; 3], f: &Formats, t: &Ternary) -> Html {
    let d = model.dispatcher();
    let onclick = Some(Callback::from(move |(r, c): (usize, usize)| {
        let (x, y) = at(r, c, PIC);
        d.dispatch(Action::Point(x, y));
    }));
    let (x, y) = model.setup.point;
    let rgba = Rc::new(map(arms, &f.px, &f.py, &f.pl));
    let bits = BITS[k] * WEIGHTS as f64;
    let ops = match k {
        3 => format!("{} additions and subtractions, {OUTPUTS} multiplies (one scale per output)", t.q.iter().filter(|&&v| v != 0.0).count()),
        2 => format!("{WEIGHTS} 8-bit integer multiply-adds"),
        _ => format!("{WEIGHTS} multiply-adds"),
    };
    html! {
        <section class={classes!("panel", "format", (k == 3).then_some("focus"))}>
            <h2>{FORMATS[k]}</h2>
            <Canvas rows={PIC} cols={PIC} {rgba} mark={Some(cell(x, y))} {onclick} class="mapc" />
            <table class="zs stats">
                <tr><td>{"test points right"}</td><td>{format!("{:.1}%", 100.0 * meas[0])}</td></tr>
                <tr><td>{"map agrees with FP32"}</td><td>{format!("{:.1}%", 100.0 * meas[1])}</td></tr>
                <tr><td>{"mean output error"}</td><td>{num(meas[2])}</td></tr>
                <tr><td>{"weight storage"}</td><td>{format!("{:.0} bits", bits)}</td></tr>
            </table>
            <p class="note">{format!("Per point: {ops}.")}</p>
        </section>
    }
}

fn formats(model: &UseReducerHandle<Model>, f: &Formats, t: &Ternary) -> Html {
    html! {
        <div class="formats">
            { for (0..3).map(|k| card(model, k, &f.maps[k], f.measures[k], f, t)) }
            { card(model, 3, &t.map, t.measures, f, t) }
        </div>
    }
}

fn glyph(v: f64) -> (&'static str, &'static str) {
    match v {
        v if v > 0.0 => ("+", "plus"),
        v if v < 0.0 => ("\u{2212}", "minus"),
        _ => ("\u{00b7}", "zero"),
    }
}

/// A layer's ternary weights as glyphs: rows are inputs, columns outputs.
fn glyphs(model: &UseReducerHandle<Model>, t: &Ternary, l: usize, ni: usize, no: usize) -> Html {
    let unit = model.unit;
    let rows = (0..ni).map(|i| {
        let cells = (0..no).map(|o| {
            let (g, c) = glyph(t.q[i * 48 + l * 16 + o]);
            let d = model.dispatcher();
            let onclick = (l == 1).then(|| Callback::from(move |_: MouseEvent| d.dispatch(Action::Unit(o))));
            html! { <td class={classes!(c, (l == 1 && o == unit).then_some("sel"))} {onclick}>{g}</td> }
        });
        html! { <tr>{ for cells }</tr> }
    });
    let kept = (0..ni).flat_map(|i| (0..no).map(move |o| (i, o))).filter(|&(i, o)| t.q[i * 48 + l * 16 + o] != 0.0).count();
    html! { <figure class="glyphfig">
        <table class={classes!("glyphs", (l == 1).then_some("pick"))}>{ for rows }</table>
        <figcaption>{code(&format!("{} s_elect_2 q", l + 1))}{format!(": {ni} x {no}, {kept} kept, scale {}", num(t.scales[l]))}</figcaption>
    </figure> }
}

fn weights(model: &UseReducerHandle<Model>, t: &Ternary) -> Html {
    let body = html! { <div class="glyphrow">
        { glyphs(model, t, 0, 2, 16) }{ glyphs(model, t, 1, 16, 16) }{ glyphs(model, t, 2, 16, 3) }
    </div> };
    panel("The ternary weights:", "q := t u:t_ern w",
        "Each layer's weights become +1, 0 or \u{2212}1: kept where the size passes t times the layer's mean size, and one scale per layer (the mean size of the kept ones). Click a column of layer 2 to follow that output.",
        model.focus == Stage::Formats, body)
}

fn additions(model: &UseReducerHandle<Model>, t: &Ternary, p: &Point) -> Html {
    let j = model.unit;
    let q = |i: usize| t.q[i * 48 + 16 + j];
    let terms = (0..16).filter(|&i| q(i) != 0.0).map(|i| {
        let (g, c) = glyph(q(i));
        html! { <span class={classes!("term", c)}>{g}{code(&num(p.h1[i]))}</span> }
    });
    let plus: f64 = (0..16).filter(|&i| q(i) > 0.0).map(|i| p.h1[i]).sum();
    let minus: f64 = (0..16).filter(|&i| q(i) < 0.0).map(|i| p.h1[i]).sum();
    let (x, y) = model.setup.point;
    let line = format!("({} * {}) + {}", num(t.scales[1]), num(p.adds[j]), num(p.b2[j]));
    let body = html! { <>
        <p class="calc">{format!("The point ({}, {}): layer 1 gives 16 numbers ", num(x), num(y))}{code("h1")}{"; output "}{j + 1}{" of layer 2 adds those where its column has + and subtracts those with \u{2212}:"}</p>
        <p class="terms">{ for terms }</p>
        <p class="calc">{"+ terms "}{code(&num(plus))}{", \u{2212} terms "}{code(&num(minus))}{": "}{code("(f_irst h1) u:a_dds 2 s_elect_2 q")}{" = "}{code(&num(p.adds[j]))}</p>
        <p class="calc">{"then one multiply by the scale, plus the bias: "}{code(&line)}{" = "}{code(&num(t.scales[1] * p.adds[j] + p.b2[j]))}</p>
        <p class="calc">{"the matrix product gives the same: "}{code("h1 u:l_ayer 2 s_elect_2 m2")}{" = "}{code(&num(p.pre[j]))}</p>
        { for ((plus - minus - p.adds[j]).abs() > 1e-9).then(|| html! { <p class="error">{"the page's sum differs from X_eTaL's"}</p> }) }
    </> };
    panel("Additions only:", "u:a_dds", "No multiplications inside a ternary layer: each output is a sum of some inputs minus a sum of others.", model.focus == Stage::Additions, body)
}

fn outputs(model: &UseReducerHandle<Model>, p: &Point) -> Html {
    let rows = (0..4).map(|k| {
        let o = p.outs[k];
        let arm = (0..3).fold(0, |b, i| if o[i] > o[b] { i } else { b });
        html! { <tr>
            <td>{FORMATS[k]}</td>
            { for (0..3).map(|i| html! { <td class={classes!((i == arm).then_some("best"))}>{code(&num(o[i]))}</td> }) }
            <td>{["a", "b", "c"][arm]}</td>
        </tr> }
    });
    let (x, y) = model.setup.point;
    html! { <section class="panel">
        <h2>{format!("The point ({}, {}) in each format", num(x), num(y))}</h2>
        <p class="note">{"Its three outputs, one per arm; the largest wins. Click any map to pick another point."}</p>
        <table class="zs outs"><tr><th>{"format"}</th><th>{"arm a"}</th><th>{"arm b"}</th><th>{"arm c"}</th><th>{"chosen"}</th></tr>{ for rows }</table>
    </section> }
}

#[function_component(App)]
pub fn app() -> Html {
    let model = use_reducer(Model::new);
    let body = match (&model.formats, &model.ternary, &model.point) {
        (Some(f), Some(t), Some(p)) => html! { <>
            { formats(&model, f, t) }
            <div class="layout even">
                <div class="col">{ weights(&model, t) }{ additions(&model, t, p) }{ outputs(&model, p) }</div>
                <div class="col">
                    <section class="panel code">
                        <h2>{"The program"}</h2>
                        <p class="note">{"The page's settings, then the core of ternary-net.xtl, run by X_eTaL in your browser; the stage you pick is highlighted."}</p>
                        { source(&model.setup, model.focus) }
                    </section>
                </div>
            </div>
        </> },
        _ => html! {},
    };
    html! {
        <>
        <header>
            { header("1.58-bit network", "One tiny network (which of three spiral arms is a point on? 2 \u{2192} 16 \u{2192} 16 \u{2192} 3) run by X_eTaL with its weights stored four ways: 32-bit and 16-bit floats, 8-bit integers, and ternary \u{2212}1 / 0 / +1 (1.58 bits each), where a layer needs only additions. Compare the maps: the network trained in full precision fails as ternary; switch to the one trained for ternary, and move the threshold.") }
            <nav class="timeline">{ for STAGES.iter().map(|&s| stage_chip(&model, s)) }</nav>
            { controls(&model) }
            { notice(&model.notice) }
        </header>
        <main>{body}</main>
        { footer() }
        </>
    }
}
