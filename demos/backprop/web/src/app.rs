//! The page: the batch, the forward pass, the loss, each layer's
//! gradient, the step, the check against finite differences, the loss
//! history, and the program.

use web_sys::HtmlInputElement;
use yew::prelude::*;

use microscope::chrome::{chip, footer, header, notice, panel, picture};
use microscope::color::{ramp, DIVERGE};
use microscope::source::code;

use crate::micro::{Step, I, J, K, N};
use crate::model::{Action, Model};
use crate::view::{source, Stage, STAGES};

fn act(m: &UseReducerHandle<Model>, a: impl Fn() -> Action + 'static) -> Callback<MouseEvent> {
    let d = m.dispatcher();
    Callback::from(move |_| d.dispatch(a()))
}

fn cell_style(v: f64, scale: f64) -> String {
    let c = ramp(&DIVERGE, 0.5 + v / (2.0 * scale.max(1e-12)));
    let light = 0.299 * c[0] as f64 + 0.587 * c[1] as f64 + 0.114 * c[2] as f64 > 140.0;
    format!("background: rgb({}, {}, {}); color: {}", c[0], c[1], c[2], if light { "#111" } else { "#eee" })
}

/// A rows x cols array as a colored table (blue negative, red positive).
fn table(name: &str, v: &[f64], rows: usize, cols: usize, digits: usize) -> Html {
    let scale = v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    html! { <figure class="arr">
        <table class="zs heat">
            { for (0..rows).map(|r| html! { <tr>{ for (0..cols).map(|c| { let x = v[r * cols + c]; html! { <td style={cell_style(x, scale)} title={format!("{x}")}>{format!("{x:.digits$}")}</td> } }) }</tr> }) }
        </table>
        <figcaption>{code(name)}{format!(" {rows} x {cols}")}</figcaption>
    </figure> }
}

fn stage_chip(m: &UseReducerHandle<Model>, s: Stage) -> Html {
    let (name, src, dims, meaning): (&str, &str, Vec<usize>, &str) = match s {
        Stage::Forward => ("forward", "nn:s_oftmax H nn:d_ense W2", vec![N, K], "each point's probabilities for the three arms"),
        Stage::Backward => ("backward", "(o_\\ u:o_nes X) '+ '* i_nner D1", vec![I + 1, J], "how the loss changes with each weight"),
        Stage::Step => ("step", "W1 - lr * G1", vec![I + 1, J], "the weights moved against their gradient"),
        Stage::Check => ("check", "F1 := ... e_ach r_ange 12", vec![I + 1, J], "each gradient again, by nudging each weight"),
    };
    chip(name, src, &dims, meaning, m.focus == s, act(m, move || Action::Focus(s)))
}

fn controls(m: &UseReducerHandle<Model>) -> Html {
    let d = m.dispatcher();
    let on_lr = Callback::from(move |e: Event| d.dispatch(Action::Rate(e.target_unchecked_into::<HtmlInputElement>().value().parse().unwrap_or(0.5))));
    html! {
        <div class="controls">
            <button class="on" onclick={act(m, || Action::Take)}>{"Take this step"}</button>
            <button onclick={act(m, || Action::Reset)}>{"Start again"}</button>
            <label class="slider">{"learning rate "}{code("lr")}
                <input type="range" min="0.05" max="3" step="0.05" value={m.setup.lr.to_string()} onchange={on_lr} />
                <b>{format!("{:.2}", m.setup.lr)}</b>
            </label>
            <span class="gen">{format!("steps taken: {} \u{00b7} X_eTaL ran the step and the check in {:.0} ms", m.losses.len().saturating_sub(1), m.ms)}</span>
        </div>
    }
}

fn forward(m: &UseReducerHandle<Model>, s: &Step) -> Html {
    let body = html! { <>
        <div class="arrs">{ table("X", &s.x, N, I, 2) }{ table("Y", &s.y, N, K, 0) }{ table("H", &s.h, N, J, 3) }{ table("P", &s.p, N, K, 3) }</div>
        <p class="calc">{"The loss, the mean of "}{code("-log")}{" of each point's probability for its own arm: "}{code(&format!("L = {:.6}", s.loss.0))}</p>
    </> };
    panel("1. Forward:", "P := nn:s_oftmax (nn:t_anh X nn:d_ense W1) nn:d_ense W2",
        "Six points (x, y) and their arms (one-hot), through a tanh layer of 4 and a softmax over 3: every point at once.",
        m.focus == Stage::Forward, body)
}

fn backward(m: &UseReducerHandle<Model>, s: &Step) -> Html {
    let body = html! { <>
        <div class="arrs">{ table("D2", &s.d2, N, K, 3) }{ table("G2", &s.g2, J + 1, K, 4) }{ table("D1", &s.d1, N, J, 3) }{ table("G1", &s.g1, I + 1, J, 4) }</div>
        <p class="note">{"D2: the loss by each output score, P minus Y over the batch size (softmax and cross-entropy together give this). G2: each layer's gradient is its input (with a column of 1s for the bias row) transposed, times what comes back. D1: back through W2 (its bias row dropped) and through tanh, whose slope is 1 - H\u{00b2}."}</p>
    </> };
    panel("2. Backward:", "G1 := (o_\\ u:o_nes X) '+ '* i_nner D1",
        "Each layer's gradient is one expression, as its forward pass is: a matrix product with a transpose.",
        m.focus == Stage::Backward, body)
}

fn step(m: &UseReducerHandle<Model>, s: &Step) -> Html {
    let (a, b) = s.loss;
    let body = html! { <>
        <div class="arrs">{ table("V1", &s.v1, I + 1, J, 3) }{ table("V2", &s.v2, J + 1, K, 3) }</div>
        <p class="calc">{"The loss now "}{code(&format!("{a:.6}"))}{", after this step "}{code(&format!("{b:.6}"))}{if b < a { " (lower)" } else { " (higher: the learning rate is too large)" }}</p>
        <p class="note">{"The loss at each step taken, now, and after this step (losses, stretched to their own range):"}</p>
        { picture(&s.chart, "the loss at each step") }
        <p class="note">{"Losses so far: "}{ m.losses.iter().chain([&b]).map(|l| format!("{l:.4}")).collect::<Vec<_>>().join(", ") }{". Drawn by the Plot library's p:l_ine!, in the program's last lines."}</p>
    </> };
    panel("3. The step:", "V1 := W1 - lr * G1", "Every weight moves against its gradient, scaled by the learning rate. Take the step to make these the weights, and run the step again from there.", m.focus == Stage::Step, body)
}

fn check(m: &UseReducerHandle<Model>, s: &Step) -> Html {
    let body = html! { <>
        <div class="arrs">{ table("G1", &s.g1, I + 1, J, 6) }{ table("F1", &s.f1, I + 1, J, 6) }{ table("G2", &s.g2, J + 1, K, 6) }{ table("F2", &s.f2, J + 1, K, 6) }</div>
        <p class="calc">{"The largest difference over all 27 weights: "}{code(&format!("{:.2e}", s.gap))}{if s.gap < 1e-8 { ": the gradients agree." } else { ": they differ." }}</p>
    </> };
    panel("4. The check:", "(L(w + e) - L(w - e)) / 2e",
        "Each gradient computed again without any calculus: nudge one weight up and down by e = 0.00001, run the whole network twice, and see how the loss moved. 27 weights, 54 forward passes.",
        m.focus == Stage::Check, body)
}

#[function_component(App)]
pub fn app() -> Html {
    let model = use_reducer(Model::new);
    let body = match &model.step {
        Some(s) => html! { <>
            <div class="layout even">
                <div class="col">{ forward(&model, s) }{ backward(&model, s) }{ step(&model, s) }{ check(&model, s) }</div>
                <div class="col">
                    <section class="panel code">
                        <h2>{"The program"}</h2>
                        <p class="note">{"Everything the page runs, in your browser: the head of backprop.xtl (the batch and weights, forward, backward, the step), then the check and the arrays the page shows; the stage you pick is highlighted."}</p>
                        { source(&model.setup, model.focus) }
                    </section>
                </div>
            </div>
        </> },
        None => html! {},
    };
    html! {
        <>
        <header>
            { header("Backprop microscope", "One step of training a small network, run by X_eTaL in your browser, with every array shown: the forward pass, the loss, each layer's gradient (one expression each), and the step. Each gradient is then checked by nudging every weight and watching the loss. Take steps and watch the loss fall.") }
            <nav class="timeline">{ for STAGES.iter().map(|&s| stage_chip(&model, s)) }</nav>
            { controls(&model) }
            { notice(&model.notice) }
        </header>
        <main>{body}</main>
        { footer() }
        </>
    }
}
