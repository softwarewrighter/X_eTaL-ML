//! The page: the cloud of words, turned by dragging or spinning, colored
//! by k-means' clusters; a word's nearest words; how the embeddings are
//! made; the program.

use gloo_timers::callback::Interval;
use yew::prelude::*;

use microscope::chrome::{footer, header, notice, panel};
use microscope::source::code;

use crate::micro::{words, NEAR};
use crate::model::{Action, Model};
use crate::view::{dots, source, COLORS, SIZE};

fn cloud(m: &UseReducerHandle<Model>) -> Html {
    let Some(c) = m.cloud.as_ref() else { return html! {} };
    let ws = words();
    let ds = dots(&c.xyz, m.yaw, m.pitch);
    let at = |w: usize| ds.iter().find(|d| d.word == w).copied();
    let near: Vec<usize> = c.near[m.picked * NEAR..(m.picked + 1) * NEAR].iter().map(|&j| j - 1).collect();
    let (d1, d2, d3, d4) = (m.dispatcher(), m.dispatcher(), m.dispatcher(), m.dispatcher());
    let down = Callback::from(move |e: PointerEvent| d1.dispatch(Action::Down(e.client_x() as f64, e.client_y() as f64)));
    let mv = Callback::from(move |e: PointerEvent| d2.dispatch(Action::Move(e.client_x() as f64, e.client_y() as f64)));
    let up = Callback::from(move |_: PointerEvent| d3.dispatch(Action::Up));
    let leave = Callback::from(move |_: PointerEvent| d4.dispatch(Action::Up));
    let lines = match at(m.picked) {
        Some(p) => near.iter().filter_map(|&j| at(j)).map(|q| html! { <line x1={format!("{:.1}", p.x)} y1={format!("{:.1}", p.y)} x2={format!("{:.1}", q.x)} y2={format!("{:.1}", q.y)} stroke="#f59f00" stroke-width="1.5" stroke-dasharray="4 3" /> }).collect::<Html>(),
        None => html! {},
    };
    html! {
        <svg class="cloud" viewBox={format!("0 0 {SIZE} {SIZE}")} width="100%" role="img" aria-label="the words in 3-D"
            onpointerdown={down} onpointermove={mv} onpointerup={up} onpointerleave={leave}>
            { lines }
            { for ds.iter().map(|d| {
                let color = COLORS[(c.cluster[d.word].max(1) - 1) % COLORS.len()];
                let picked = d.word == m.picked || near.contains(&d.word);
                let size = 10.0 + 3.0 * d.depth;
                let pick = { let dd = m.dispatcher(); let w = d.word; Callback::from(move |_: MouseEvent| dd.dispatch(Action::Pick(w))) };
                html! { <g class={classes!("word", picked.then_some("near"))} onclick={pick}>
                    <circle cx={format!("{:.1}", d.x)} cy={format!("{:.1}", d.y)} r={format!("{:.1}", (4.0 + 1.5 * d.depth).max(2.0))} fill={color} />
                    <text x={format!("{:.1}", d.x + 6.0)} y={format!("{:.1}", d.y + 4.0)} font-size={format!("{size:.1}")} fill={color}>{ws[d.word]}</text>
                </g> }
            }) }
        </svg>
    }
}

fn nearest(m: &UseReducerHandle<Model>) -> Html {
    let Some(c) = m.cloud.as_ref() else { return html! {} };
    let ws = words();
    let near: Vec<&str> = c.near[m.picked * NEAR..(m.picked + 1) * NEAR].iter().map(|&j| ws[j - 1]).collect();
    html! { <>
        <p class="calc"><b>{ws[m.picked]}</b>{": "}{near.join(", ")}</p>
        <p class="note">{"Click any word in the cloud to see its five nearest, by the cosine of their 64-number rows (the 3-D picture keeps only part of the distances)."}</p>
    </> }
}

#[function_component(App)]
pub fn app() -> Html {
    let model = use_reducer(Model::new);
    {
        // Spin a little every frame while spinning.
        let d = model.dispatcher();
        use_effect_with((), move |_| {
            let i = Interval::new(40, move || d.dispatch(Action::Tick));
            move || drop(i)
        });
    }
    let spinning = model.spinning;
    let spin = { let d = model.dispatcher(); Callback::from(move |_: MouseEvent| d.dispatch(Action::Spin(!spinning))) };
    let kept = model.cloud.as_ref().map_or(0.0, |c| c.kept);
    html! {
        <>
        <header>
            { header("Embedding explorer", "Word embeddings made from counts, in X_eTaL: 4,000 short sentences over 70 words; each word's embedding 64 numbers from which words occur near it. Words used alike point alike. Learn's PCA turns the 64 dimensions into the 3 that spread the words most; drag the cloud to turn it, click a word for its nearest.") }
            <div class="controls">
                <button class={classes!(spinning.then_some("on"))} onclick={spin}>{ if spinning { "Stop spinning" } else { "Spin" } }</button>
                <span class="gen">{format!("drag to turn \u{00b7} X_eTaL made the cloud in {:.0} ms", model.ms)}</span>
            </div>
            { notice(&model.notice) }
        </header>
        <main>
            <div class="layout even">
                <div class="col">
                    { panel("The words in 3-D:", "(mu, V, ev) ml:p_roject E", "Each word where the 3 directions of most spread put it, colored by the cluster k-means finds in the 64 numbers (it is not told the kinds). The animals, foods, colors, numbers and places gather; the little words (the, a, in) sit apart.", true, cloud(&model)) }
                    { panel("Nearest words:", "(E, 5) em:n_earest i", "", false, nearest(&model)) }
                    { panel("How the embeddings are made:", "em:p_pmi C", "", false, html! { <>
                        <p class="calc">{"1. Count how often each word occurs within 2 words of each other (sorting the pairs, then the length of each run): a 70 x 70 table, "}{code("C")}{"."}</p>
                        <p class="calc">{"2. Weight each count by how much more often than chance (positive pointwise mutual information): "}{code("P")}{"."}</p>
                        <p class="calc">{"3. Project each row to 64 numbers by a fixed +1/-1 matrix (random indexing), then scale to length 1: "}{code("E")}{"."}</p>
                        <p class="calc">{format!("4. PCA to 3 directions, which keep {:.0}% of the spread.", 100.0 * kept)}</p>
                    </> }) }
                </div>
                <div class="col">
                    <section class="panel code">
                        <h2>{"The program"}</h2>
                        <p class="note">{"Everything the page runs, in your browser, once: the head of embedding-explorer.xtl (the corpus, the embeddings, the 3-D directions), then the lines that give the page each word's place, cluster and nearest. Turning the cloud is the page's own drawing."}</p>
                        { source() }
                    </section>
                </div>
            </div>
        </main>
        { footer() }
        </>
    }
}
