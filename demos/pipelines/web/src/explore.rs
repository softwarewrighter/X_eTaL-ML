//! Compose's stage view: the first pipeline of the program a person
//! ran, drawn as math a row a stage and as written with a button a
//! stage, the chosen stage highlighted in both; what it gives, and in
//! 3D what it reads and gives, in the right pane.

use yew::prelude::*;

use microscope::chrome::panel;
use microscope::source::{block, NONE};

use crate::app::{no_stage, row_picker, stage_view, Split, Turn};
use crate::micro::{clickable, explore, stage_ranges};

#[derive(Properties, PartialEq)]
pub struct Props {
    /// The program that was run.
    pub src: String,
}

#[function_component(Stages)]
pub fn stages(p: &Props) -> Html {
    // Run once per program, not on every render.
    let found = use_memo(p.src.clone(), |src| explore(src));
    let chosen = use_state(|| None::<usize>);
    let turn = use_state(Turn::default);
    {
        // A new program, a new choice.
        let chosen = chosen.clone();
        use_effect_with(p.src.clone(), move |_| chosen.set(None));
    }
    let x = match found.as_ref() {
        None => return html! {},
        Some(Err(why)) => {
            return panel("The stages", "flow:s_tages<", "Its first pipeline's stages could not be shown: run at the top of the program, a pipeline's names must be bound there (not inside a function).", false, html! { <pre class="err">{ why.clone() }</pre> })
        }
        Some(Ok(x)) => x,
    };
    let on = *chosen;
    let c = chosen.clone();
    let pick = row_picker(Callback::from(move |k| c.set(Some(k))));
    let texts = stage_ranges(&x.call);
    let at = on.and_then(|k| texts.get(k - 1).copied()).unwrap_or(NONE);
    let chips = html! { <div class="stages">
        { for (1..=texts.len()).map(|k| {
            let c = chosen.clone();
            let label = x.trace.iter().filter(|v| v.stage == k).map(|v| v.label.clone()).filter(|l| !l.is_empty()).collect::<Vec<_>>().join(", ");
            let title = if label.is_empty() { format!("stage {k}") } else { format!("stage {k}: {label}") };
            html! { <button class={classes!((on == Some(k)).then_some("on"))} onclick={Callback::from(move |_| c.set(Some(k)))}>{title}</button> }
        }) }
    </div> };
    let shown = clickable(&x.tex, on);
    let left = panel("The stages", "flow:m_ath<", "The program's first pipeline, drawn as math by Flow's m_ath< (a row a stage) and as written: click a row or a stage to see on the right what it gives, its numbers and in 3D the stages it reads.", false, html! { <>
        <div class="tex" style="overflow-x: auto" onclick={pick} data-tex={shown.clone()}>{shown.clone()}</div>
        { block(&x.call, at) }
        { chips }
    </> });
    let right = match on {
        Some(k) => stage_view(&x.trace, &x.call, k, "of your pipeline", &turn),
        None => no_stage(),
    };
    let c = chosen.clone();
    html! { <Split idle={on.is_none()} close={Callback::from(move |_| c.set(None))} {left} {right} /> }
}
