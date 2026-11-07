//! A computation stepped through, forward and back, from a recorded
//! trace (docs/plan.md, F6: a trace, not a re-implementation). The
//! trace is what X_eTaL-ML's Flow reports for a pipeline (`@
//! flow:s_tages<`): every stage's value and shape, from one run.
//! Stepping only moves a position in that record, so going back, or
//! playing it again, runs nothing and changes no value.

use gloo_timers::callback::Timeout;
use microscope::stages::{inputs, stage_ranges, values};
use yew::prelude::*;

/// One stage of a pipeline, as its run recorded it.
#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    /// Its number, from 1 (the data first).
    pub number: usize,
    /// The name the pipeline gives its value (`h`), or "" for the last.
    pub label: String,
    pub shape: Vec<usize>,
    pub items: Vec<f64>,
    /// The earlier stages it reads, by number.
    pub reads: Vec<usize>,
    /// Its text in the pipeline, as written.
    pub text: String,
}

/// A pipeline's stages in order, from one run.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Trace {
    pub stages: Vec<Stage>,
}

/// The trace of a pipeline: `line` is what `@ flow:s_tages<` printed
/// for it, `statement` the pipeline as the program writes it (from
/// which each stage's text, and so what it reads, is taken).
pub fn trace(line: &str, statement: &str) -> Result<Trace, String> {
    let vals = values(line)?;
    let ranges = stage_ranges(statement);
    if ranges.len() != vals.len() {
        return Err(format!("the pipeline has {} stages and its run recorded {}", ranges.len(), vals.len()));
    }
    let labels: Vec<(usize, String)> = vals.iter().filter(|v| !v.label.is_empty()).map(|v| (v.stage, v.label.clone())).collect();
    let stages = vals
        .into_iter()
        .zip(ranges)
        .map(|(v, (from, to))| {
            let text = statement[from..to].to_string();
            Stage { number: v.stage, reads: inputs(&text, v.stage, &labels), label: v.label, shape: v.shape, items: v.items, text }
        })
        .collect();
    Ok(Trace { stages })
}

/// The step after `step` of `len` (from 0), staying on the last.
pub fn next(step: usize, len: usize) -> usize {
    (step + 1).min(len.saturating_sub(1))
}

/// The step before `step`, staying on the first.
pub fn previous(step: usize) -> usize {
    step.saturating_sub(1)
}

#[derive(Properties, PartialEq)]
pub struct StepperProps {
    /// How many steps the trace has.
    pub len: usize,
    /// The step shown, from 0.
    pub step: usize,
    /// Asked to show another step.
    pub onstep: Callback<usize>,
    /// How long a step is held while playing, in milliseconds.
    #[prop_or(1600)]
    pub hold: u32,
    /// What the page calls a step ("Stage"), for "Stage 2 of 4".
    #[prop_or(AttrValue::Static("Step"))]
    pub unit: AttrValue,
}

/// The controls: to the start, back, play or pause, forward, and where
/// in the trace the page is. Buttons, so they work by keyboard and
/// touch. Playing moves forward one step at a time and stops at the end.
#[function_component(Stepper)]
pub fn stepper(props: &StepperProps) -> Html {
    let playing = use_state(|| false);
    let (len, step) = (props.len, props.step);
    {
        let (playing, onstep, hold) = (playing.clone(), props.onstep.clone(), props.hold);
        use_effect_with((*playing, step, len), move |&(on, step, len)| {
            let timer = (on && step + 1 < len).then(|| Timeout::new(hold, move || onstep.emit(step + 1)));
            if on && step + 1 >= len {
                playing.set(false);
            }
            move || drop(timer)
        });
    }
    let go = |to: usize| { let onstep = props.onstep.clone(); Callback::from(move |_| onstep.emit(to)) };
    let toggle = {
        let (playing, onstep) = (playing.clone(), props.onstep.clone());
        Callback::from(move |_| {
            // Play from the end starts over.
            if !*playing && step + 1 >= len {
                onstep.emit(0);
            }
            playing.set(!*playing);
        })
    };
    html! {
        <div class="stepper" role="group" aria-label="Step through the computation">
            <button type="button" onclick={go(0)} disabled={step == 0} title="Back to the first">{"Start"}</button>
            <button type="button" onclick={go(previous(step))} disabled={step == 0}>{"Back"}</button>
            <button type="button" class="play" onclick={toggle} aria-pressed={playing.to_string()}>{ if *playing { "Pause" } else { "Play" } }</button>
            <button type="button" onclick={go(next(step, len))} disabled={step + 1 >= len}>{"Next"}</button>
            <span class="where" role="status">{ format!("{} {} of {}", props.unit, step + 1, len) }</span>
            <span class="track" aria-hidden="true">
                { for (0..len).map(|i| html! { <span class={classes!("seg", (i <= step).then_some("done"), (i == step).then_some("now"))}></span> }) }
                if *playing && step + 1 < len {
                    <span class="tick" key={step} style={format!("animation-duration: {}ms", props.hold)}></span>
                }
            </span>
        </div>
    }
}

/// Where the stages are drawn in the flow picture: `n` of them in a
/// row `width` wide; the center of stage `i` (from 0).
pub fn node_x(i: usize, n: usize, width: f64) -> f64 {
    width * (i as f64 + 0.5) / n as f64
}

/// The stages in a row, with the numbers a stage reads shown moving
/// into it: the stage on show is ringed, each stage it reads sends a
/// dot along an arc to it. The dots move once each time the stage
/// changes (give the picture a `key` that changes with the step); a
/// reader who has asked their system for less motion sees the arcs
/// without the dots. It draws the record: it changes no value.
pub fn flow(names: &[&str], step: usize, reads: &[usize]) -> Html {
    let (width, height, base) = (460.0, 92.0, 62.0);
    let n = names.len().max(1);
    let to = node_x(step, n, width);
    let arcs = reads.iter().map(|r| {
        // `r` is a stage's number, from 1.
        let from = node_x(r - 1, n, width);
        let lift = base - 18.0 - 30.0 * ((to - from).abs() / width);
        let path = format!("M {from} {} Q {} {lift} {to} {}", base - 16.0, (from + to) / 2.0, base - 16.0);
        html! { <>
            <path d={path.clone()} class="arc" />
            <circle r="5" class="dot">
                // Written as a named tag: SVG's element names keep their capitals,
                // and html! would lower them, leaving an element that does nothing.
                <@{"animateMotion"} dur="0.9s" fill="freeze" path={path} />
            </circle>
        </> }
    });
    let nodes = names.iter().enumerate().map(|(i, name)| {
        let x = node_x(i, n, width);
        let class = if i == step { "node on" } else if reads.contains(&(i + 1)) { "node read" } else if i < step { "node done" } else { "node" };
        html! { <g class={class}><circle cx={x.to_string()} cy={base.to_string()} r="16" /><text x={x.to_string()} y={base.to_string()}>{ *name }</text></g> }
    });
    html! {
        <svg class="flow" viewBox={format!("0 0 {width} {height}")} width={width.to_string()} height={height.to_string()} role="img"
            aria-label={format!("Stage {} of {}: {}", step + 1, names.len(), names.get(step).copied().unwrap_or(""))}>
            { for arcs }
            { for nodes }
        </svg>
    }
}
