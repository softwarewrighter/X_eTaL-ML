//! The axis: a line with an arrow; each year's tick and label, and in
//! a year with events its months, ticked where the year is wide enough
//! and lettered where it is wider; each folded run of empty years a
//! break in the line with its years written on it.

use ml_timeline::{Scale, Segment};
use yew::prelude::*;

/// Months are ticked in a year at least this wide, and lettered in one
/// at least `LETTERED` wide.
const TICKED: f64 = 120.0;
const LETTERED: f64 = 300.0;

#[derive(Properties, PartialEq)]
pub struct Props {
    pub scale: Scale,
    pub y: f64,
}

#[function_component]
pub fn Axis(props: &Props) -> Html {
    let Props { scale, y } = props;
    let (left, right) = (scale.margin / 2.0, scale.width() - scale.margin / 2.0);
    let parts = scale.segments.iter().map(|s| segment(s, *y));
    html! {
        <>
            <div class="line" style={format!("left:{left}px;top:{y}px;width:{}px", right - left)} />
            { for parts }
        </>
    }
}

fn segment(s: &Segment, y: f64) -> Html {
    if s.folded() {
        let years = format!("{}-{}", s.from, s.to - 1);
        return html! {
            <div class="fold" style={format!("left:{}px;top:{y}px;width:{}px", s.left, s.width)} title={format!("{years}: no events")}>
                <span class="break" />
                <span class="years">{ years }</span>
            </div>
        };
    }
    let busy = s.events > 0;
    let at = move |m: u8| s.left + s.width * f64::from(m) / 12.0;
    let ticks = (busy && s.width >= TICKED).then(|| {
        (1..12).map(move |m| html! { <div class="month" style={format!("left:{}px;top:{y}px", at(m))} /> })
    });
    let lettered = busy && s.width >= LETTERED;
    let letters = lettered.then(|| {
        (0..12u8).map(move |m| {
            let middle = at(m) + s.width / 24.0;
            let i = usize::from(m);
            html! { <div class="letter" style={format!("left:{middle}px;top:{}px", y + 6.0)}>{ &"JFMAMJJASOND"[i..=i] }</div> }
        })
    });
    let class = if busy { "year busy" } else { "year" };
    let below = if lettered { 26.0 } else { 10.0 };
    html! {
        <>
            <div class="tick" style={format!("left:{}px;top:{y}px", s.left)} />
            { for ticks.into_iter().flatten() }
            { for letters.into_iter().flatten() }
            <div {class} style={format!("left:{}px;top:{}px", s.left + s.width / 2.0, y + below)}>{ s.from }</div>
        </>
    }
}
