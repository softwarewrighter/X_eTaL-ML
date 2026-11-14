//! The minimap: every year from the first to the last at the same
//! width, always fitting the window; a tick for each event (a demo's in
//! its own color), the decades labeled, the folded runs of empty years
//! shaded (the minimap keeps them, so a fold never hides where the view
//! is), and a frame around what the strip shows. A click centers the
//! strip there; a drag moves it along.

use ml_timeline::{Go, Scale, mini, mini_year};
use wasm_bindgen::JsCast;
use web_sys::{Element, PointerEvent};
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct Props {
    pub scale: Scale,
    /// Each event's place in years, and whether it is a demo.
    pub marks: Vec<(f64, bool)>,
    /// The years at the strip's left and right edges.
    pub frame: (f64, f64),
    pub go: Callback<(Go, bool)>,
}

#[function_component]
pub fn Minimap(props: &Props) -> Html {
    let Props { scale, marks, frame, go } = props;
    let pct = |at: f64| 100.0 * mini(scale, at);
    let decades = (scale.first() / 10 + 1..=scale.last() / 10).map(|d| {
        let year = d * 10;
        html! { <span class="decade" style={format!("left:{}%", pct(f64::from(year)))}>{ year }</span> }
    });
    let folds = scale.segments.iter().filter(|s| s.folded()).map(|s| {
        let (a, b) = (pct(f64::from(s.from)), pct(f64::from(s.to)));
        html! { <span class="folded" style={format!("left:{a}%;width:{}%", b - a)} title={format!("{}-{}: folded on the strip", s.from, s.to - 1)} /> }
    });
    let ticks = marks.iter().map(|&(at, demo)| {
        let class = if demo { "mark demo" } else { "mark" };
        html! { <span {class} style={format!("left:{}%", pct(at))} /> }
    });
    let (a, b) = (pct(frame.0), pct(frame.1));
    let (scale_at, go_to) = (scale.clone(), go.clone());
    let place = move |e: &PointerEvent, smooth: bool| {
        let Some(bar) = e.current_target().and_then(|t| t.dyn_into::<Element>().ok()) else { return };
        let rect = bar.get_bounding_client_rect();
        let fraction = (f64::from(e.client_x()) - rect.left()) / rect.width().max(1.0);
        go_to.emit((Go::Center(mini_year(&scale_at, fraction)), smooth));
    };
    let down = {
        let place = place.clone();
        Callback::from(move |e: PointerEvent| {
            if let Some(bar) = e.current_target().and_then(|t| t.dyn_into::<Element>().ok()) {
                let _ = bar.set_pointer_capture(e.pointer_id());
            }
            place(&e, true);
        })
    };
    let drag = Callback::from(move |e: PointerEvent| {
        if e.buttons() & 1 == 1 {
            place(&e, false);
        }
    });
    html! {
        <div class="minimap" onpointerdown={down} onpointermove={drag} role="slider" aria-label="The years: click or drag to move the strip"
            aria-valuemin={scale.first().to_string()} aria-valuemax={scale.last().to_string()} aria-valuenow={format!("{:.0}", frame.0.floor())}>
            <div class="band">
                { for folds }
                { for ticks }
                <span class="frame" style={format!("left:{a}%;width:{}%", (b - a).max(0.4))} />
            </div>
            <div class="decades">{ for decades }</div>
        </div>
    }
}
