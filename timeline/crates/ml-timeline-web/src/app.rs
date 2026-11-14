//! The page: the controls (`|<-`, `<<-`, `->>`, `->|` and the year
//! box), the minimap of every year, and the strip of months within
//! years, which scrolls sideways. The view opens at the URL's hash
//! (`#1989`, `#2017-06`) or at the start, and the hash follows it, so a
//! year can be linked. Home, End, Page Up and Down and the arrow keys
//! move it too.
//!
//! As the Lisp timeline's, the strip is drawn twice: first with each
//! box as tall as `Event::size` guesses, then with the heights the
//! browser's fonts actually take, so no words are cut off. The widths,
//! and so the scale, are fixed by the kind of box, so the view does not
//! move between the two drawings.

use std::cell::RefCell;
use std::rc::Rc;

use gloo_events::EventListener;
use gloo_timers::callback::Timeout;
use ml_timeline::{Event, FIRST, Go, NOW, Placed, Scale, Side, events, frame, hash, key, parse, place, scroll_for, target};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{Element, HtmlElement, HtmlInputElement, KeyboardEvent, ScrollBehavior, ScrollToOptions};
use yew::prelude::*;

use crate::axis::Axis;
use crate::callout::Callout;
use crate::minimap::Minimap;

/// The events, built in, so the page fetches nothing for them.
const DATA: &str = include_str!("../../../data/timeline.json");

/// Room before the first year and after the last, for callouts that fan
/// out past the ends.
const MARGIN: f64 = 260.0;

/// Callouts start this far from the axis line, on either side.
const BAND: f64 = 44.0;

/// Below this window width the boxes are a phone's.
const PHONE: f64 = 640.0;

/// The timeline.
#[function_component]
pub fn App() -> Html {
    let list = use_memo((), |()| events(DATA).unwrap_or_default());
    let wide = use_window_width();
    let narrow = wide < PHONE;
    let scale = Rc::new(scale_of(&list, narrow));
    let (strip, track) = (use_node_ref(), use_node_ref());
    let placed = place(&use_sizes(&list, &track, narrow), &scale);
    let axis = reach(&placed, Side::Above) + BAND + 8.0;
    let height = axis + reach(&placed, Side::Below) + BAND + 16.0;
    let view = use_state(|| (0.0, wide));
    let go = mover(&strip, &scale);
    use_opening(&strip, &scale);
    use_keys(&go);
    let pending = use_mut_ref(|| None::<Timeout>);
    let onscroll = follow(&strip, &scale, view.setter(), pending);
    let (left, width) = *view;
    let marks: Vec<(f64, bool)> = list.iter().map(|e| (e.at(), e.demo.is_some())).collect();
    let callouts = list.iter().zip(&placed).enumerate().map(|(tone, (event, placed))| {
        html! { <Callout event={event.clone()} placed={*placed} {axis} band={BAND} x={scale.x(event.at())} {tone} /> }
    });
    html! {
        <>
            { controls(&go) }
            <Minimap scale={(*scale).clone()} {marks} frame={frame(&scale, left, width)} go={go.clone()} />
            <div class="strip" ref={strip.clone()} tabindex="0" {onscroll} onwheel={sideways(strip)} aria-label="The timeline: scroll sideways">
                <div class="track" ref={track} style={format!("width:{}px;height:{height}px", scale.width())}>
                    <Axis scale={(*scale).clone()} y={axis} />
                    { for callouts }
                </div>
            </div>
        </>
    }
}

/// The scale for the events' boxes, a phone's when `narrow`.
fn scale_of(list: &[Event], narrow: bool) -> Scale {
    let sized: Vec<(f64, f64)> = list.iter().map(|e| (e.at(), e.size(narrow).0)).collect();
    Scale::new(FIRST, NOW, &sized, MARGIN)
}

/// The buttons and the year box: a move each, and a year (or a month,
/// or a decade picked from the list) typed and entered.
fn controls(go: &Callback<(Go, bool)>) -> Html {
    let button = |label: &str, tip: &str, to: Go| {
        let go = go.clone();
        html! {
            <button class="go" title={tip.to_string()} aria-label={tip.to_string()}
                onclick={Callback::from(move |_| go.emit((to, true)))}>{ label }</button>
        }
    };
    let jump = {
        let go = go.clone();
        Callback::from(move |e: SubmitEvent| {
            e.prevent_default();
            let input = e.target().and_then(|f| f.dyn_into::<Element>().ok()).and_then(|f| f.query_selector("input").ok().flatten());
            if let Some(input) = input.and_then(|i| i.dyn_into::<HtmlInputElement>().ok()) {
                match parse(&input.value()) {
                    Some(when) => {
                        input.set_class_name("");
                        go.emit((Go::To(target(when)), true));
                    }
                    None => input.set_class_name("bad"),
                }
            }
        })
    };
    let decades = (FIRST / 10..=NOW / 10).map(|d| html! { <option value={format!("{}s", d * 10)} /> });
    html! {
        <nav class="controls" aria-label="Move through the years">
            { button("|<-", "To the start, 1943", Go::Start) }
            { button("<<-", "Back by a window", Go::Back) }
            { button("->>", "On by a window", Go::On) }
            { button("->|", "To the end: today, and the newest demos", Go::End) }
            <form class="jump" onsubmit={jump}>
                <label for="year">{ "Go to" }</label>
                <input id="year" list="decades" inputmode="numeric" placeholder="1989, 2017-06, 1990s" size="14" />
                <datalist id="decades">{ for decades }</datalist>
                <button type="submit">{ "Go" }</button>
            </form>
        </nav>
    }
}

/// A move of the strip: smoothly, or at once while the minimap is dragged.
fn mover(strip: &NodeRef, scale: &Rc<Scale>) -> Callback<(Go, bool)> {
    let (strip, scale) = (strip.clone(), scale.clone());
    Callback::from(move |(go, smooth): (Go, bool)| move_strip(&strip, &scale, go, smooth))
}

fn move_strip(strip: &NodeRef, scale: &Scale, go: Go, smooth: bool) {
    let Some(strip) = strip.cast::<Element>() else { return };
    let left = scroll_for(go, scale, f64::from(strip.scroll_left()), f64::from(strip.client_width()));
    let options = ScrollToOptions::new();
    options.set_left(left);
    options.set_behavior(if smooth { ScrollBehavior::Smooth } else { ScrollBehavior::Instant });
    strip.scroll_to_with_scroll_to_options(&options);
}

/// Open at the hash's year, or at the start; and go to a hash typed
/// into the address bar later.
#[hook]
fn use_opening(strip: &NodeRef, scale: &Rc<Scale>) {
    let (strip, scale) = (strip.clone(), scale.clone());
    use_effect_with((), move |()| {
        let at_hash = move |smooth: bool| {
            let hash = web_sys::window().and_then(|w| w.location().hash().ok()).unwrap_or_default();
            let go = parse(&hash).map_or(Go::Start, |when| Go::To(target(when)));
            move_strip(&strip, &scale, go, smooth);
        };
        at_hash(false);
        let window = web_sys::window().expect("a window");
        let listener = EventListener::new(&window, "hashchange", move |_| at_hash(true));
        move || drop(listener)
    });
}

/// The keys the page answers, wherever the focus is but in the year box.
#[hook]
fn use_keys(go: &Callback<(Go, bool)>) {
    let go = go.clone();
    use_effect_with((), move |()| {
        let window = web_sys::window().expect("a window");
        let listener = EventListener::new(&window, "keydown", move |e| {
            let Some(e) = e.dyn_ref::<KeyboardEvent>() else { return };
            let typing = e.target().and_then(|t| t.dyn_into::<HtmlInputElement>().ok()).is_some();
            if let (false, Some(to)) = (typing || e.alt_key() || e.ctrl_key() || e.meta_key(), key(&e.key())) {
                e.prevent_default();
                go.emit((to, true));
            }
        });
        move || drop(listener)
    });
}

/// On a scroll, the view's edges for the minimap's frame at once, and
/// the hash a moment after the scrolling stops (browsers limit how
/// often a page may rewrite its address). The waiting timeout is kept
/// across drawings in `pending`: each scroll draws the page again, and
/// a timeout dropped with its callback would never fire.
fn follow(
    strip: &NodeRef,
    scale: &Rc<Scale>,
    view: UseStateSetter<(f64, f64)>,
    pending: Rc<RefCell<Option<Timeout>>>,
) -> Callback<yew::Event> {
    let (strip, scale) = (strip.clone(), scale.clone());
    Callback::from(move |_| {
        let Some(el) = strip.cast::<Element>() else { return };
        let left = f64::from(el.scroll_left());
        view.set((left, f64::from(el.client_width())));
        let scale = scale.clone();
        *pending.borrow_mut() = Some(Timeout::new(250, move || write_hash(&hash(&scale, left))));
    })
}

fn write_hash(hash: &str) {
    let Some(window) = web_sys::window() else { return };
    if window.location().hash().ok().as_deref() != Some(hash)
        && let Ok(history) = window.history()
    {
        let _ = history.replace_state_with_url(&JsValue::NULL, "", Some(hash));
    }
}

/// The window's width, kept up to date as it is resized.
#[hook]
fn use_window_width() -> f64 {
    let read = || web_sys::window().and_then(|w| w.inner_width().ok()).and_then(|v| v.as_f64()).unwrap_or(1280.0);
    let wide = use_state(read);
    let set = wide.setter();
    use_effect_with((), move |()| {
        let window = web_sys::window().expect("a window");
        let listener = EventListener::new(&window, "resize", move |_| set.set(read()));
        move || drop(listener)
    });
    *wide
}

/// Each event's place and box size: estimated, until the first drawing
/// has been measured in `track`, and then as measured.
#[hook]
fn use_sizes(list: &[Event], track: &NodeRef, narrow: bool) -> Vec<(f64, (f64, f64), Option<Side>)> {
    let measured = use_state(|| None::<Vec<f64>>);
    let (track, set) = (track.clone(), measured.clone());
    use_effect_with(narrow, move |_| {
        let found = track.cast::<Element>().map(|t| t.query_selector_all(".callout"));
        if let Some(Ok(boxes)) = found {
            let each = (0..boxes.length()).filter_map(|i| boxes.item(i)?.dyn_into::<HtmlElement>().ok());
            set.set(Some(each.map(|e| f64::from(e.offset_height())).collect()));
        }
    });
    let known = measured.as_ref();
    let height = |i: usize, guess: f64| known.and_then(|m| m.get(i).copied()).unwrap_or(guess).max(guess.min(40.0));
    let size = |(i, e): (usize, &Event)| {
        let (w, h) = e.size(narrow);
        (e.at(), (w, height(i, h)), e.side)
    };
    list.iter().enumerate().map(size).collect()
}

/// How far the callouts on `side` reach from the axis.
fn reach(placed: &[Placed], side: Side) -> f64 {
    let on = placed.iter().filter(|p| p.side == side);
    on.map(|p| p.offset + p.height).fold(0.0, f64::max)
}

/// A wheel turned up or down moves the strip left or right, since it
/// scrolls mostly one way.
fn sideways(strip: NodeRef) -> Callback<WheelEvent> {
    Callback::from(move |turn: WheelEvent| {
        let (down, across) = (turn.delta_y(), turn.delta_x());
        if let (true, false, Some(el)) = (down.abs() > across.abs(), turn.shift_key(), strip.cast::<Element>()) {
            el.scroll_by_with_x_and_y(down, 0.0);
        }
    })
}
