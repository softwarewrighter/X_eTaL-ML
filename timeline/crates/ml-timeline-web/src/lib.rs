//! The ML timeline in a browser: `ml-timeline` says how wide each year
//! is, where each callout goes and where each move takes the view, and
//! this draws the controls, the minimap, the strip, its axis and its
//! callouts with Yew, in the element `#timeline`.

mod app;
mod axis;
mod callout;
mod minimap;

use wasm_bindgen::prelude::wasm_bindgen;

/// Draw the timeline into `#timeline`, or into the body without one.
#[wasm_bindgen(start)]
pub fn start() {
    let document = web_sys::window().and_then(|w| w.document());
    let root = document.and_then(|d| d.get_element_by_id("timeline"));
    match root {
        Some(root) => yew::Renderer::<app::App>::with_root(root).render(),
        None => yew::Renderer::<app::App>::new().render(),
    };
}
