//! Around every demo page: the header (logo, title, lede), stage chips,
//! panels, notices and the footer (as the X_eTaL live demo shows it).

use yew::prelude::*;

use crate::source::{code, shape};

pub const REPO: &str = "https://github.com/softwarewrighter/X_eTaL-ML";

/// The logo (linking back to the catalog), the title and a lede.
pub fn header(title: &str, lede: &str) -> Html {
    html! { <>
        <div class="brand">
            <a href="../" title="All demos"><img class="logo" src="modern-xetal-logo.jpg" alt="X_eTaL" /></a>
            <h1>{title}</h1>
        </div>
        <p class="lede">{lede}</p>
    </> }
}

/// One stage of the computation: its name, its code, its result's shape.
pub fn chip(name: &str, src: &str, dims: &[usize], meaning: &str, active: bool, onclick: Callback<MouseEvent>) -> Html {
    html! {
        <button class={classes!("stage", active.then_some("active"))} {onclick}>
            <span class="sname">{name}</span>
            { code(src) }
            { shape(dims, meaning) }
        </button>
    }
}

/// A panel: a title followed by its code, a note, a body; outlined when
/// it shows the stage in focus.
pub fn panel(title: &str, src: &str, note: &str, focus: bool, body: Html) -> Html {
    html! {
        <section class={classes!("panel", focus.then_some("focus"))}>
            <h2>{title}{" "}{code(src)}</h2>
            <p class="note">{note}</p>
            {body}
        </section>
    }
}

/// A picture the program showed with `[]S_HOW` (an SVG document X_eTaL
/// wrote from numbers), scaled to the panel.
pub fn picture(svg: &str, label: &str) -> Html {
    html! { <div class="picture" role="img" aria-label={label.to_string()}>{ Html::from_html_unchecked(AttrValue::from(svg.to_string())) }</div> }
}

/// Why the last change was not shown, if there is a reason.
pub fn notice(text: &Option<String>) -> Html {
    html! { for text.iter().map(|n| html! { <p class="notice" role="status">{n}</p> }) }
}

fn sep() -> Html {
    html! { <span class="sep">{ "\u{00b7}" }</span> }
}

/// The footer, as the X_eTaL live demo shows it, plus the pinned
/// X_eTaL commit and the way back to the catalog.
pub fn footer() -> Html {
    html! {
        <footer>
            <span>{ "Copyright (c) 2026 Michael A Wright" }</span>{ sep() }
            <span>{ "MIT License" }</span>{ sep() }
            <a href={REPO} target="_blank">{ "Repository" }</a>{ sep() }
            <a href="../">{ "All demos" }</a>{ sep() }
            <span>{ format!("X_eTaL {}", env!("XETAL_SHA")) }</span>{ sep() }
            <span>{ format!("build (host {}, sha {}, {})", env!("BUILD_HOST"), env!("BUILD_SHA"), env!("BUILD_TIMESTAMP")) }</span>
        </footer>
    }
}
