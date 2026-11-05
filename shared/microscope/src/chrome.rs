//! Around every demo page: the header (logo, title, lede), stage chips,
//! panels, notices and the footer (as the X_eTaL live demo shows it).

use yew::prelude::*;

use crate::source::{code, shape};

pub const REPO: &str = "https://github.com/softwarewrighter/X_eTaL-ML";

/// Whose page it is (X_eTaL-fine-tuning's ask FM2): the repository's
/// name, its URL, its commit, and the links of its header and footer
/// (relative to the page). A page of another repository builds its own
/// (its commit from its own build.rs) and passes it to `header_of` and
/// `footer_of`; this repository's pages use `THIS`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Site {
    pub name: &'static str,
    pub url: &'static str,
    pub commit: &'static str,
    /// Where the logo leads, and its title.
    pub catalog: (&'static str, &'static str),
    /// The footer's links after Repository: (text, href).
    pub links: &'static [(&'static str, &'static str)],
}

/// The commit of X_eTaL-ML this crate was built from (for another
/// repository's page, the commit of the X_eTaL-ML clone it pins).
pub const ML_COMMIT: &str = env!("BUILD_SHA");

/// This repository's pages.
pub const THIS: Site = Site {
    name: "X_eTaL-ML",
    url: REPO,
    commit: ML_COMMIT,
    catalog: ("../", "All demos"),
    links: &[("All demos", "../"), ("Libraries", "../libs/"), ("Cross-reference", "../doc/")],
};

/// The footer's provenance, as text: X_eTaL-ML's commit when the page is
/// another repository's, the pinned X_eTaL, then the build (host, the
/// site's commit, time).
pub fn provenance(site: &Site) -> Vec<String> {
    let ml = (site.name != THIS.name).then(|| format!("X_eTaL-ML {ML_COMMIT}"));
    ml.into_iter()
        .chain([format!("X_eTaL {}", env!("XETAL_SHA")), format!("build (host {}, sha {}, {})", env!("BUILD_HOST"), site.commit, env!("BUILD_TIMESTAMP"))])
        .collect()
}

/// The logo (linking back to the catalog), the title and a lede.
pub fn header(title: &str, lede: &str) -> Html {
    header_of(&THIS, title, lede)
}

/// The header of `site`'s page.
pub fn header_of(site: &Site, title: &str, lede: &str) -> Html {
    let (href, tip) = site.catalog;
    html! { <>
        <div class="brand">
            <a {href} title={tip}><img class="logo" src="xetal-logo-orange.png" alt="X_eTaL" /></a>
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
/// X_eTaL commit, the way back to the catalog and the cross-reference
/// (pages/doc).
pub fn footer() -> Html {
    footer_of(&THIS)
}

/// The footer of `site`'s page: its repository, its links, then
/// `provenance`.
pub fn footer_of(site: &Site) -> Html {
    let links = site.links.iter().map(|(text, href)| html! { <><a href={*href}>{ *text }</a>{ sep() }</> });
    let last = provenance(site).len() - 1;
    let marks = provenance(site).into_iter().enumerate().map(|(i, p)| html! { <><span>{ p }</span>if i < last { { sep() } }</> });
    html! {
        <footer>
            <span>{ "Copyright (c) 2026 Michael A Wright" }</span>{ sep() }
            <span>{ "MIT License" }</span>{ sep() }
            <a href={site.url} target="_blank">{ "Repository" }</a>{ sep() }
            { for links }
            { for marks }
        </footer>
    }
}
