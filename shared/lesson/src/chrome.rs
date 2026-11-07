//! Around every page: the header and footer of the repository whose
//! pages these are (named at build time, `LESSON_SITE`; see build.rs),
//! the frames of a page, asides. The header is microscope's, told whose
//! page it is (`microscope::chrome::Site`). The footer is this crate's:
//! the build compact and without labels, and the pins the dependent
//! names (`LESSON_PINS`).

use microscope::chrome::{header_of, Site};
use yew::prelude::*;

include!(concat!(env!("OUT_DIR"), "/site.rs"));

/// The repository whose pages these are, on GitHub.
pub const REPO: &str = SITE_URL;
pub const ML: &str = "https://github.com/softwarewrighter/X_eTaL-ML";

const LINKS: &[(&str, &str)] = &[(CATALOG, "../"), ("Cross-reference", "../doc/")];

/// Whose pages these are, for microscope's header (and its footer,
/// should a page want that form): the name, URL and catalog title of
/// `LESSON_SITE`, the commit of `LESSON_ROOT`.
pub const SITE: Site = Site { name: SITE_NAME, url: SITE_URL, commit: SHA, catalog: ("../", CATALOG), links: LINKS };

/// The header of a page here: the logo (leading back to the catalog),
/// the title and a lede.
pub fn header(title: &str, lede: &str) -> Html {
    header_of(&SITE, title, lede)
}

/// The build's provenance, compact and without labels, as the catalog's
/// footer shows it: the build host, the site's short commit and the
/// time, yyyymmddThhmm (UTC, no zone shown).
pub fn build() -> String {
    format!("{HOST} {} {TIMESTAMP}", sha())
}

/// The site's short commit at the build.
pub fn sha() -> &'static str {
    SHA
}

/// The pins the footer shows, as the dependent names them
/// (`LESSON_PINS`): each repository's name and the full commit pinned.
pub fn pins() -> &'static [(&'static str, &'static str)] {
    PINS
}

/// The short commit pinned for the repository `name`, or "unknown".
pub fn pin(name: &str) -> &'static str {
    PINS.iter().find(|(n, _)| *n == name).map_or("unknown", |(_, c)| if c.len() >= 7 { &c[..7] } else { c })
}

/// The commit of X_eTaL-ML this crate is built from (the clone a
/// dependent pins).
pub fn ml_commit() -> &'static str {
    if ML_FULL.len() >= 7 { &ML_FULL[..7] } else { ML_FULL }
}

fn sep() -> Html {
    html! { <span class="sep">{ "\u{00b7}" }</span> }
}

/// The footer, as the sibling repositories' live pages have it:
/// copyright, license, the repository on GitHub, the way back to the
/// catalog, the cross-reference (pages/doc), each pin (linked to its
/// commit) and the build (host, short commit, time).
pub fn footer() -> Html {
    let pins = PINS.iter().map(|(name, commit)| {
        let short = pin(name);
        html! { <><span>{ format!("{name} ") }<a href={format!("https://github.com/softwarewrighter/{name}/commit/{commit}")} target="_blank">{ short }</a></span>{ sep() }</> }
    });
    html! {
        <footer>
            <span>{ "Copyright (c) 2026 Michael A Wright" }</span>{ sep() }
            <span>{ "MIT License" }</span>{ sep() }
            <a href={REPO} target="_blank">{ "GitHub" }</a>{ sep() }
            <a href="../">{ CATALOG }</a>{ sep() }
            <a href="../doc/">{ "Cross-reference" }</a>{ sep() }
            { for pins }
            <span class="build" data-sha={sha()}>{ build() }</span>
        </footer>
    }
}

/// The parts that frame every page (the user's review of the first
/// one, 2026-11-05: a page may not jump into its subject): the
/// question in a newcomer's words, the background needed, the
/// rationale (why this approach), the purpose (what this page is and
/// is not), and after the experiment a summary and what is next.
/// scripts/browser-check.sh fails a page that lacks one.
/// A dependent may add kinds of its own (`Frame::Other("proof")`), each
/// marked with its name, which its own browser check can require.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Frame {
    Question,
    Background,
    Rationale,
    Purpose,
    Summary,
    Next,
    Other(&'static str),
}

/// Every frame, in the order a page shows them.
pub const FRAMES: [Frame; 6] = [Frame::Question, Frame::Background, Frame::Rationale, Frame::Purpose, Frame::Summary, Frame::Next];

impl Frame {
    /// Its name in the page (`data-frame`), which the browser check reads.
    pub fn name(self) -> &'static str {
        match self {
            Frame::Question => "question",
            Frame::Background => "background",
            Frame::Rationale => "rationale",
            Frame::Purpose => "purpose",
            Frame::Summary => "summary",
            Frame::Next => "next",
            Frame::Other(name) => name,
        }
    }
}

/// A framing section: a titled panel of prose, marked with its kind.
pub fn frame(kind: Frame, title: &str, body: Html) -> Html {
    html! {
        <section class="panel frame" data-frame={kind.name()}>
            <h2>{title.to_string()}</h2>
            {body}
        </section>
    }
}

/// An aside: something a reader may want explained and another may
/// skip, hidden until its summary is opened (keyboard and touch too).
pub fn aside(id: &str, summary: &str, body: Html) -> Html {
    html! {
        <details class="aside" data-aside={id.to_string()}>
            <summary>{summary.to_string()}</summary>
            <div class="aside-body">{body}</div>
        </details>
    }
}

/// A panel as microscope's (`microscope::chrome::panel`), its heading's
/// code given already drawn, so it can be linked (`code::Linker`).
pub fn panel(title: &str, code: Html, note: &str, body: Html) -> Html {
    html! {
        <section class="panel">
            <h2>{title.to_string()}{" "}{code}</h2>
            <p class="note">{note.to_string()}</p>
            {body}
        </section>
    }
}
