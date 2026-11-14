//! The ML timeline: its events, the scale that puts a year and a month
//! at a place across the strip (each year as wide as its callouts need,
//! empty years narrow and long runs of them folded), where each callout
//! goes, and how the view moves: the buttons, the keys, the year box,
//! the minimap and the URL's hash. Nothing here needs a browser, so all
//! of it is tested natively; the Yew crate beside this one only draws
//! what it is given.
//!
//! Forked from the Lisp timeline of web-sw-cor24-macrolisp
//! (`lisp-timeline/`), itself from sw-apl's by way of the Forth one:
//! the layout is theirs; the months, the kinds, the folded scale, the
//! minimap and the navigation are new.

mod event;
mod layout;
mod nav;
mod scale;

pub use event::{Event, Kind, check, events};
pub use layout::{Placed, Side, place};
pub use nav::{Go, LEAD, frame, hash, key, mini, mini_year, parse, scroll_for, target};
pub use scale::{EMPTY, FIRST, FOLD, FOLD_RUN, MIN_YEAR, NOW, Scale, Segment};
