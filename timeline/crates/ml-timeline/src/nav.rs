//! How the view moves: to the start and the end, back and on by most of
//! a window, a little with the arrow keys, to a year or a month typed in
//! the year box or named in the URL's hash, and to where the minimap is
//! clicked; and which year the view is at, for the hash to follow.

use crate::scale::Scale;

/// Where a year opens: this far in from the view's left edge.
pub const LEAD: f64 = 48.0;

/// A move of the view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Go {
    /// To the first year (`|<-`, Home).
    Start,
    /// Back by most of a window (`<<-`, Page Up).
    Back,
    /// On by most of a window (`->>`, Page Down).
    On,
    /// To the last year and the newest demos (`->|`, End).
    End,
    /// A little back (the left arrow).
    Left,
    /// A little on (the right arrow).
    Right,
    /// To a year with its fraction, opened `LEAD` in from the left.
    To(f64),
    /// To a year with its fraction in the middle of the view (the minimap).
    Center(f64),
}

/// How far the strip scrolls for `go` from `left`, in a view `view`
/// wide: held between the start and the end.
#[must_use]
pub fn scroll_for(go: Go, scale: &Scale, left: f64, view: f64) -> f64 {
    let most = (scale.width() - view).max(0.0);
    let to = match go {
        Go::Start => 0.0,
        Go::End => most,
        Go::Back => left - 0.8 * view,
        Go::On => left + 0.8 * view,
        Go::Left => left - (0.2 * view).max(120.0),
        Go::Right => left + (0.2 * view).max(120.0),
        Go::To(at) => scale.x(at) - LEAD,
        Go::Center(at) => scale.x(at) - view / 2.0,
    };
    to.clamp(0.0, most)
}

/// The key's move, for the keys the strip answers.
#[must_use]
pub fn key(name: &str) -> Option<Go> {
    match name {
        "Home" => Some(Go::Start),
        "End" => Some(Go::End),
        "PageUp" => Some(Go::Back),
        "PageDown" => Some(Go::On),
        "ArrowLeft" => Some(Go::Left),
        "ArrowRight" => Some(Go::Right),
        _ => None,
    }
}

/// A year, and maybe a month, from a hash or the year box: `#1989`,
/// `1989`, `2017-06`, `#2017-6`, or a decade, `1990s`. None for
/// anything else, or a month outside 1 to 12.
#[must_use]
pub fn parse(text: &str) -> Option<(i32, Option<u8>)> {
    let text = text.trim().trim_start_matches('#');
    let text = text.strip_suffix('s').unwrap_or(text);
    let (year, month) = match text.split_once('-') {
        Some((y, m)) => (y, Some(m)),
        None => (text, None),
    };
    if year.len() != 4 || !year.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let month = match month {
        None => None,
        Some(m) if (1..=2).contains(&m.len()) && m.bytes().all(|b| b.is_ascii_digit()) => {
            Some(m.parse::<u8>().ok().filter(|m| (1..=12).contains(m))?)
        }
        Some(_) => return None,
    };
    Some((year.parse().ok()?, month))
}

/// Where a parsed year and month open: the start of the month, or of
/// the year.
#[must_use]
pub fn target((year, month): (i32, Option<u8>)) -> f64 {
    f64::from(year) + month.map_or(0.0, |m| f64::from(m - 1) / 12.0)
}

/// The hash for the view scrolled to `left`: the year `LEAD` in from its
/// left edge, with the month (`#2017-06`) inside a year with events
/// once past its first month; the year alone (`#1989`) otherwise.
#[must_use]
pub fn hash(scale: &Scale, left: f64) -> String {
    let at = scale.year_at(left + LEAD) + 1e-9;
    #[allow(clippy::cast_possible_truncation)]
    let year = at.floor() as i32;
    let open = scale.segment(year).is_some_and(|s| !s.folded() && s.events > 0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let month = (((at - f64::from(year)) * 12.0).floor() as u8 + 1).min(12);
    if open && month > 1 {
        format!("#{year}-{month:02}")
    } else {
        format!("#{year}")
    }
}

/// The years at the view's left and right edges, for the minimap's frame.
#[must_use]
pub fn frame(scale: &Scale, left: f64, view: f64) -> (f64, f64) {
    (scale.year_at(left), scale.year_at(left + view))
}

/// Where `at` falls across the minimap, 0 to 1: every year as wide.
#[must_use]
pub fn mini(scale: &Scale, at: f64) -> f64 {
    let (first, last) = (f64::from(scale.first()), f64::from(scale.last() + 1));
    ((at - first) / (last - first)).clamp(0.0, 1.0)
}

/// The year with its fraction at `fraction` (0 to 1) across the minimap.
#[must_use]
pub fn mini_year(scale: &Scale, fraction: f64) -> f64 {
    let (first, last) = (f64::from(scale.first()), f64::from(scale.last() + 1));
    first + fraction.clamp(0.0, 1.0) * (last - first)
}
