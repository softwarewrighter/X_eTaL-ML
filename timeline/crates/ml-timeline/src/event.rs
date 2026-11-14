//! One event on the timeline, as `data/timeline.json` writes it.

use serde::Deserialize;

use crate::layout::Side;
use crate::scale::{FIRST, NOW};

/// What an event is: a key paper, a milestone (a system, a result, an
/// event), a line of context, or one of this repository's demos.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Paper,
    Milestone,
    Note,
    Demo,
}

/// An event: its year and, when known, its month; when it was, in
/// words; what it was and, for a paper, by whom and where; where it
/// leads; a few topics; and, for a demo, its slug and picture.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Event {
    pub year: i32,
    /// 1 to 12; absent when only the year is known.
    #[serde(default)]
    pub month: Option<u8>,
    pub when: String,
    pub title: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub href: Option<String>,
    pub kind: Kind,
    #[serde(default)]
    pub topics: Vec<String>,
    /// For a demo: its slug, `demos/<slug>/`.
    #[serde(default)]
    pub demo: Option<String>,
    #[serde(default)]
    pub image: Option<String>,
    /// What the picture says when the pointer is over it.
    #[serde(default)]
    pub tip: Option<String>,
    /// The side of the axis the callout must take, when the layout's own
    /// choice is the wrong one.
    #[serde(default)]
    pub side: Option<Side>,
}

impl Event {
    /// Where it stands, in years: the middle of its month, or of its
    /// year when the month is not known.
    #[must_use]
    pub fn at(&self) -> f64 {
        let within = self.month.map_or(0.5, |m| (f64::from(m) - 0.5) / 12.0);
        f64::from(self.year) + within
    }

    /// Its callout's width and estimated height in pixels: room for the
    /// picture, if it has one, then a line of the date and title for
    /// every `per` characters of them and of the words for every `per`
    /// and a third more. On a phone (`narrow`) the boxes are narrower.
    /// The browser measures the real heights before the final layout.
    #[must_use]
    pub fn size(&self, narrow: bool) -> (f64, f64) {
        let pictured = self.image.is_some();
        let width = match (pictured, self.kind) {
            (true, _) => 300.0,
            (false, Kind::Note) => 190.0,
            (false, _) => 230.0,
        } * if narrow { 0.8 } else { 1.0 };
        let per = width / 9.0;
        #[allow(clippy::cast_precision_loss)]
        let lines = |text: &str, per: f64| (text.chars().count() as f64 / per).ceil();
        let title = lines(&format!("{}: {}", self.when, self.title), per);
        let words = lines(&self.text, per * 1.3);
        let topics = if self.topics.is_empty() { 0.0 } else { 1.0 };
        let picture = if pictured { width * 0.56 } else { 0.0 };
        (width, picture + 20.0 + 21.0 * title + 18.0 * (words + topics))
    }
}

/// The events in `json`, in time order (those at the same time in the
/// order written).
///
/// # Errors
/// When `json` is not a list of events.
pub fn events(json: &str) -> Result<Vec<Event>, serde_json::Error> {
    let mut list: Vec<Event> = serde_json::from_str(json)?;
    list.sort_by(|a, b| a.at().total_cmp(&b.at()));
    Ok(list)
}

/// What is wrong with `list`, one line each: a year outside the axis, a
/// month outside 1 to 12, a title or a `when` left empty, a `when` that
/// does not name its year, a demo without its slug or a slug on what is
/// not a demo, a paper with nowhere to lead.
#[must_use]
pub fn check(list: &[Event]) -> Vec<String> {
    let mut wrong = Vec::new();
    for e in list {
        let name = format!("{} {:?}", e.year, e.title);
        if !(FIRST..=NOW).contains(&e.year) {
            wrong.push(format!("{name}: the year is outside {FIRST}-{NOW}"));
        }
        if e.month.is_some_and(|m| !(1..=12).contains(&m)) {
            wrong.push(format!("{name}: the month is not 1 to 12"));
        }
        if e.title.trim().is_empty() || e.when.trim().is_empty() {
            wrong.push(format!("{name}: no title or no when"));
        }
        if !e.when.contains(&e.year.to_string()) {
            wrong.push(format!("{name}: when {:?} does not name the year", e.when));
        }
        if (e.kind == Kind::Demo) != e.demo.is_some() {
            wrong.push(format!("{name}: a demo needs its slug, and only a demo has one"));
        }
        if e.kind == Kind::Paper && e.href.is_none() {
            wrong.push(format!("{name}: a paper needs a link"));
        }
    }
    wrong
}
