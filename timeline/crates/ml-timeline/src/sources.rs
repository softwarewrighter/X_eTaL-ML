//! Where each event comes from: `docs/ml-timeline-sources.md`, a table
//! with a row for every event but the demos (which are this
//! repository's own), naming its year, month and title as the event
//! does, and the link it was checked at.

use crate::event::{Event, Kind};

/// A row of the sources' table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source {
    pub year: i32,
    pub month: Option<u8>,
    pub title: String,
    /// What was checked there: the authors, the title, the venue.
    pub what: String,
    pub link: String,
}

/// The rows of the table in `markdown`: every line of the form `| year
/// | month or - | title | what | <link> |`, the header and its rule
/// left out.
#[must_use]
pub fn sources(markdown: &str) -> Vec<Source> {
    markdown.lines().filter_map(row).collect()
}

fn row(text: &str) -> Option<Source> {
    let cells: Vec<&str> = text.trim().strip_prefix('|')?.strip_suffix('|')?.split('|').map(str::trim).collect();
    let [year, month, title, what, link] = cells[..] else { return None };
    Some(Source {
        year: year.parse().ok()?,
        month: if month == "-" { None } else { Some(month.parse().ok()?) },
        title: title.to_string(),
        what: what.to_string(),
        link: link.trim_start_matches('<').trim_end_matches('>').to_string(),
    })
}

/// What is wrong between the events and their sources, one line each:
/// an event (not a demo) without a row of its year, month and title; a
/// row whose link is not a web address, or is not where the event
/// leads; a row with no event; a row twice.
#[must_use]
pub fn unsourced(list: &[Event], rows: &[Source]) -> Vec<String> {
    let mut wrong = Vec::new();
    let same = |e: &Event, s: &Source| e.year == s.year && e.month == s.month && e.title == s.title;
    for e in list.iter().filter(|e| e.kind != Kind::Demo) {
        let name = format!("{} {:?}", e.year, e.title);
        match rows.iter().find(|s| same(e, s)) {
            None => wrong.push(format!("{name}: no source of its year, month and title")),
            Some(s) if e.href.as_ref().is_some_and(|h| *h != s.link) => {
                wrong.push(format!("{name}: leads to {:?}, its source is {:?}", e.href.as_deref().unwrap_or(""), s.link));
            }
            Some(_) => {}
        }
    }
    for (i, s) in rows.iter().enumerate() {
        let name = format!("{} {:?}", s.year, s.title);
        if !s.link.starts_with("https://") || s.what.is_empty() {
            wrong.push(format!("{name}: a source needs what was checked and an https link"));
        }
        if !list.iter().any(|e| same(e, s)) {
            wrong.push(format!("{name}: a source with no event"));
        }
        if rows[..i].iter().any(|r| r.year == s.year && r.month == s.month && r.title == s.title) {
            wrong.push(format!("{name}: a source twice"));
        }
    }
    wrong
}
