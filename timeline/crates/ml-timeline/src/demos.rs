//! This repository's demos on the timeline. A demo event in
//! `data/timeline.json` names only its slug, where it stands (at the
//! idea it shows, beside that paper's event) and what it shows; its
//! title, words, picture and link come from `demos/<slug>/demo.toml`,
//! so nothing is written twice.

use crate::event::{Event, Kind};

/// What the timeline takes from a demo's `demo.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Demo {
    pub slug: String,
    pub title: String,
    pub summary: String,
    /// `live` for a demo with a page.
    pub status: String,
}

impl Demo {
    #[must_use]
    pub fn live(&self) -> bool {
        self.status == "live"
    }

    /// The summary's first clause: up to its first colon or full stop.
    #[must_use]
    pub fn short(&self) -> String {
        let end = self.summary.find([':', '.']).unwrap_or(self.summary.len());
        let mut short = self.summary[..end].trim().to_string();
        short.push('.');
        short
    }
}

/// The demo in a `demo.toml`'s text: its top-level `slug`, `title`,
/// `summary` and `status` strings (each `key = "..."` on one line, as
/// those files write them), or nothing without a slug.
#[must_use]
pub fn demo(toml: &str) -> Option<Demo> {
    let mut d = Demo::default();
    for line in toml.lines().take_while(|l| !l.trim_start().starts_with('[')) {
        let Some((key, value)) = line.split_once('=') else { continue };
        let Some(value) = string(value) else { continue };
        match key.trim() {
            "slug" => d.slug = value,
            "title" => d.title = value,
            "summary" => d.summary = value,
            "status" => d.status = value,
            _ => {}
        }
    }
    (!d.slug.is_empty()).then_some(d)
}

/// A basic TOML string, `"..."` with `\"` and `\\` escaped.
fn string(text: &str) -> Option<String> {
    let mut chars = text.trim().strip_prefix('"')?.chars();
    let mut out = String::new();
    loop {
        match chars.next()? {
            '"' => return Some(out),
            '\\' => out.push(chars.next()?),
            c => out.push(c),
        }
    }
}

/// Fill each demo event from its demo: the title, the summary's first
/// clause, the summary whole as the picture's tip, the picture
/// (`shots/<slug>.webp`, made at build time) and the live page. What
/// is wrong, one line each: a slug with no demo, or one not live; a
/// demo event that writes what its demo says.
pub fn fill(list: &mut [Event], demos: &[Demo]) -> Vec<String> {
    let mut wrong = Vec::new();
    for e in list.iter_mut().filter(|e| e.kind == Kind::Demo) {
        let slug = e.demo.clone().unwrap_or_default();
        if !e.title.is_empty() || !e.text.is_empty() || e.href.is_some() || e.image.is_some() || e.tip.is_some() {
            wrong.push(format!("{} demo {slug}: its title, words, picture and link come from its demo.toml", e.year));
        }
        match demos.iter().find(|d| d.slug == slug) {
            Some(d) if d.live() => {
                e.title.clone_from(&d.title);
                e.text = d.short();
                e.tip = Some(d.summary.clone());
                e.image = Some(format!("shots/{slug}.webp"));
                e.href = Some(format!("../{slug}/"));
            }
            Some(_) => wrong.push(format!("{} demo {slug}: not live", e.year)),
            None => wrong.push(format!("{} demo {slug}: no such demo", e.year)),
        }
    }
    wrong
}

/// What is wrong between the demos and the timeline: a live demo with
/// no demo event, or with more than one.
#[must_use]
pub fn undemoed(list: &[Event], demos: &[Demo]) -> Vec<String> {
    let on = |d: &Demo| list.iter().filter(|e| e.demo.as_deref() == Some(d.slug.as_str())).count();
    demos
        .iter()
        .filter(|d| d.live())
        .filter_map(|d| match on(d) {
            1 => None,
            0 => Some(format!("demo {}: live, but not on the timeline", d.slug)),
            n => Some(format!("demo {}: on the timeline {n} times", d.slug)),
        })
        .collect()
}
