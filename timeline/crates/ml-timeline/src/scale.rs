//! Years across the strip, each as wide as it needs. A year with events
//! is as wide as its callouts take, alternating above and below the
//! axis, and at least `MIN_YEAR`; a year with none is `EMPTY` wide; and a
//! run of `FOLD_RUN` or more empty years folds into one break `FOLD`
//! wide, so a quiet decade is not a blank screen. Within a segment the
//! scale is linear, so months fall evenly across their year, and a
//! place on the strip maps back to its year (`Scale::year_at`).

use crate::layout::GAP;

/// The first year on the axis: the McCulloch-Pitts neuron.
pub const FIRST: i32 = 1943;
/// The last year the axis marks.
pub const NOW: i32 = 2026;
/// The narrowest a year with an event is drawn.
pub const MIN_YEAR: f64 = 120.0;
/// How wide a year with no event is.
pub const EMPTY: f64 = 40.0;
/// How wide a folded run of empty years is.
pub const FOLD: f64 = 84.0;
/// How many empty years in a row fold into one break.
pub const FOLD_RUN: i32 = 3;

/// A stretch of the axis: the years `from` up to but not including
/// `to` (one year, or a folded run), where it starts across the strip,
/// how wide it is, and how many events fall in it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    pub from: i32,
    pub to: i32,
    pub left: f64,
    pub width: f64,
    pub events: usize,
}

impl Segment {
    /// Whether it is a folded run of empty years.
    #[must_use]
    pub fn folded(&self) -> bool {
        self.to - self.from > 1
    }

    /// Its right edge.
    #[must_use]
    pub fn right(&self) -> f64 {
        self.left + self.width
    }
}

/// The axis's segments, in order, and the margin before the first and
/// after the last.
#[derive(Debug, Clone, PartialEq)]
pub struct Scale {
    pub segments: Vec<Segment>,
    pub margin: f64,
}

impl Scale {
    /// The scale from `first` to `last` (both included) for events given
    /// as where each stands (`Event::at`) and its callout's width.
    #[must_use]
    pub fn new(first: i32, last: i32, events: &[(f64, f64)], margin: f64) -> Scale {
        let years = usize::try_from(last - first + 1).unwrap_or(0);
        let (mut need, mut count) = (vec![0.0; years], vec![0usize; years]);
        for &(at, width) in events {
            #[allow(clippy::cast_possible_truncation)]
            let year = (at.floor() as i32).clamp(first, last);
            let i = usize::try_from(year - first).unwrap_or(0);
            need[i] += f64::midpoint(width, GAP);
            count[i] += 1;
        }
        let mut segments = Vec::new();
        let mut left = margin;
        let mut push = |from: i32, to: i32, width: f64, events: usize| {
            segments.push(Segment { from, to, left, width, events });
            left += width;
        };
        let mut year = first;
        while year <= last {
            let i = usize::try_from(year - first).unwrap_or(0);
            if count[i] > 0 {
                push(year, year + 1, (need[i] + GAP).max(MIN_YEAR), count[i]);
                year += 1;
                continue;
            }
            let run = (year..=last)
                .take_while(|&y| count[usize::try_from(y - first).unwrap_or(0)] == 0)
                .count();
            let end = year + i32::try_from(run).unwrap_or(1);
            if end - year >= FOLD_RUN {
                push(year, end, FOLD, 0);
            } else {
                (year..end).for_each(|y| push(y, y + 1, EMPTY, 0));
            }
            year = end;
        }
        Scale { segments, margin }
    }

    /// The first year on the axis.
    #[must_use]
    pub fn first(&self) -> i32 {
        self.segments.first().map_or(FIRST, |s| s.from)
    }

    /// The last year on the axis.
    #[must_use]
    pub fn last(&self) -> i32 {
        self.segments.last().map_or(NOW, |s| s.to - 1)
    }

    /// The segment that holds `year`.
    #[must_use]
    pub fn segment(&self, year: i32) -> Option<&Segment> {
        self.segments.iter().find(|s| s.from <= year && year < s.to)
    }

    /// Where `at` (a year with its fraction) is from the strip's left
    /// edge; before the first year or after the last, at that end.
    #[must_use]
    pub fn x(&self, at: f64) -> f64 {
        let at = at.clamp(f64::from(self.first()), f64::from(self.last() + 1));
        let held = |s: &&Segment| f64::from(s.from) <= at && at < f64::from(s.to);
        match self.segments.iter().find(held).or(self.segments.last()) {
            Some(s) => s.left + (at - f64::from(s.from)) / f64::from(s.to - s.from) * s.width,
            None => self.margin,
        }
    }

    /// The year with its fraction at `x` across the strip: the inverse
    /// of `x`, held to the axis at either end.
    #[must_use]
    pub fn year_at(&self, x: f64) -> f64 {
        let Some(s) = self.segments.iter().find(|s| x < s.right()).or(self.segments.last()) else {
            return f64::from(FIRST);
        };
        let within = ((x - s.left) / s.width).clamp(0.0, 1.0);
        f64::from(s.from) + within * f64::from(s.to - s.from)
    }

    /// The whole strip's width.
    #[must_use]
    pub fn width(&self) -> f64 {
        self.segments.last().map_or(0.0, Segment::right) + self.margin
    }
}
