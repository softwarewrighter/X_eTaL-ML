//! Where each callout goes (the Lisp timeline's layout, unchanged but
//! for the scale it reads). In time order, the callouts alternate above
//! and below the axis, every box against it, so none sticks out past
//! the rest. Across the strip each box goes as near its year as it can
//! while the boxes keep their years' order and those on a side keep
//! clear of each other: the least-squares fit under those constraints,
//! found exactly by pooling adjacent violators. A crowd therefore spreads
//! evenly about its years, the earlier boxes leaning left, their leaders
//! slanting right, and the later ones right.

use crate::scale::Scale;

/// Above the axis or below it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Above,
    Below,
}

/// A callout's box: its left edge and size, its side, how far it
/// stands from the axis, and `anchor`, where its year is on the axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placed {
    pub left: f64,
    pub width: f64,
    pub height: f64,
    pub side: Side,
    pub offset: f64,
    pub anchor: f64,
}

/// The space kept between two boxes on a side.
pub(crate) const GAP: f64 = 16.0;

/// Every callout, in time order: each given as where it stands (`Event::at`), its box's width
/// and height -- measured in the browser or, before that, estimated by
/// `Event::size` -- and the side it is pinned to, if any.
#[must_use]
pub fn place(items: &[(f64, (f64, f64), Option<Side>)], scale: &Scale) -> Vec<Placed> {
    let sides = alternate(items);
    let room = room(items, &sides);
    let wanted: Vec<f64> = items
        .iter()
        .zip(&room)
        .map(|(item, r)| scale.x(item.0) - r)
        .collect();
    let fitted = rising(&wanted);
    let at = |i: usize| {
        let (year, (width, height), _) = items[i];
        let left = fitted[i] + room[i] - width / 2.0;
        Placed {
            left,
            width,
            height,
            side: sides[i],
            offset: 0.0,
            anchor: scale.x(year),
        }
    };
    (0..items.len()).map(at).collect()
}

/// How far along the strip each box must be from the first, at least:
/// half the wider of two neighbors on opposite sides, which keeps every
/// other box -- the next on the same side -- clear; the whole of both
/// when a pin puts two neighbors on one side.
fn room(items: &[(f64, (f64, f64), Option<Side>)], sides: &[Side]) -> Vec<f64> {
    let mut room = vec![0.0];
    for (i, pair) in items.windows(2).enumerate() {
        let (a, b) = (pair[0].1.0, pair[1].1.0);
        let together = sides[i] == sides[i + 1];
        let needed = if together {
            f64::midpoint(a, b) + GAP
        } else {
            f64::midpoint(a.max(b), GAP)
        };
        room.push(room[i] + needed);
    }
    room
}

/// Each callout's side: its pin if it has one, else the side the one
/// before it did not take, starting above.
fn alternate(items: &[(f64, (f64, f64), Option<Side>)]) -> Vec<Side> {
    let mut sides: Vec<Side> = Vec::with_capacity(items.len());
    for item in items {
        let other = match sides.last() {
            Some(Side::Above) => Side::Below,
            _ => Side::Above,
        };
        sides.push(item.2.unwrap_or(other));
    }
    sides
}

/// The non-decreasing sequence nearest `wanted` in least squares: runs
/// that go down are pooled into their mean until none does.
fn rising(wanted: &[f64]) -> Vec<f64> {
    let mut runs: Vec<(f64, f64, usize)> = Vec::new();
    for &value in wanted {
        runs.push((value, 1.0, 1));
        while let [.., (low, a, n), (high, b, m)] = runs[..] {
            if low <= high {
                break;
            }
            runs.truncate(runs.len() - 2);
            runs.push(((low * a + high * b) / (a + b), a + b, n + m));
        }
    }
    runs.into_iter()
        .flat_map(|(mean, _, n)| std::iter::repeat_n(mean, n))
        .collect()
}
