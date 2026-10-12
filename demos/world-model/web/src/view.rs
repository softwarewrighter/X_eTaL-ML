//! The program the page shows (everything it runs) and the pictures:
//! the window's three frames, the predicted frame, the dream.

use microscope::color::Rgb;
use microscope::source::{between, block, Range};
use yew::Html;

use crate::micro::{begin_tail, head, step_tail, Place, State, SIDE};

/// What the page runs, all of it: the head, then the first run's lines
/// from the start state, then a step's lines from the window and the
/// true state the page holds.
pub fn program(start: State, window: &[Place; 3], truth: State, stepped: bool) -> String {
    let first = begin_tail(start);
    let run = match stepped {
        false => first,
        true => format!("{first}# -- each step: the window the page holds through the network, the world moved on --\n{}", step_tail(window, truth)),
    };
    format!("{}# -- the first run: the window from the start --\n{run}", head())
}

/// The step's prediction (or the first run's window).
pub fn range(src: &str, stepped: bool) -> Range {
    match stepped {
        true => between(src, "pr := ", "pr := "),
        false => between(src, "w0 := ", "w0 := "),
    }
}

pub fn source(start: State, window: &[Place; 3], truth: State, stepped: bool) -> Html {
    let src = program(start, window, truth, stepped);
    block(&src, range(&src, stepped))
}

/// A cell is CELL pixels; a frame is PX by PX.
pub const CELL: usize = 10;
pub const PX: usize = CELL * SIDE;
/// The three frames side by side, a gap between.
pub const GAP: usize = 6;
pub const WIDE: usize = 3 * PX + 2 * GAP;

const INK: Rgb = [30, 30, 30];
const ORANGE: Rgb = [255, 122, 0];
const GRAY: Rgb = [150, 150, 150];
const GRID: Rgb = [235, 235, 235];

fn put(px: &mut [u8], cols: usize, r: i64, c: i64, rgb: Rgb) {
    let rows = px.len() / (4 * cols);
    if (0..rows as i64).contains(&r) && (0..cols as i64).contains(&c) {
        let i = 4 * (r as usize * cols + c as usize);
        px[i..i + 3].copy_from_slice(&rgb);
    }
}

/// The ball at (x, y) filled, offset by `dx` pixels.
fn ball(px: &mut [u8], cols: usize, (x, y): Place, dx: usize, rgb: Rgb) {
    for r in 0..2 * CELL {
        for c in 0..2 * CELL {
            put(px, cols, (y as usize * CELL + r) as i64, (dx + x as usize * CELL + c) as i64, rgb);
        }
    }
}

/// The ball's outline, 2 pixels wide.
fn outline(px: &mut [u8], cols: usize, (x, y): Place, dx: usize, rgb: Rgb) {
    let (top, left, n) = ((y as usize * CELL) as i64, (dx + x as usize * CELL) as i64, (2 * CELL) as i64);
    for i in 0..n {
        for w in 0..2 {
            put(px, cols, top + w, left + i, rgb);
            put(px, cols, top + n - 1 - w, left + i, rgb);
            put(px, cols, top + i, left + w, rgb);
            put(px, cols, top + i, left + n - 1 - w, rgb);
        }
    }
}

fn grid(px: &mut [u8], cols: usize, dx: usize) {
    for i in 0..=SIDE {
        for j in 0..PX {
            put(px, cols, (i * CELL).min(PX - 1) as i64, (dx + j) as i64, GRID);
            put(px, cols, j as i64, (dx + (i * CELL).min(PX - 1)) as i64, GRID);
        }
    }
}

/// The window: three frames side by side, the ball in each.
pub fn frames(window: &[Place; 3]) -> Vec<u8> {
    let mut px = vec![255u8; 4 * PX * WIDE];
    for (i, &p) in window.iter().enumerate() {
        let dx = i * (PX + GAP);
        grid(&mut px, WIDE, dx);
        ball(&mut px, WIDE, p, dx, INK);
    }
    px
}

/// The predicted frame: each pixel as dark as its probability; the true
/// ball outlined in orange.
pub fn predicted(probs: &[f64], truth: State) -> Vec<u8> {
    let mut px = vec![255u8; 4 * PX * PX];
    grid(&mut px, PX, 0);
    for (i, &p) in probs.iter().enumerate() {
        let (r0, c0) = ((i / SIDE) * CELL, (i % SIDE) * CELL);
        let v = (255.0 * (1.0 - p.clamp(0.0, 1.0))) as u8;
        if p > 0.02 {
            for r in 1..CELL {
                for c in 1..CELL {
                    put(&mut px, PX, (r0 + r) as i64, (c0 + c) as i64, [v, v, v]);
                }
            }
        }
    }
    outline(&mut px, PX, (truth[0], truth[1]), 0, ORANGE);
    px
}

/// The dream: the true ball gray, the dreamed ball's outline orange,
/// their paths as dots at the balls' centers (the latest largest).
pub fn dream(dreamed: &[Place], real: &[Place]) -> Vec<u8> {
    let mut px = vec![255u8; 4 * PX * PX];
    grid(&mut px, PX, 0);
    let dot = |px: &mut [u8], (x, y): Place, rgb: Rgb, off: i64| {
        let (cr, cc) = ((y as usize * CELL + CELL) as i64, (x as usize * CELL + CELL) as i64);
        for dr in -1..=1 {
            for dc in -1..=1 {
                put(px, PX, cr + dr + off, cc + dc + off, rgb);
            }
        }
    };
    for &p in real.iter().take(real.len().saturating_sub(1)) {
        dot(&mut px, p, GRAY, -2);
    }
    for &p in dreamed.iter().take(dreamed.len().saturating_sub(1)) {
        dot(&mut px, p, ORANGE, 2);
    }
    if let Some(&p) = real.last() {
        ball(&mut px, PX, p, 0, GRAY);
    }
    if let Some(&p) = dreamed.last() {
        outline(&mut px, PX, p, 0, ORANGE);
    }
    px
}
